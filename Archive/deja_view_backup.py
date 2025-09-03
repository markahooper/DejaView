#!/usr/bin/env python3
import io, os, zipfile, re
import numpy as np
import streamlit as st
import pandas as pd

# Try to import OpenCV; show a helpful message if missing (e.g., on Streamlit Cloud)
try:
    import cv2
except ModuleNotFoundError:
    st.error(
        "OpenCV (cv2) is not installed. On Streamlit Cloud, add `opencv-python-headless` to your "
        "`requirements.txt` and pin Python to 3.11.\n"
        "Example requirements:\n"
        "streamlit==1.36.0\nopencv-python-headless>=4.12.0.88,<5\nnumpy==1.26.4\npandas==2.2.2\n"
    )
    st.stop()

# -------------------- Small compatibility helpers --------------------
def image_compat(img, **kwargs):
    # Render image with compatibility across Streamlit versions.
    try:
        return st.image(img, **kwargs)
    except TypeError:
        if "use_container_width" in kwargs:
            k = dict(kwargs)
            k["use_column_width"] = k.pop("use_container_width")
            return st.image(img, **k)
        return st.image(img)

def dataframe_compat(df, **kwargs):
    try:
        return st.dataframe(df, **kwargs)
    except TypeError:
        k = dict(kwargs)
        k.pop("use_container_width", None)
        return st.dataframe(df, **k)

# -------------------- Core utilities --------------------
def imread_rgb_from_bytes(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imencode_png(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("PNG encoding failed.")
    return buf.tobytes()

def rotate_bound(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH))

def gen_variants(rgb, try_flips=True, try_rot=True, extra_angles=None):
    outs = [("orig", rgb)]
    if try_rot:
        outs += [("rot90", np.rot90(rgb,1)),
                 ("rot180", np.rot90(rgb,2)),
                 ("rot270", np.rot90(rgb,3))]
    if extra_angles:
        for ang in extra_angles:
            a = float(ang) % 360.0
            if a in (0.0, 90.0, 180.0, 270.0):
                continue
            outs.append((f"rot{int(round(a))}", rotate_bound(rgb, a)))
    if try_flips:
        outs += [("flip_h", np.flip(rgb, axis=1))]
    return outs

def build_text_suppression_mask(rgb):
    # Build a mask (255 = allowed, 0 = suppressed) for text-like regions using MSER.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    mser = cv2.MSER_create()
    try:
        mser.setMinArea(60)
        mser.setMaxArea(int(0.06 * H * W))
    except Exception:
        pass
    regions, _ = mser.detectRegions(gray)
    text_mask = np.zeros_like(gray, dtype=np.uint8)
    img_area = H * W
    for pts in regions:
        if len(pts) < 6:
            continue
        hull = cv2.convexHull(pts.reshape(-1,1,2))
        area = cv2.contourArea(hull)
        if area < 30 or area > 0.08 * img_area:
            continue
        x,y,w,h = cv2.boundingRect(hull)
        if w < 6 or h < 6:
            continue
        ar = w / float(h)
        if ar < 0.15 or ar > 25.0:
            continue
        cv2.drawContours(text_mask, [hull], -1, 255, -1)
    k = max(3, (H + W) // 400)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    return cv2.bitwise_not(text_mask)

def orb_match_details(A, B, ratio=0.78, nfeatures=2000, fastThreshold=10, suppress_text=True):
    # Return (inliers, H, good, kpa, kpb, mask) with optional text suppression masks.
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=fastThreshold, edgeThreshold=15, patchSize=31)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    gA = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    gB = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
    maskA = build_text_suppression_mask(A) if suppress_text else None
    maskB = build_text_suppression_mask(B) if suppress_text else None
    kpa, desA = orb.detectAndCompute(gA, maskA)
    kpb, desB = orb.detectAndCompute(gB, maskB)
    if desA is None or desB is None or len(kpa) < 8 or len(kpb) < 8:
        return 0, None, [], kpa, kpb, None
    m = bf.knnMatch(desA, desB, k=2)
    good = [a for a,b in m if a.distance < ratio*b.distance]
    if len(good) < 8:
        return len(good), None, good, kpa, kpb, None
    src = np.float32([kpa[x.queryIdx].pt for x in good]).reshape(-1,1,2)
    dst = np.float32([kpb[x.trainIdx].pt for x in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers, H, good, kpa, kpb, mask

def draw_inlier_matches(A, B, kpa, kpb, good, mask, max_draw=120):
    if mask is None or len(good) == 0:
        draw = good[:max_draw]
    else:
        mask_list = mask.ravel().tolist()
        draw = [g for g, m in zip(good, mask_list) if m]
        draw = draw[:max_draw]
    A_bgr = cv2.cvtColor(A, cv2.COLOR_RGB2BGR)
    B_bgr = cv2.cvtColor(B, cv2.COLOR_RGB2BGR)
    vis = cv2.drawMatches(A_bgr, kpa, B_bgr, kpb, draw, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def color_match_to(A, B):
    A32 = A.astype(np.float32); B32 = B.astype(np.float32)
    for c in range(3):
        ma, sa = A32[...,c].mean(), A32[...,c].std()+1e-6
        mb, sb = B32[...,c].mean(), B32[...,c].std()+1e-6
        B32[...,c] = (B32[...,c]-mb)*(sa/sb)+ma
    return np.clip(B32, 0, 255).astype(np.uint8)

def make_overlay_hull(A, B, H, inlier_ptsA, edge_color=(0,255,0), fill_color=(0,255,0), alpha=0.25, thickness=3):
    Ha, Wa = A.shape[:2]
    warpedB = cv2.warpPerspective(B, H, (Wa, Ha))
    adjB = color_match_to(A, warpedB)
    blend = cv2.addWeighted(A, 0.65, adjB, 0.35, 0)
    if inlier_ptsA is not None and len(inlier_ptsA) >= 3:
        pts = np.array(inlier_ptsA, dtype=np.float32).reshape(-1,2)
        hull = cv2.convexHull(pts.astype(np.float32))
        hull_int = hull.astype(np.int32)
        overlay = blend.copy()
        cv2.fillPoly(overlay, [hull_int], fill_color)
        blend = cv2.addWeighted(overlay, alpha, blend, 1 - alpha, 0)
        cv2.polylines(blend, [hull_int], isClosed=True, color=edge_color, thickness=thickness, lineType=cv2.LINE_AA)
    return blend

def equal_grid_split(rgb, rows, cols, trim_frac=0.03, header_frac=0.0, target_max=900):
    H, W = rgb.shape[:2]
    y0 = int(H * header_frac)
    crop = rgb[y0:H, :].copy()
    Ch, Cw = crop.shape[:2]
    panels = []
    for r in range(rows):
        for c in range(cols):
            y1 = int(r * Ch/rows); y2 = int((r+1) * Ch/rows)
            x1 = int(c * Cw/cols); x2 = int((c+1) * Cw/cols)
            my = int(trim_frac*(y2-y1)); mx = int(trim_frac*(x2-x1))
            y1 += my; y2 -= my; x1 += mx; x2 -= mx
            p = crop[y1:y2, x1:x2].copy()
            ph, pw = p.shape[:2]
            scale = float(target_max) / max(ph, pw)
            if scale < 1.0:
                p = cv2.resize(p, (int(pw*scale), int(ph*scale)), interpolation=cv2.INTER_AREA)
            panels.append(p)
    return panels

def auto_split_panels(rgb, skip_top=0.05, min_area_frac=0.015, max_area_frac=0.7, target_max=900):
    H, W = rgb.shape[:2]
    start_y = int(H * skip_top)
    work = rgb[start_y:H, :].copy()
    Gh, Gw = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    k = max(7, int(0.004 * max(Gh, Gw)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dil = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_area = Gh * Gw
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        frac = area / img_area
        if frac < min_area_frac or frac > max_area_frac:
            continue
        roi = mask[y:y+h, x:x+w]
        density = roi.mean()/255.0
        if density < 0.05:
            continue
        boxes.append((start_y + y, x, start_y + y + h, x + w))

    if not boxes:
        return []

    def contains(a,b):
        return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]
    def iou(a,b):
        ax1, ay1, ax2, ay2 = a[1], a[0], a[3], a[2]
        bx1, by1, bx2, by2 = b[1], b[0], b[3], b[2]
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
        inter = iw*ih
        area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
        union = area_a + area_b - inter + 1e-6
        return inter/union

    merged = True
    while merged:
        merged = False
        out = []
        skip = set()
        for i in range(len(boxes)):
            if i in skip: continue
            a = boxes[i]
            ay0, ax0, ay1, ax1 = a
            for j in range(i+1, len(boxes)):
                if j in skip: continue
                b = boxes[j]
                if iou(a,b) > 0.25 or contains(a,b) or contains(b,a):
                    ay0 = min(ay0, b[0]); ax0 = min(ax0, b[1])
                    ay1 = max(ay1, b[2]); ax1 = max(ax1, b[3])
                    skip.add(j); merged = True
            out.append((ay0, ax0, ay1, ax1))
        boxes = out

    def sort_key(b):
        y0, x0, y1, x1 = b
        cy = (y0 + y1) / 2; cx = (x0 + x1) / 2
        return (round(cy / 50.0), cx)
    boxes = sorted(boxes, key=sort_key)

    panels = []
    for (y0, x0, y1, x1) in boxes:
        p = rgb[y0:y1, x0:x1].copy()
        ph, pw = p.shape[:2]
        scale = float(target_max) / max(ph, pw)
        if scale < 1.0:
            p = cv2.resize(p, (int(pw*scale), int(ph*scale)), interpolation=cv2.INTER_AREA)
        panels.append(p)

    return panels

def list_sample_images():
    regex = re.compile(r'(?i)^Image([1-9]|1[0-2])\.png$')  # Image1.png .. Image12.png
    items = []
    for f in os.listdir('.'):
        m = regex.match(f)
        if m:
            try:
                n = int(m.group(1))
            except:
                n = 999
            items.append((n, f))
    items.sort(key=lambda x: x[0])
    return [f for (_, f) in items]

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Deja View", layout="wide")

# Global CSS: Poppins font + colour scheme
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
:root {
  --ink: #57554e;
  --accent: #b5b1ed;
  --sun: #fdc474;
}
html, body, [class*="css"]  {
  font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  color: var(--ink);
}
h1, h2, h3, h4 { font-weight: 600; letter-spacing: 0.2px; color: var(--ink); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.card {
  background: #ffffff;
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.06);
  border: 1px solid rgba(0,0,0,0.05);
  margin-bottom: 16px;
  border-left: 6px solid var(--accent);
}
.header-accent { height: 6px; background: linear-gradient(90deg, var(--accent), var(--sun)); border-radius: 999px; margin: 6px 0 14px 0; }
.stButton>button, .stDownloadButton>button {
  background: var(--accent);
  color: #1b1b1b;
  border: 1px solid var(--ink);
  border-radius: 12px;
  padding: 0.6rem 1rem;
}
.stButton>button:hover, .stDownloadButton>button:hover { background: var(--sun); }
[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.06); }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
/* Bigger label for sample checkbox */
.sample-label { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.25rem; color: var(--ink); }
.thumb { border: 1px solid rgba(0,0,0,0.08); border-radius: 10px; padding: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Deja View")
st.caption("By Mark Hooper · Detects potential overlaps between scientific figure panels and visualizes inlier keypoint matches + matched-region polygon.")
st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Composite (auto-split)", "Composite (rows × cols)", "Upload panels"])
    min_inliers = st.number_input("Min inliers to keep", min_value=5, max_value=500, value=35, step=5)
    max_size = st.number_input("Downscale largest side (px)", min_value=300, max_value=3000, value=900, step=50)
    try_rot = st.checkbox("Try rotations (90/180/270°)", value=True)
    try_flip = st.checkbox("Try horizontal flip", value=True)
    fine_rot = st.checkbox("Enable fine rotation sweep", value=False)
    if fine_rot:
        angle_step = st.selectbox("Angle step (°)", [5, 10, 15, 30], index=2)
    else:
        angle_step = None
    suppress_text = st.checkbox("Suppress text labels (MSER)", value=True, help="Reduces matches on figure labels by ignoring text-like regions during keypoint detection.")

    if mode == "Composite (auto-split)":
        skip_top = st.slider("Skip top fraction", 0.0, 0.3, 0.05, 0.01)
        min_area_frac = st.slider("Min panel area fraction", 0.001, 0.2, 0.015, 0.001)
        max_area_frac = st.slider("Max panel area fraction", 0.1, 0.95, 0.7, 0.05)
    elif mode == "Composite (rows × cols)":
        rows = st.number_input("Rows", min_value=1, max_value=10, value=2, step=1)
        cols = st.number_input("Cols", min_value=1, max_value=10, value=3, step=1)

    st.markdown("---")
    st.subheader("Samples")
    sample_files = list_sample_images()
    if sample_files:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sample_files:
                with open(f, "rb") as fh:
                    z.writestr(os.path.basename(f), fh.read())
        st.download_button("Download sample images to try", data=buf.getvalue(), file_name="deja_view_samples.zip")
    else:
        st.caption("No sample images found in the app folder.")

# --------------- Inputs (with sample picker + thumbnail preview) ---------------
sample_choice_comp = None
sample_choice_panels = None

if mode.startswith("Composite"):
    comp_file = st.file_uploader("Upload composite image", type=["png","jpg","jpeg","tif","tiff","bmp","webp"])
    cols_upload = st.columns([1, 2, 2])
    with cols_upload[0]:
        st.markdown('<div class="sample-label">Use one of our sample images to test</div>', unsafe_allow_html=True)
        use_sample_comp = st.checkbox("", value=False, key="use_sample_comp", label_visibility="collapsed")
    with cols_upload[1]:
        if st.session_state.get('use_sample_comp'):
            options = sample_files
            if options:
                sample_choice_comp = st.selectbox("Select sample", options, key="sample_comp_select")
            else:
                st.caption("No samples found")
    with cols_upload[2]:
        if sample_choice_comp:
            try:
                with open(sample_choice_comp, "rb") as fh:
                    image_compat(fh.read(), caption=sample_choice_comp, use_container_width=True)
            except Exception:
                pass
else:
    panel_files = st.file_uploader("Upload panel images (2 or more)", type=["png","jpg","jpeg","tif","tiff","bmp","webp"], accept_multiple_files=True)
    cols_upload = st.columns([1, 2, 2])
    with cols_upload[0]:
        st.markdown('<div class="sample-label">Use one of our sample images to test</div>', unsafe_allow_html=True)
        use_sample_pan = st.checkbox("", value=False, key="use_sample_pan", label_visibility="collapsed")
    with cols_upload[1]:
        if st.session_state.get('use_sample_pan'):
            options = sample_files
            if options:
                sample_choice_panels = st.selectbox("Select sample", options, key="sample_pan_select")
            else:
                st.caption("No samples found")
    with cols_upload[2]:
        if sample_choice_panels:
            try:
                with open(sample_choice_panels, "rb") as fh:
                    image_compat(fh.read(), caption=sample_choice_panels, use_container_width=True)
            except Exception:
                pass

run = st.button("Run detection", type="primary")

# -------------------- Run pipeline --------------------
if run:
    try:
        input_preview = None
        if mode == "Composite (auto-split)":
            if sample_choice_comp:
                with open(sample_choice_comp, "rb") as fh:
                    data = fh.read()
            else:
                if not comp_file: st.stop()
                data = comp_file.read()
            input_preview = data
            rgb = imread_rgb_from_bytes(data)
            panels = auto_split_panels(rgb, skip_top=skip_top,
                                       min_area_frac=min_area_frac, max_area_frac=max_area_frac,
                                       target_max=max_size)
            if len(panels) < 2:
                st.error("Auto-split failed to find at least two panels. Try adjusting 'skip top' and area sliders.")
                st.stop()

        elif mode == "Composite (rows × cols)":
            if sample_choice_comp:
                with open(sample_choice_comp, "rb") as fh:
                    data = fh.read()
            else:
                if not comp_file: st.stop()
                data = comp_file.read()
            input_preview = data
            rgb = imread_rgb_from_bytes(data)
            panels = equal_grid_split(rgb, int(rows), int(cols), header_frac=0.0, target_max=max_size)
            if len(panels) < 2:
                st.error("Grid split produced fewer than two panels — check rows/cols.")
                st.stop()

        else:
            if sample_choice_panels:
                with open(sample_choice_panels, "rb") as fh:
                    data = fh.read()
                rgb = imread_rgb_from_bytes(data)
                panels = auto_split_panels(rgb, skip_top=0.05, min_area_frac=0.015, max_area_frac=0.7, target_max=max_size)
                input_preview = data  # composite preview in expander
                if len(panels) < 2:
                    st.error("Sample image did not split into at least two panels. Try a different sample.")
                    st.stop()
            else:
                if not panel_files or len(panel_files) < 2:
                    st.error("Please upload at least two panel images or choose a sample.")
                    st.stop()
                panels = []
                thumbs = []
                for f in panel_files:
                    b = f.read()
                    arr = imread_rgb_from_bytes(b)
                    ph, pw = arr.shape[:2]
                    scale = float(max_size) / max(ph, pw)
                    if scale < 1.0:
                        arr = cv2.resize(arr, (int(pw*scale), int(ph*scale)), interpolation=cv2.INTER_AREA)
                    panels.append(arr)
                    thumbs.append(arr)
                input_preview = thumbs

        # Pairwise matching
        rows_data = []
        views = []  # (i, j, matches_png, overlay_png, inliers, variant)
        n = len(panels)

        # Create progress bar (handle older Streamlit without 'text')
        try:
            prog = st.progress(0.0, text="Matching...")
        except TypeError:
            prog = st.progress(0.0)

        total = n*(n-1)//2
        done = 0
        for i in range(n):
            for j in range(i+1, n):
                A = panels[i]
                best = {"inliers":0, "H":None, "variant":"orig", "good":None, "kpa":None, "kpb":None, "mask":None, "Bimg":None}
                angle_step_val = angle_step if angle_step else 0
                extra_angles = (list(range(angle_step_val, 360, angle_step_val)) if angle_step_val and angle_step_val > 0 else None)
                for name, Bv in gen_variants(panels[j], try_flips=try_flip, try_rot=try_rot, extra_angles=extra_angles):
                    inl, Hm, good, kpa, kpb, mask = orb_match_details(A, Bv, ratio=0.78, nfeatures=2000, fastThreshold=10, suppress_text=suppress_text)
                    if inl > best["inliers"]:
                        best = {"inliers": inl, "H": Hm, "variant": name, "good": good, "kpa": kpa, "kpb": kpb, "mask": mask, "Bimg": Bv}
                rows_data.append({"pair": f"{i+1}-{j+1}", "inliers": int(best["inliers"]), "best_variant_B": best["variant"]})
                if best["H"] is not None and best["inliers"] >= min_inliers:
                    inlier_ptsA = None
                    if best["mask"] is not None and len(best["good"]) > 0:
                        mlist = best["mask"].ravel().tolist()
                        inlier_ptsA = [best["kpa"][g.queryIdx].pt for g, m in zip(best["good"], mlist) if m]
                    matches_rgb = draw_inlier_matches(A, best["Bimg"], best["kpa"], best["kpb"], best["good"], best["mask"])
                    matches_png = imencode_png(matches_rgb)
                    overlay_rgb = make_overlay_hull(A, best["Bimg"], best["H"], inlier_ptsA)
                    overlay_png = imencode_png(overlay_rgb)
                    views.append((i+1, j+1, matches_png, overlay_png, int(best["inliers"]), best["variant"]))
                done += 1
                try:
                    prog.progress(done/total, text=f"Matching {done}/{total}")
                except TypeError:
                    prog.progress(done/total)

        df = pd.DataFrame(rows_data).sort_values("inliers", ascending=False).reset_index(drop=True)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Candidate overlaps")
            dataframe_compat(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if views:
            st.subheader("Matches & Matched Region")
            for (i, j, matches_png, overlay_png, inl, name) in views:
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**Panels {i}–{j}** · transform `{name}` · inliers **{inl}**")
                    image_compat(matches_png, caption="Inlier keypoint matches", use_container_width=True)
                    image_compat(overlay_png, caption="Overlay with matched region (convex hull) on Panel A", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("summary.csv", df.to_csv(index=False))
                for (i, j, matches_png, overlay_png, inl, name) in views:
                    z.writestr(f"pair_{i}_{j}_{name}_inl{inl}_matches.png", matches_png)
                    z.writestr(f"pair_{i}_{j}_{name}_inl{inl}_overlay_hull.png", overlay_png)
            st.download_button("Download results (ZIP)", data=zip_buf.getvalue(), file_name="deja_view_results.zip")

        else:
            st.info("No pairs met the inlier threshold. Try lowering 'Min inliers to keep' or increasing 'Downscale largest side'.")

        # --- Original upload at the bottom inside an expander ---
        with st.expander("Show original upload", expanded=False):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Original upload")
            if isinstance(input_preview, list):
                image_compat(input_preview, caption=[f"Panel {i+1}" for i in range(len(input_preview))], use_container_width=True)
            else:
                image_compat(input_preview, caption="Composite image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.exception(e)
