#!/usr/bin/env python3
"""
image_diagnostics.py
--------------------
Produce a compact diagnostics JSON bundle for an input image so an LLM can decide
which restoration tools to run (denoise, deblur, super-resolution, colorization, etc.).

Dependencies (install as needed):
    pip install pillow opencv-python imagehash piexif numpy scikit-image langdetect

Optional:
    - pytesseract (plus system Tesseract) for OCR text extraction
    - pybrisque (and libsvm model) for BRISQUE; or piq/niqe alternatives
Notes:
    The script degrades gracefully if optional libs are missing.
"""
import argparse, json, math, hashlib, io, os, sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

import numpy as np

import cv2
from PIL import Image, ImageOps, ImageCms, ExifTags
import piexif
import imagehash
import pytesseract
from langdetect import detect as lang_detect
from skimage.transform import radon
from tqdm import tqdm

repo_root = Path(__file__).parent.parent
catpllmdir = repo_root / "catp_base"
if str(catpllmdir) not in sys.path:
    sys.path.insert(0, str(catpllmdir))
from src.config import GlobalTaskConfig
default_test_seq_tasks = GlobalTaskConfig.default_test_seq_tasks
default_test_nonseq_tasks = GlobalTaskConfig.default_test_nonseq_tasks

def fail(msg):
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(2)

def load_image(path: Path):
    if Image is None:
        fail("Pillow is required. Install with: pip install pillow")
    im = Image.open(path)
    return im

def image_bytes_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get_icc_summary(pil_img) -> Optional[Dict[str, Any]]:
    icc_bytes = pil_img.info.get("icc_profile")
    if not icc_bytes:
        return None
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        desc = ImageCms.getProfileName(prof)
    except Exception:
        desc = None
    icc_sha = hashlib.sha256(icc_bytes).hexdigest()
    return {"profile_name": desc, "icc_sha256": icc_sha}

def get_exif_dict(pil_img) -> Dict[str, Any]:
    try:
        exif_raw = pil_img.getexif()
        if not exif_raw:
            return {}
        exif = {}
        for k, v in exif_raw.items():
            tag = ExifTags.TAGS.get(k, str(k))
            # rational to float
            if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, int) for x in v) and v[1] != 0:
                try:
                    v = v[0] / v[1]
                except Exception:
                    pass
            exif[tag] = v
        return exif
    except Exception:
        return {}

def variance_of_laplacian(gray: np.ndarray) -> float:
    if cv2 is None:
        return float('nan')
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def tenengrad(gray: np.ndarray) -> float:
    if cv2 is None:
        return float('nan')
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g2 = gx**2 + gy**2
    return float(np.mean(g2))

def noise_sigma(gray: np.ndarray) -> float:
    if cv2 is None:
        return float('nan')
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    resid = gray.astype(np.float32) - blur.astype(np.float32)
    return float(resid.std())

def entropy_bits(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum() or 1.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def exposure_stats(gray: np.ndarray) -> Dict[str, float]:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum() or 1.0
    mean = float((hist * np.arange(256)).sum() / total)
    var = float(((np.arange(256) - mean) ** 2 * hist).sum() / total)
    std = math.sqrt(var)
    clipped_shadows = float(hist[0] / total * 100.0)
    clipped_highlights = float(hist[-1] / total * 100.0)
    ent = entropy_bits(gray)
    return {
        "mean_luma_0_255": round(mean, 3),
        "stdev_luma": round(std, 3),
        "percent_clipped_shadows": round(clipped_shadows, 3),
        "percent_clipped_highlights": round(clipped_highlights, 3),
        "entropy_bits": round(ent, 3),
    }

def hasler_susstrunk_colorfulness(rgb: np.ndarray) -> float:
    img = rgb.astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    rg_mean, rg_std = rg.mean(), rg.std()
    yb_mean, yb_std = yb.mean(), yb.std()
    return float(np.sqrt(rg_std**2 + yb_std**2) + 0.3*np.sqrt(rg_mean**2 + yb_mean**2))

def is_grayscale(rgb: np.ndarray, tol: float = 1.0) -> bool:
    # If channels are nearly equal for most pixels
    diff_rg = np.abs(rgb[:,:,0].astype(np.int16) - rgb[:,:,1].astype(np.int16))
    diff_gb = np.abs(rgb[:,:,1].astype(np.int16) - rgb[:,:,2].astype(np.int16))
    ratio_close = np.mean((diff_rg <= tol) & (diff_gb <= tol))
    return bool(ratio_close > 0.98)

def jpeg_quant_hint(pil_img) -> Optional[Dict[str, Any]]:
    try:
        qtables = getattr(pil_img, "quantization", None)
        if not qtables:
            return None
        avgs = [sum(tbl)/len(tbl) for tbl in qtables.values()]
        hint = float(sum(avgs)/len(avgs))
        return {"qtables_count": len(qtables), "avg_table_value": round(hint, 2)}
    except Exception:
        return None

def blockiness_score(gray: np.ndarray) -> float:
    # measure average gradient change at 8x8 boundaries vs inside blocks
    if cv2 is None:
        return float('nan')
    h, w = gray.shape
    # vertical boundaries every 8 columns (excluding border)
    vert_lines = np.arange(8, w-1, 8)
    horiz_lines = np.arange(8, h-1, 8)
    # gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx*gx + gy*gy)
    # boundary samples
    vb = g[:, vert_lines].mean() if vert_lines.size > 0 else 0.0
    hb = g[horiz_lines, :].mean() if horiz_lines.size > 0 else 0.0
    # interior (exclude a 1-pixel band around boundaries)
    mask = np.ones_like(g, dtype=bool)
    if vert_lines.size > 0:
        mask[:, vert_lines] = False
    if horiz_lines.size > 0:
        mask[horiz_lines, :] = False
    interior = g[mask].mean() if mask.any() else 0.0
    # Higher boundary vs interior implies blockiness
    if interior <= 1e-6:
        return float(0.0)
    return float(max(0.0, ((vb + hb)/2.0 - interior) / (interior + 1e-6)))

def ringing_score(gray: np.ndarray) -> float:
    # crude: energy in high-frequency band of radial spectrum near edges
    f = np.fft.fft2(gray.astype(np.float32))
    F = np.fft.fftshift(f)
    mag = np.log1p(np.abs(F))
    h, w = mag.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy)**2 + (X - cx)**2)
    r1, r2 = min(h, w)*0.30/2, min(h, w)*0.48/2
    band = (R >= r1) & (R <= r2)
    outer = (R > r2)
    band_energy = mag[band].mean() if band.any() else 0.0
    outer_energy = mag[outer].mean() if outer.any() else 1.0
    return float(band_energy / (outer_energy + 1e-6))

def estimate_motion_blur(gray: np.ndarray) -> Dict[str, Any]:
    # Heuristic: use radon transform of log spectrum to find a dominant blur angle
    if radon is None:
        return {"angle_deg": None, "length_px": None, "confidence": 0.0, "method": "radon_spectrum", "note": "skimage unavailable"}
    g = gray.astype(np.float32)
    g = g - g.mean()
    G = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(G))
    # Normalize
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    theta = np.linspace(0., 180., max(gray.shape)//2, endpoint=False)
    sinogram = radon(mag, theta=theta, circle=False)
    proj = sinogram.mean(axis=0)  # average over radius
    angle_idx = int(np.argmax(proj))
    angle_deg = float(theta[angle_idx])
    # crude confidence: peak prominence
    prom = float((proj[angle_idx] - np.median(proj)) / (np.std(proj) + 1e-6))
    # length heuristic: stronger low-freq suppression -> longer blur. Use variance of Laplacian inverse.
    vol = variance_of_laplacian(gray) if cv2 is not None else float('nan')
    if np.isnan(vol) or vol <= 0:
        length_px = None
    else:
        length_px = float(np.clip(50.0 / (vol**0.5 + 1e-6), 0, 50))
    conf = float(np.clip(prom / 5.0, 0.0, 1.0))
    return {"angle_deg": round(angle_deg, 2), "length_px": None if length_px is None else round(length_px, 2), "confidence": conf, "method": "radon_spectrum"}

def face_sizes(rgb: np.ndarray) -> List[Dict[str, Any]]:
    if cv2 is None:
        return []
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24,24))
        out = []
        for (x,y,w,h) in faces:
            # approximate inter-ocular distance ~ 0.46 * face width (heuristic)
            out.append({"bbox":[int(x),int(y),int(w),int(h)], "interocular_px": round(0.46*w, 2)})
        return out
    except Exception:
        return []

def text_region_stats(gray: np.ndarray) -> Dict[str, Any]:
    # Use MSER to guess if text-like regions present and estimate avg component height
    if cv2 is None:
        return {"present": None, "avg_char_px_height": None, "component_count": None}
    try:
        mser = cv2.MSER_create(_min_area=30, _max_area=5000)
        regions, _ = mser.detectRegions(gray)
        heights = []
        for pts in regions:
            x,y,w,h = cv2.boundingRect(pts.reshape(-1,1,2))
            if w>5 and h>5 and h<gray.shape[0]*0.5:
                heights.append(h)
        if len(heights)==0:
            return {"present": False, "avg_char_px_height": None, "component_count": 0}
        avg_h = float(np.median(heights))
        return {"present": True, "avg_char_px_height": round(avg_h,2), "component_count": int(len(heights))}
    except Exception:
        return {"present": None, "avg_char_px_height": None, "component_count": None}

def ocr_text(pil_img) -> Dict[str, Any]:
    if pytesseract is None:
        return {"text": None, "lang": None, "confidence": None, "note": "pytesseract not installed"}
    try:
        # pytesseract image should be RGB
        txt = pytesseract.image_to_string(pil_img)
        txt = txt.strip()
        if not txt:
            return {"text": "", "lang": None, "confidence": None}
        lang = None
        if lang_detect is not None:
            try:
                lang = lang_detect(txt)
            except Exception:
                lang = None
        return {"text": txt, "lang": lang, "confidence": None}
    except Exception as e:
        return {"text": None, "lang": None, "confidence": None, "error": str(e)}

def aspect_ratio_str(w: int, h: int) -> str:
    from math import gcd
    g = gcd(w, h) if (w>0 and h>0) else 1
    return f"{w//g}:{h//g}"

def perceptual_hashes(pil_img) -> Dict[str, Any]:
    if imagehash is None:
        return {"phash": None, "dhash": None, "note": "imagehash not installed"}
    try:
        rgb = pil_img.convert("RGB")
        ph = imagehash.phash(rgb)
        dh = imagehash.dhash(rgb)
        return {"phash": str(ph), "dhash": str(dh)}
    except Exception:
        return {"phash": None, "dhash": None}

def build_bundle(image_path: Path) -> Dict[str, Any]:
    pil = load_image(image_path)
    icc = get_icc_summary(pil)
    exif = get_exif_dict(pil)
    dpi = pil.info.get("dpi")
    mime = Image.MIME.get(pil.format, None)
    width, height = pil.size

    # Normalized arrays
    pil_exif_oriented = ImageOps.exif_transpose(pil)
    rgb_np = np.array(pil_exif_oriented.convert("RGB"))
    gray = (cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY) if cv2 is not None
            else np.dot(rgb_np[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8))

    # Metrics
    vol = variance_of_laplacian(gray)
    tgrad = tenengrad(gray)
    noise = noise_sigma(gray)
    colorfulness = hasler_susstrunk_colorfulness(rgb_np)
    exp = exposure_stats(gray)
    blocky = blockiness_score(gray)
    ring = ringing_score(gray)
    motion = estimate_motion_blur(gray)
    faces = face_sizes(rgb_np)
    textstats = text_region_stats(gray)
    ocr = ocr_text(pil_exif_oriented)
    hashes = perceptual_hashes(pil_exif_oriented)
    qhint = jpeg_quant_hint(pil)
    gray_flag = is_grayscale(rgb_np)

    sha256 = image_bytes_sha256(image_path)
    ar = aspect_ratio_str(width, height)

    # Try to pick exposure-capture items
    iso = exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity")
    exposure_time = exif.get("ExposureTime")
    fnumber = exif.get("FNumber")
    focal = exif.get("FocalLength")

    bundle = {
        "source": {
            "filename": image_path.name,
            "sha256": sha256,
            "format": pil.format,
            "mime": mime,
            "byte_size": image_path.stat().st_size
        },
        "resolution": {
            "width_px": width,
            "height_px": height,
            "aspect_ratio": ar,
            "dpi": dpi[0] if isinstance(dpi, tuple) else dpi
        },
        "color": {
            "is_grayscale": gray_flag,
            "icc_profile": (icc or {}).get("profile_name"),
            "icc_sha256": (icc or {}).get("icc_sha256"),
            "colorfulness": round(colorfulness, 3),
            "bit_depth": 8
        },
        "quality": {
            "blur_vol": None if math.isnan(vol) else round(vol, 3),
            "tenengrad": None if math.isnan(tgrad) else round(tgrad, 3),
            "motion_blur": motion,
            "noise_sigma_luma": None if math.isnan(noise) else round(noise, 3),
            "banding_score": None,  # TODO: more robust banding metric if needed
            "blockiness": round(blocky, 4),
            "ringing": round(ring, 4),
            "brisque": None,  # left None unless you plug in pybrisque/niqe
            "percent_clipped_shadows": exp["percent_clipped_shadows"],
            "percent_clipped_highlights": exp["percent_clipped_highlights"],
            "entropy_bits": exp["entropy_bits"]
        },
        "content": {
            "faces": faces,
            "text": textstats,
            "objects_summary": {"count": None, "smallest_obj_px": None},  # placeholder if you add an OD model
            "scene_type": "unknown"
        },
        "ocr": ocr,
        "compression": {
            "format": "JPEG" if (mime == "image/jpeg") else pil.format,
            "jpeg_quant_hint": qhint
        },
        "exposure_capture": {
            "iso": iso,
            "shutter_s": float(exposure_time) if isinstance(exposure_time, (int, float)) else exposure_time,
            "aperture_f": float(fnumber) if isinstance(fnumber, (int, float)) else fnumber,
            "focal_length_mm": float(focal) if isinstance(focal, (int, float)) else focal
        },
        "identity": hashes,
        "pipeline": {
            "created_utc": None,
            "tools": [
                {"name": "pillow", "version": getattr(Image, "__version__", "unknown")},
                {"name": "opencv-python", "version": getattr(cv2, "__version__", "not_installed")},
                {"name": "imagehash", "version": getattr(imagehash, "__version__", "not_installed")},
            ]
        }
    }

    # Add normalized blur in [0,1] for quick thresholding (cap tuned at 250; adjust per pipeline)
    vol_val = bundle["quality"]["blur_vol"]
    if vol_val is not None:
        bundle["quality"]["blur_normalized_0_1"] = round(min(vol_val/250.0, 1.0), 3)
    else:
        bundle["quality"]["blur_normalized_0_1"] = None

    return bundle

def iter_image_files(path: Path, allowed_task_ids: Optional[Set[int]] = None):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    if path.is_file():
        if path.suffix.lower() in exts:
            yield path
        return
    for p in path.rglob("*"):
        if "outputs" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in exts:
            if allowed_task_ids is not None:
                parts = p.parts
                try:
                    img_idx = len(parts) - 1 - parts[::-1].index("images")
                    if img_idx - 2 < 0 or parts[img_idx - 1] != "inputs":
                        continue
                    tid = int(parts[img_idx - 2])
                except Exception:
                    continue
                if tid not in allowed_task_ids:
                    continue
            yield p

def compute_out_path(img_path: Path, args) -> Path:
    # Priority: --out (explicit file) > --out-dir (mirrored structure) > default alongside image
    if args.out:
        return Path(args.out)
    if args.out_dir:
        # Determine in_root for relative path
        in_root = Path(args.in_root) if args.in_root else (Path(args.image) if Path(args.image).is_dir() else img_path.parent)
        try:
            rel_dir = img_path.parent.relative_to(in_root)
        except Exception:
            # If not under in_root, just use the image's parent name
            rel_dir = img_path.parent.name
        rel_dir = Path(rel_dir) if isinstance(rel_dir, (str,)) else rel_dir
        out_dir = Path(args.out_dir) / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / (img_path.stem + ".diagnostics.json")
    # Default next to image
    return img_path.with_suffix(".diagnostics.json")

def is_valid_existing_json(path: Path) -> bool:
    try:
        if (not path.exists()) or path.stat().st_size <= 2:
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return len(data) > 0
        return bool(data)
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Compute diagnostics bundle for image(s) (LLM planning input).")
    parser.add_argument("image", help="Path to input image OR directory of images")
    parser.add_argument("--out", "-o", help="Path to write a single JSON (only valid for single-file input)")
    parser.add_argument("--out-dir", help="Directory to write outputs, preserving input folder structure under --in-root (or input dir).")
    parser.add_argument("--in-root", help="When using --out-dir, mirror paths relative to this directory. Defaults to the provided input directory; for single files, defaults to the image's parent.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--only-test", action="store_true", help="Process only test tasks from config (filters by task IDs)")
    parser.add_argument("--test-type", choices=["seq", "nonseq", "both"], default="both", help="Which test set list to use when --only-test is set")
    args = parser.parse_args()

    in_path = Path(args.image)
    if not in_path.exists():
        fail(f"Input not found: {in_path}")

    # Validate --out usage
    if in_path.is_dir() and args.out:
        fail("--out cannot be used when the input is a directory. Use --out-dir instead.")

    wrote = []

    allowed_tasks: Optional[Set[int]] = None
    if args.only_test:
        if args.test_type == "seq":
            allowed_tasks = set(int(x) for x in default_test_seq_tasks)
        elif args.test_type == "nonseq":
            allowed_tasks = set(int(x) for x in default_test_nonseq_tasks)
        else:
            allowed_tasks = set(int(x) for x in default_test_seq_tasks) | set(int(x) for x in default_test_nonseq_tasks)

    if in_path.is_file():
        iterable = tqdm([in_path], total=1, desc="Images", unit="img") if (tqdm is not None) else [in_path]
        for img in iterable:
            out_path = compute_out_path(img, args)
            if is_valid_existing_json(out_path):
                continue
            bundle = build_bundle(img)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2 if args.pretty else None)
            wrote.append(out_path)
    else:
        if not args.out_dir:
            fail("When input is a directory, please provide --out-dir to control output location.")
        images = list(iter_image_files(in_path, allowed_tasks))
        iterator = tqdm(images, desc="Images", unit="img") if (tqdm is not None) else images
        for img in iterator:
            try:
                out_path = compute_out_path(img, args)
                if is_valid_existing_json(out_path):
                    continue
                bundle = build_bundle(img)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(bundle, f, indent=2 if args.pretty else None)
                wrote.append(out_path)
            except Exception as e:
                print(f"[warn] Failed on {img}: {e}", file=sys.stderr)

    # Print newline-separated paths for easy piping
    for p in wrote:
        print(str(p))


if __name__ == "__main__":
    main()
