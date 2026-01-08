import os
import re
import csv
from typing import Dict, List, Tuple, Optional


def _guess_captions_file(data_dir: str) -> str:
    """
    Try common Flickr30k caption filenames.
    """
    candidates = [
        "captions.txt",
        "captions.token",
        "results_20130124.token",  # common in Flickr30k
    ]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Cannot find captions file in {data_dir}. "
        f"Please put one of: {', '.join(candidates)}"
    )


def load_flickr30k(
    data_dir: str,
    captions_file: Optional[str] = None,
    images_subdir: str = "images",
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Returns:
      docs:     List of captions (one per line/document)
      doc2img:  List of image filenames aligned with docs
      img2path: Mapping image filename -> absolute path
    Supports common formats:
      1) image.jpg<TAB>caption text...
      2) image.jpg#0<TAB>caption text...
      3) image.jpg#0 caption text...
      4) image.jpg|caption text...
    """
    if captions_file is None:
        captions_path = _guess_captions_file(data_dir)
    else:
        captions_path = captions_file if os.path.isabs(captions_file) else os.path.join(data_dir, captions_file)

    images_dir = os.path.join(data_dir, images_subdir)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    docs: List[str] = []
    doc2img: List[str] = []

    with open(captions_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            # Skip header line: image_name,comment_number,comment
            if row[0].strip().lower() == "image_name":
                continue

            # Expect: image_name, comment_number, comment
            if len(row) >= 3:
                img_name = row[0].strip()
                cap = ",".join(row[2:]).strip()  # caption may contain commas
                if img_name and cap:
                    docs.append(cap)
                    doc2img.append(img_name)
                continue

            # Fallback (if some weird line appears)
            line = ",".join(row).strip()
            if not line:
                continue

    # Build image path map
    img2path: Dict[str, str] = {}
    # We don't want to scan all files if not necessary, but it is ok for Flickr30k size.
    for fname in os.listdir(images_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            img2path[fname] = os.path.join(images_dir, fname)

    return docs, doc2img, img2path


_WORD_RE = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> List[str]:
    """
    Lowercase alphanumeric tokenizer for BM25 baseline.
    """
    return _WORD_RE.findall(text.lower())
