from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, Iterator, Tuple, List, Optional

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle as sk_shuffle
from joblib import dump


# ---------- 1) Datenerfassung und Label-Parsing ----------
registry = {}
LABEL_PATTERN = re.compile(r"^(?P<source>.+)_(?P<char>[A-Za-z0-9])_(?P<version>\d+)(?:\.[A-Za-z0-9]+)?$")
CHAR ='char'
SOURCE='source'
VERSION ='version'

def parse_label_from_filename(p: Path,pattern:re.Pattern=LABEL_PATTERN) -> str:
    m = LABEL_PATTERN.match(p.stem)
    if not m:
        raise ValueError(f"Kein Label im Dateinamen gefunden: {p.name}")
    info = m.groupdict()
    return str(info[CHAR])


def find_images(root: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def to_grayscale(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        return ImageOps.grayscale(img)
    return img

def binarize(img: Image.Image, threshold: Optional[int] = None) -> Image.Image:
    # Optional: einfache globale Schwelle
    if threshold is None:
        arr = np.asarray(img, dtype=np.uint8)
        thr = np.percentile(arr, 50) 
        threshold = int(thr)
    return img.point(lambda x: 255 if x > threshold else 0, mode="1").convert("L")

def crop_to_content(img: Image.Image, margin: int = 2) -> Image.Image:
    # Zuschneiden auf Bounding Box der Nicht-weiß-Pixel
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    mask = arr < 250  # nicht weiß
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, arr.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, arr.shape[1])
    return img.crop((x0, y0, x1, y1))

def deskew(img: Image.Image) -> Image.Image:
    # Deskew via Zentralmomente (leichte Korrektur)
    arr = np.asarray(img, dtype=np.float32)
    arr = 255 - arr  # Vordergrund hell
    m = arr.sum()
    if m == 0:
        return img
    cy, cx = np.array(np.indices(arr.shape)) @ (arr.reshape(-1, 1)) / m
    y, x = np.indices(arr.shape)
    y = y - cy
    x = x - cx
    mu11 = (x * y * arr).sum() / m
    mu20 = ((x ** 2) * arr).sum() / m
    mu02 = ((y ** 2) * arr).sum() / m
    denom = mu20 + mu02
    if denom < 1e-6:
        return img
    skew = mu11 / denom
    # Affine Korrektur in x-Richtung
    matrix = (1, skew, -skew * img.size[1] / 2, 0, 1, 0)
    return img.transform(img.size, Image.Transform.AFFINE, matrix, resample=Image.Resampling.BILINEAR)

def preprocess_image(path: Path, size=(32, 32), do_binarize=False, do_deskew=True) -> np.ndarray:
    img = Image.open(path)
    img = to_grayscale(img)
    img = crop_to_content(img)
    if do_deskew:
        img = deskew(img)
    img = ImageOps.pad(img, size, method=Image.Resampling.BILINEAR, color=255, centering=(0.5, 0.5))
    if do_binarize:
        img = binarize(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

# ---------- 3) HOG-Features ----------

def hog_features(img_arr: np.ndarray) -> np.ndarray:
    feats = hog(
        img_arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        transform_sqrt=True,
        feature_vector=True,
    )
    return feats.astype(np.float32, copy=False)

# ---------- 4) Dataset Builder ----------

def build_dataset(image_paths: List[Path], *, shuffle=True, random_state=0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels : list[str]= [parse_label_from_filename(p) for p in image_paths]
    if shuffle:        
        image_paths, labels =sk_shuffle(image_paths, labels, random_state=random_state)        # type: ignore
    X = []
    for p in image_paths:
        arr = preprocess_image(p)
        X.append(hog_features(arr))
    X = np.vstack(X)
    y = np.array(labels)
    classes = sorted(set(labels))
    return X, y, classes

def save_npz(out_path: Path, X: np.ndarray, y: np.ndarray, classes: List[str]) -> None:
    np.savez_compressed(out_path, X=X, y=y, classes=np.array(classes, dtype=object))

# ---------- 5) Inkrementelles Training ----------

def iter_minibatches(paths: List[Path], batch_size: int, random_state=0) -> Iterator[List[Path]]:
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [paths[j] for j in idx[i:i+batch_size]]

def partial_fit_chars(
    image_paths: List[Path],
    *,
    batch_size: int = 1024,
    random_state: int = 0,
    model_out: Optional[Path] = None,
) -> Pipeline:
    # Klassen stabilisieren
    labels = [parse_label_from_filename(p) for p in image_paths]
    le = LabelEncoder().fit(labels)
    classes = le.classes_

    clf = SGDClassifier(
        loss="modified_huber",
        alpha=1e-4,
        learning_rate="optimal",
        penalty="l2",
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=random_state,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),  # auf HOG-Vektoren
        ("clf", clf),
    ])

    # Einmaliger "kalter" partial_fit mit Klassenliste
    # Wir füttern in Batches, um RAM zu schonen
    for batch in iter_minibatches(image_paths, batch_size=batch_size, random_state=random_state):
        Xb = []
        yb = []
        for p in batch:
            arr = preprocess_image(p)
            Xb.append(hog_features(arr))
            yb.append(parse_label_from_filename(p))
        Xb = np.vstack(Xb)
        yb = le.transform(np.array(yb))

        # scaler inkrementell updaten
        pipe.named_steps["scaler"].partial_fit(Xb)
        Xb_s = pipe.named_steps["scaler"].transform(Xb)
        pipe.named_steps["clf"].partial_fit(Xb_s, yb, classes=np.arange(len(classes)))

    # Optional abspeichern
    if model_out:
        dump({"pipeline": pipe, "label_encoder": le}, model_out)

    return pipe

# ---------- 6) Bequemer CLI-Helper ----------

def prepare_npz_from_directory(data_dir: Path, out_npz: Path) -> Path:
    paths = find_images(data_dir)
    if not paths:
        raise RuntimeError(f"Keine Bilder unter {data_dir}")
    X, y, classes = build_dataset(paths)
    save_npz(out_npz, X, y, classes)
    return out_npz