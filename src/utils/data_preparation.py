from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple, List, Optional

import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import classification_report, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle as sk_shuffle
from joblib import dump

from typing import TypedDict

class EvalMetrics(TypedDict):
    accuracy: float
    f1_macro: float
    report: str

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


# ---------- Image Preprocessing ----------
def find_images(root: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def to_grayscale(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img= ImageOps.grayscale(img)
    return img

def binarize(img: Image.Image, threshold: Optional[int] = None) -> Image.Image:
    # Optional: einfache globale Schwelle
    if threshold is None:
        arr = np.asarray(img, dtype=np.uint8)
        thr = np.percentile(arr, 50) 
        threshold = int(thr)
    return img.point(lambda x: 255 if x > threshold else 0, mode="1").convert("L")

def crop_to_content(img: Image.Image, margin: int = 2) -> Image.Image:
    arr = np.asarray(img.convert("L"))
    # „Nicht weiß“ konservativ: alles < 250
    mask = arr < 250
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, arr.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, arr.shape[1])
    # Falls das Fenster zu dünn ist, nichts machen
    if (y1 - y0) < 3 or (x1 - x0) < 3:
        return img
    return img.crop((x0, y0, x1, y1))

def deskew(img: Image.Image) -> Image.Image:
    # Deskew via Zentralmomente (leichte Korrektur)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.mean(arr, axis=2)
    w = 255.0 - arr
    w[w < 0] = 0.0

    m = w.sum()
    if m < 1e-6:
        return img  # leeres Bild, nichts zu tun

    yy, xx = np.indices(w.shape)  # yy: Zeilen (y), xx: Spalten (x)

    cy = (yy * w).sum() / m
    cx = (xx * w).sum() / m

    y = yy - cy
    x = xx - cx

    mu11 = (x * y * w).sum() / m
    mu20 = ((x ** 2) * w).sum() / m
    mu02 = ((y ** 2) * w).sum() / m
    
    denom = mu20 + mu02
    if denom < 1e-6:
        return img
    shear = float(mu11 / denom)
    # Affine Korrektur in x-Richtung
    
    w_img, h_img = img.size
    a, b, d, e = 1.0, shear, 0.0, 1.0
    c = -shear * (h_img / 2.0)  # hält das Zentrum halbwegs fix
    f = 0.0

    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BILINEAR,
    )

def preprocess_image(path: Path, size=(40, 40), do_binarize=False, do_deskew=True) -> np.ndarray:
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

# -------------- Alternatives Image-Preprocessing kürzer -------

# def preprocess_image(path: Path, size=(32, 32)) -> np.ndarray:
#     img = Image.open(path).convert("L")
#     img = ImageOps.invert(img) if np.mean(img) > 127 else img
#     img = ImageOps.pad(img, size, method=Image.BILINEAR, color=255, centering=(0.5, 0.5))
#     arr = np.asarray(img, dtype=np.float32) / 255.0
#     return arr

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

# ---------- 4) Dataset Utilities ----------

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
        
def iter_minibatches_frames(frame: pd.DataFrame, batch_size: int, random_state=0) -> Iterator[List[Path]]:
    rng = np.random.RandomState(random_state)
    idx = np.arange(frame.shape[0])#rows
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [frame.iloc[j] for j in idx[i:i+batch_size]]


# ---------- Training ----------
### Hardcoded 'modified_huber' optional VALID_LOSSES: set[str] = {   
# "hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"}


def create_pipeline( random_state=0) -> Pipeline:
    clf = SGDClassifier(
        loss='modified_huber',
        alpha=1e-4,
        learning_rate="optimal",
        penalty="l2",
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=random_state,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
def partial_fit_chars(
    image_paths: List[Path],
    *,
    batch_size: int = 1024,
    random_state: int = 0,
    model_out: Optional[Path] = None,
    existing_model: Optional[Path] = None,
) -> Tuple[Pipeline, LabelEncoder]:
    # Neues Modell oder bestehendes weitertrainieren
    labels = [parse_label_from_filename(p) for p in image_paths]
    le = LabelEncoder().fit(labels)
    classes = np.arange(len(le.classes_))

    if existing_model and Path(existing_model).exists():
        data = np.load(existing_model)
        pipe = data["pipeline"]
        le = data["label_encoder"]
        print(f"Modell geladen aus {existing_model}")
    else:
        pipe = create_pipeline(random_state=random_state)
        print("Neues Modell initialisiert.")

    for batch in iter_minibatches(image_paths, batch_size=batch_size, random_state=random_state):
        Xb = [hog_features(preprocess_image(p)) for p in batch]
        yb = le.transform([parse_label_from_filename(p) for p in batch])
        Xb = np.vstack(Xb)
        pipe.named_steps["scaler"].partial_fit(Xb)
        Xb_s = pipe.named_steps["scaler"].transform(Xb)
        pipe.named_steps["clf"].partial_fit(Xb_s, yb, classes=classes)


    if model_out:
        model_out = Path(model_out)
        model_out.parent.mkdir(parents=True, exist_ok=True)  # <— das fehlte
        dump({"pipeline": pipe, "label_encoder": le}, model_out)
        print(f"Modell gespeichert unter {model_out.resolve()}")


    return pipe, le

def partial_fit_chars_from_df(
    df: pd.DataFrame,
    *,
    label_column:str='!',
    batch_size: int = 1024,
    random_state: int = 0,
    model_out: Optional[Path] = None,
    existing_model: Optional[Path] = None,
) -> Tuple[Pipeline, LabelEncoder]:
    # Neues Modell oder bestehendes weitertrainieren
    labels = df[label_column].unique()
    df=df.drop(label_column)
    if any(df[df>1]):
        df =df.astype(int)/255.0
    le = LabelEncoder().fit(labels)
    classes = np.arange(len(le.classes_))

    if existing_model and Path(existing_model).exists():
        data = np.load(existing_model)
        pipe = data["pipeline"]
        le = data["label_encoder"]
        print(f"Modell geladen aus {existing_model}")
    else:
        pipe = create_pipeline(random_state=random_state)
        print("Neues Modell initialisiert.")
    try:
            
        for batch in iter_minibatches_frames(pd.DataFrame(df.iloc[:, 0]), batch_size=batch_size, random_state=random_state):
            Xb = [p for p in batch]
            yb = le.transform(labels.iloc[p] for p in batch)
            Xb = np.vstack(Xb)
            pipe.named_steps["scaler"].partial_fit(Xb)
            Xb_s = pipe.named_steps["scaler"].transform(Xb)
            pipe.named_steps["clf"].partial_fit(Xb_s, yb, classes=classes)


        if model_out:
            model_out = Path(model_out)
            model_out.parent.mkdir(parents=True, exist_ok=True)  # <— das fehlte
            dump({"pipeline": pipe, "label_encoder": le}, model_out)
            print(f"Modell gespeichert unter {model_out.resolve()}")
    except Exception as ex:
        print(ex)

    return pipe, le
# ---------- Evaluation ----------

def evaluate(pipe: Pipeline, le: LabelEncoder, image_paths: List[Path]) ->EvalMetrics:
    X = [hog_features(preprocess_image(p)) for p in image_paths]
    y = [parse_label_from_filename(p) for p in image_paths]
    X = np.vstack(X)
    y = le.transform(y)
    Xs = pipe.named_steps["scaler"].transform(X)
    y_pred = pipe.named_steps["clf"].predict(Xs)
  
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "report": str(classification_report(y, y_pred, digits=3, zero_division=0)),
    }

# ---------- Utility ----------

def train_test_split_paths(paths: List[Path], test_size=0.2, random_state=42):
    labels = [parse_label_from_filename(p) for p in paths]
    tr, te = train_test_split(paths, test_size=test_size, stratify=labels, random_state=random_state)
    return tr, te
# ---------- 6) Bequemer CLI-Helper ----------

def prepare_npz_from_directory(data_dir: Path, out_npz: Path) -> Path:
    paths = find_images(data_dir)
    if not paths:
        raise RuntimeError(f"Keine Bilder unter {data_dir}")
    X, y, classes = build_dataset(paths)
    save_npz(out_npz, X, y, classes)
    return out_npz