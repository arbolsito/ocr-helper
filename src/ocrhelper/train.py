# src/ocrhelper/train.py
from __future__ import annotations
import os, re, json, math, pathlib, time
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import click
import numpy as np
import joblib
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import filters, measure, morphology, util

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle as sk_shuffle


# =========================
# Utility
# =========================

def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def compile_regex(regex: str) -> re.Pattern:
    return re.compile(regex)

def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


# =========================
# HOG Feature-Extractor (32x32)
# =========================

@dataclass
class HogCfg:
    size: Tuple[int, int] = (32, 32)
    ppc: Tuple[int, int] = (8, 8)         # pixels_per_cell
    cpb: Tuple[int, int] = (2, 2)         # cells_per_block

def hog32(img_gray: np.ndarray, cfg: HogCfg) -> np.ndarray:
    img = resize(img_gray, cfg.size, anti_aliasing=True)
    feat = hog(img, pixels_per_cell=cfg.ppc, cells_per_block=cfg.cpb, feature_vector=True)
    return feat.astype(np.float32, copy=False)

# =========================
# Daten: Zeichen (data_digits/0..9)
# =========================

def iter_digit_paths(data_dir: pathlib.Path) -> Iterable[Tuple[int, pathlib.Path]]:
    for d in range(10):
        cls_dir = data_dir / str(d)
        if cls_dir.exists():
            for p in cls_dir.glob("*.png"):
                yield d, p

def load_digits_matrix(data_dir: pathlib.Path, cfg: HogCfg, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    pairs = list(iter_digit_paths(data_dir))
    if not pairs:
        raise RuntimeError(f"Keine Ziffernbilder unter {data_dir} gefunden (erwartet Ordner 0..9).")
    if limit:
        pairs = pairs[:limit]
    X, y = [], []
    for label, path in tqdm(pairs, desc="HOG(Zeichen)", unit="img"):
        g = imread(path, as_gray=True)
        X.append(hog32(g, cfg))
        y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

# =========================
# Sequenz-Features (für Pattern-Kalibrator)
#  - simple Segmentierung + Klassifikation mit Zeichenmodell
# =========================

def preprocess_gray(image: np.ndarray) -> np.ndarray:
    # erwartet Graustufen [0..1] oder [0..255]
    g = util.img_as_float(image) if image.dtype != np.float32 else image
    # adaptiver Schwellenwert
    thr = filters.threshold_local(g, block_size=31, offset=0.02)
    bw = (g < thr).astype(np.uint8)  # Ziffern dunkel -> True
    bw = morphology.remove_small_objects(measure.label(bw), min_size=16)
    bw = (bw > 0).astype(np.uint8)
    return bw

def segment_glyphs(gray_u8: np.ndarray) -> List[np.ndarray]:
    """Segmentiert verbundene Komponenten, gibt sortierte Patches als Graubilder zurück."""
    # Erwartet Graustufen uint8 [0..255]
    g = util.img_as_float(gray_u8)
    bw = preprocess_gray(g)
    lab = measure.label(bw, connectivity=2)
    props = measure.regionprops(lab)
    boxes = []
    for pr in props:
        y, x, h, w = pr.bbox[0], pr.bbox[1], pr.bbox[2]-pr.bbox[0], pr.bbox[3]-pr.bbox[1]
        if h < 10 or w < 5:
            continue
        ar = h / max(w, 1)
        if 0.6 <= ar <= 6.5:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    patches = []
    for x, y, w, h in boxes:
        crop = gray_u8[y:y+h, x:x+w]
        # Padding und Normierung
        pad = 2
        crop = util.pad(crop, ((pad, pad), (pad, pad)), mode="constant", constant_values=255)
        crop = resize(crop, (32, 32), anti_aliasing=True)
        patches.append(util.img_as_float32(crop))
    return patches

def predict_sequence(img_path: pathlib.Path, char_clf: SGDClassifier, cfg: HogCfg) -> Tuple[str, List[float]]:
    """Liest ein Bild, segmentiert, klassifiziert jeden Patch mit char_clf.
       Rückgabe: Sequenz-String und Margin-Liste (decision_function Abstand)."""
    raw = imread(img_path, as_gray=True)
    # normalisiere zu uint8 0..255
    if raw.dtype != np.uint8:
        g = util.img_as_ubyte(raw)
    else:
        g = raw
    patches = segment_glyphs(g)
    if not patches:
        return "", []
    X = np.stack([hog32(p, cfg) for p in patches], axis=0)
    y_scores = char_clf.decision_function(X)  # shape [N, 10]
    y_pred = np.argmax(y_scores, axis=1)
    # Margin = top1 - top2
    sorted_scores = np.sort(y_scores, axis=1)
    margins = (sorted_scores[:, -1] - sorted_scores[:, -2]).tolist()
    seq = "".join(str(int(d)) for d in y_pred.tolist())
    return seq, margins

def seq_features(seq: str, margins: List[float], regex: re.Pattern) -> np.ndarray:
    """Baut einfache Feature-Vektoren für den Pattern-Kalibrator."""
    L = len(seq)
    if margins:
        m_mean = float(np.mean(margins))
        m_min = float(np.min(margins))
        m_std = float(np.std(margins))
        frac_strong = float(np.mean(np.array(margins) > 0.8))
    else:
        m_mean = m_min = m_std = 0.0
        frac_strong = 0.0
    # erwartete Länge aus Regex grob schätzen (nur für \d{a,b})
    exp_lo, exp_hi = None, None
    m = re.search(r"\\d\{(\d+),(\d+)\}", regex.pattern)
    if m:
        exp_lo, exp_hi = int(m.group(1)), int(m.group(2))
    len_dev = 0.0
    if exp_lo is not None and exp_hi is not None:
        if L < exp_lo:
            len_dev = float(exp_lo - L)
        elif L > exp_hi:
            len_dev = float(L - exp_hi)
        else:
            len_dev = 0.0
    # regex match indicator
    re_ok = 1.0 if regex.fullmatch(seq) or (regex.search(seq) is not None) else 0.0
    return np.array([L, m_mean, m_min, m_std, frac_strong, len_dev, re_ok], dtype=np.float32)


# =========================
# CLI
# =========================

@click.group(help="Training CLI für OCR: Zeichen-Basis und Pattern-Kalibrator")
def cli():
    pass


# ---------- A) Zeichen-Basis (inkrementell, SGD hinge) ----------

@cli.command("train-chars")
@click.option("--ask-path", is_flag=True, help="Beim Start nach data_digits Pfad fragen.")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="Pfad zu data_digits/ (Ordner mit 0..9).")
@click.option("--models-dir", type=click.Path(file_okay=False), default="models", show_default=True)
@click.option("--epochs", type=int, default=10, show_default=True)
@click.option("--batch", type=int, default=2048, show_default=True)
@click.option("--val-split", type=float, default=0.15, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--continue-from", "continue_from", type=click.Path(exists=True), default=None,
              help="Bestehendes SGD-Modell (.joblib) für Fine-Tuning.")
@click.option("--limit", type=int, default=None, help="Nur erste N Bilder laden (Debug).")
@click.option("--regex", default=r"\d{17,21}", show_default=True,
              help="Nur als Metadatum gespeichert.")
def train_chars(ask_path, data_dir, models_dir, epochs, batch, val_split, seed, continue_from, limit, regex):
    # Pfad wählen
    if data_dir is None:
        if ask_path:
            data_dir = click.prompt("Pfad zu data_digits", type=click.Path(exists=True, file_okay=False))
        else:
            data_dir = "data_digits"
    data_dir = pathlib.Path(data_dir)
    models_dir = pathlib.Path(models_dir); ensure_dir(models_dir)

    cfg = HogCfg()
    X, y = load_digits_matrix(data_dir, cfg, limit=limit)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=val_split, stratify=y, random_state=seed)

    if continue_from:
        bundle = joblib.load(continue_from)
        clf: SGDClassifier = bundle["clf"]
        click.echo(f"🔁 Fine-Tuning von {continue_from}")
    else:
        clf = SGDClassifier(loss="hinge", alpha=1e-4, learning_rate="optimal",
                            random_state=seed, warm_start=True)

        # Initialisierung mit Klassenangabe
        classes = np.arange(10, dtype=np.int64)
        init = min(len(Xtr), batch)
        clf.partial_fit(Xtr[:init], ytr[:init], classes=classes)

    for ep in range(1, epochs + 1):
        Xsh, ysh = sk_shuffle(Xtr, ytr, random_state=seed + ep)
        for i in tqdm(range(0, len(Xsh), batch), desc=f"Chars Epoch {ep}/{epochs}", unit="batch"):
            xb, yb = Xsh[i:i+batch], ysh[i:i+batch]
            clf.partial_fit(xb, yb)
        val = accuracy_score(yte, clf.predict(Xte))
        click.echo(f"  → val_acc={val:.4f}")

    ypred = clf.predict(Xte)
    click.echo("\n📊 Klassifikationsreport (Zeichen):")
    click.echo(classification_report(yte, ypred, digits=4))

    out_path = models_dir / f"char_sgd_{now_tag()}.joblib"
    joblib.dump({"clf": clf, "regex": regex, "hog_cfg": cfg.__dict__}, out_path)
    click.echo(f"💾 Gespeichert: {out_path}")


# ---------- B) Pattern-Kalibrator (binär, inkrementell) ----------

@cli.command("train-pattern")
@click.option("--pattern-name", required=True, help="Name in der Registry (z. B. 'Beleg/Pic-Muster').")
@click.option("--regex", default=r"\d{17,21}", show_default=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="data_patterns/<pattern_name>/ mit pos/, neg/ und optional labels.csv")
@click.option("--char-model", type=click.Path(exists=True), required=True,
              help="Pfad zu char_sgd_*.joblib (Zeichenklassifikator).")
@click.option("--models-dir", type=click.Path(file_okay=False), default="models", show_default=True)
@click.option("--epochs", type=int, default=6, show_default=True)
@click.option("--batch", type=int, default=1024, show_default=True)
@click.option("--seed", type=int, default=7, show_default=True)
@click.option("--continue-from", "continue_from", type=click.Path(exists=True), default=None,
              help="Bestehendes Pattern-Modell (.joblib) für inkrementelles Lernen.")
def train_pattern(pattern_name, regex, data_dir, char_model, models_dir, epochs, batch, seed, continue_from):
    data_dir = pathlib.Path(data_dir)
    models_dir = pathlib.Path(models_dir); ensure_dir(models_dir)

    # Lade Zeichenmodell + HOG-Config
    bundle = joblib.load(char_model)
    char_clf: SGDClassifier = bundle["clf"]
    hog_cfg_dict = bundle.get("hog_cfg", {}) or {}
    cfg = HogCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in hog_cfg_dict.items()})

    # Sammle Sequenzdaten
    pos_dir = data_dir / "pos"
    neg_dir = data_dir / "neg"
    pos_imgs = sorted(pos_dir.glob("*.png")) if pos_dir.exists() else []
    neg_imgs = sorted(neg_dir.glob("*.png")) if neg_dir.exists() else []
    if not pos_imgs and not neg_imgs:
        raise RuntimeError(f"Keine pos/neg Bilder in {data_dir} gefunden.")

    pat = compile_regex(regex)

    def build_samples(paths: List[pathlib.Path], label: int) -> Tuple[np.ndarray, np.ndarray]:
        feats, ys = [], []
        for p in tqdm(paths, desc=f"SeqFeat {'pos' if label==1 else 'neg'}", unit="img"):
            seq, margins = predict_sequence(p, char_clf, cfg)
            seq_d = digits_only(seq)
            f = seq_features(seq_d, margins, pat)
            feats.append(f); ys.append(label)
        if not feats:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.asarray(feats, dtype=np.float32), np.asarray(ys, dtype=np.int64)

    X_pos, y_pos = build_samples(pos_imgs, 1)
    X_neg, y_neg = build_samples(neg_imgs, 0)
    X = np.vstack([X_pos, X_neg]) if len(X_pos) and len(X_neg) else (X_pos if len(X_pos) else X_neg)
    y = np.concatenate([y_pos, y_neg]) if len(y_pos) and len(y_neg) else (y_pos if len(y_pos) else y_neg)

    if len(X) == 0:
        raise RuntimeError("Keine Sequenz-Features erzeugt.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # Modell laden/erstellen
    if continue_from:
        pm = joblib.load(continue_from)
        clf: SGDClassifier = pm["clf"]
        click.echo(f"🔁 Fine-Tuning Pattern von {continue_from}")
    else:
        # Binärer SGD-Klassifikator (logistische Regression per SGD)
        clf = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="optimal",
                            random_state=seed, warm_start=True)
        # Initial partial_fit mit Klassenliste
        clf.partial_fit(Xtr[: min(len(Xtr), batch)],
                        ytr[: min(len(ytr), batch)],
                        classes=np.array([0, 1], dtype=np.int64))

    # Epochen
    for ep in range(1, epochs + 1):
        Xsh, ysh = sk_shuffle(Xtr, ytr, random_state=seed + ep)
        for i in tqdm(range(0, len(Xsh), batch), desc=f"Pattern Epoch {ep}/{epochs}", unit="batch"):
            clf.partial_fit(Xsh[i:i+batch], ysh[i:i+batch])
        val = accuracy_score(yte, (clf.predict_proba(Xte)[:, 1] > 0.5).astype(int))
        click.echo(f"  → val_acc={val:.4f}")

    yhat = (clf.predict_proba(Xte)[:, 1] > 0.5).astype(int)
    click.echo("\n📊 Klassifikationsreport (Pattern):")
    click.echo(classification_report(yte, yhat, digits=4))

    out_path = models_dir / f"pattern_{slugify(pattern_name)}_{now_tag()}.joblib"
    joblib.dump({
        "clf": clf,
        "regex": regex,
        "pattern_name": pattern_name,
        "feature_names": ["len", "m_mean", "m_min", "m_std", "frac_strong", "len_dev", "regex_hit"]
    }, out_path)
    click.echo(f"💾 Gespeichert: {out_path}")


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", name.strip(), flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-").lower()
    return s or "pattern"


if __name__ == "__main__":
    cli()
