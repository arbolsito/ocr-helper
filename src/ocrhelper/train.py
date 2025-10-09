import pathlib, re, os, json
import numpy as np
import joblib
import click
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score

# --------------------- Datenladen mit Progress ---------------------

def iter_digit_paths(data_dir: pathlib.Path):
    # Erwartet data_dir/0..9/*.png — groß genug, um auf einem Share zu liegen
    for d in range(10):
        cls_dir = data_dir / str(d)
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.png"):
            yield d, p

def load_digit_dataset(data_dir: str | pathlib.Path, limit: int | None = None):
    data_dir = pathlib.Path(data_dir)
    X, y = [], []
    paths = list(iter_digit_paths(data_dir))
    if not paths:
        raise RuntimeError(f"Keine Bilder unter {data_dir} gefunden (erwartet Unterordner 0..9).")
    if limit:
        paths = paths[:limit]
    for d, p in tqdm(paths, desc="HOG Features", unit="img"):
        img = imread(p, as_gray=True)
        img = resize(img, (32, 32), anti_aliasing=True)
        feat = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        X.append(feat); y.append(d)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y

# --------------------- CLI ---------------------

@click.command()
@click.option("--regex", default=r"\d{17,21}", show_default=True,
              help="Zielmuster, wird nur als Metadatum gespeichert.")
@click.option("--models-dir", default="models", show_default=True,
              help="Ablageort für .joblib")
@click.option("--epochs", default=10, show_default=True, type=int)
@click.option("--batch", default=2048, show_default=True, type=int)
@click.option("--val-split", default=0.15, show_default=True, type=float)
@click.option("--shuffle-seed", default=42, show_default=True, type=int)
@click.option("--limit", default=None, type=int,
              help="Optional nur die ersten N Bilder laden (Debug).")
@click.option("--continue-from", "continue_from", default=None, type=str,
              help="Pfad zu bestehendem SGD-Modell (.joblib) für Fine-Tuning.")
@click.option("--ask-path", is_flag=True,
              help="Beim Start interaktiv nach dem Datenordner fragen.")
@click.option("--data-dir", default=None, type=str,
              help="Alternativ: Pfad mit Trainingsdaten (überschreibt --ask-path-Eingabe).")
def cli(regex, models_dir, epochs, batch, val_split, shuffle_seed, limit, continue_from, ask_path, data_dir):
    # 1) Datenordner erfragen oder übernehmen
    if data_dir is None:
        if ask_path:
            data_dir = click.prompt("Pfad zu data_digits (UNC/Netzpfad erlaubt)",
                                    type=click.Path(exists=True, file_okay=False, dir_okay=True))
        else:
            # Fallback auf ./data_digits
            data_dir = "data_digits"
    data_dir = pathlib.Path(data_dir)

    # 2) Daten laden
    print(f"📂 Lade Daten aus: {data_dir}")
    X, y = load_digit_dataset(data_dir, limit=limit)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=val_split, stratify=y, random_state=shuffle_seed)

    # 3) Modell laden oder neu anlegen
    if continue_from:
        print(f"🔁 Fine-Tuning von: {continue_from}")
        bundle = joblib.load(continue_from)
        clf: SGDClassifier = bundle["clf"]
    else:
        clf = SGDClassifier(
            loss="hinge",           # SVM-ähnlich
            alpha=1e-4,
            learning_rate="optimal",
            random_state=shuffle_seed,
            early_stopping=False,   # wir kontrollieren Epochen selbst
            warm_start=True
        )
        # initial classes bekannt machen
        classes = np.arange(10, dtype=np.int64)
        init_n = min(len(Xtr), batch)
        clf.partial_fit(Xtr[:init_n], ytr[:init_n], classes=classes)

    # 4) Epochen-Training mit Progress
    print("🏋️ Training...")
    for ep in range(1, epochs + 1):
        sh = shuffle(Xtr, ytr, random_state=shuffle_seed + ep)
        Xsh, ysh = sh if sh is not None else (Xtr, ytr)
        for i in tqdm(range(0, len(Xsh), batch), desc=f"Epoch {ep}/{epochs}", unit="batch"):
            xb = Xsh[i:i+batch]; yb = ysh[i:i+batch]
            clf.partial_fit(xb, yb)
        val_acc = accuracy_score(yte, clf.predict(Xte))
        print(f"  → val_acc={val_acc:.4f}")

    # 5) Report + Speichern
    ypred = clf.predict(Xte)
    print("\n📊 Klassifikationsreport (Val):")
    print(classification_report(yte, ypred, digits=4))

    models_dir = pathlib.Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "digit_sgd.joblib"
    joblib.dump({"clf": clf, "regex": regex}, out_path)
    print(f"💾 Gespeichert: {out_path}")

if __name__ == "__main__":
    cli()
