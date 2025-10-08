import pathlib, click, re, numpy as np, joblib
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .synth import generate

def load_digit_dataset(data_dir: str):
    X, y = [], []
    root = pathlib.Path(data_dir)
    for d in range(10):
        for p in (root / str(d)).glob("*.png"):
            img = imread(p, as_gray=True)
            img = resize(img, (32, 32), anti_aliasing=True)
            feat = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            X.append(feat); y.append(d)
    return np.array(X), np.array(y)

@click.command()
@click.option("--regex", default=r"\d{17,21}", help="Zielmuster (Python-Regex).")
@click.option("--data-dir", default="data_digits", show_default=True)
@click.option("--models-dir", default="models", show_default=True)
@click.option("--synth", is_flag=True, help="Digit-Datensatz synthetisch erzeugen.")
def cli(regex, data_dir, models_dir, synth):
    # Regex wird für die Evaluierung und spätere API validierung gespeichert
    pattern = re.compile(regex)
    if synth:
        generate(out_dir=data_dir)

    X, y = load_digit_dataset(data_dir)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    clf = LinearSVC(dual=False)
    clf.fit(Xtr, ytr)
    print(classification_report(yte, clf.predict(Xte)))

    pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "regex": regex}, f"{models_dir}/digit_svm.joblib")
    print(f"Gespeichert: {models_dir}/digit_svm.joblib")
