
from pathlib import Path
from joblib import load
import numpy as np
from PIL import Image
from sklearn.preprocessing._label import LabelEncoder
from sklearn.pipeline import Pipeline

from utils.data_preparation import preprocess_image,hog_features

# Falls deine Utils anders heißen/liegen, anpassen.

def load_model(model_path: Path)->tuple[Pipeline, LabelEncoder]:
    bundle = load(model_path)
    return bundle["pipeline"], bundle["label_encoder"]

def predict_digit_from_path(model_path: Path, img_path: Path):
    pipe, le = load_model(model_path)
    arr = preprocess_image(img_path)        # identisch zum Training
    x = hog_features(arr).reshape(1, -1)    # (1, n_features)
    x_s = pipe.named_steps["scaler"].transform(x)
    y_pred = pipe.named_steps["clf"].predict(x_s)[0]
    # zurück zum String-Label
    label = le.inverse_transform([y_pred])[0]

    # Scores: modified_huber hat keine kalibrierten Wahrscheinlichkeiten
    # decision_function gibt Margin-Scores; softmaxen ist gelogen, aber zum Rankingen ok.
    scores = pipe.named_steps["clf"].decision_function(x_s)  # shape (1, n_classes)
    scores = scores[0]

    # Top-k anzeigen
    top_idx = np.argsort(scores)[::-1]
    top_labels = le.inverse_transform(top_idx)
    top_scores = scores[top_idx]

    return {
        "label": label,
        "score_raw": float(scores[le.transform([label])[0]]),
        "top3": [(str(lab), float(scr)) for lab, scr in zip(top_labels[:3], top_scores[:3])]
    }
