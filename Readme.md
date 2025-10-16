# 📘 Lokale OCR-API: Architektur und Strategie

## 🎯 Zielsetzung

Diese Anwendung bietet eine **lokale, datenschutzkonforme OCR-Schnittstelle**, die speziell für die **Erkennung bestimmter Textmuster** (z. B. 17–21-stellige Ziffernfolgen) konzipiert ist. Ziel ist eine Lösung, die:

* **keine externen Dienste** oder Cloud-APIs nutzt,
* **trainierbare Modelle** sowohl auf Zeichen- als auch auf Muster-Ebene bereitstellt,
* **inkrementelles Training** ermöglicht,
* **einfache Integration** in bestehende Anwendungen (z. B. via HTMX/FastAPI) erlaubt.

---

## 🧱 Systemarchitektur

### 1. **Basisschicht – Zeichenklassifikation**

* **Ziel:** Erkennung einzelner Zeichen (Ziffern oder Buchstaben) aus Bildsegmenten.
* **Datensatz:**

  ```
  data_digits/
   ├─ 0/*.png
   ├─ 1/*.png
   ├─ …
   └─ 9/*.png
  ```
* **Modell:** `SGDClassifier (hinge)` mit HOG-Features (oder optional CNN)
* **Training:**

  * Initial: Volltraining über CLI (`mini-ocr-train`)
  * Fortlaufend: `partial_fit` für inkrementelles Lernen
* **Ergebnis:** `models/char_sgd_vX.joblib`
* **Aufgabe:** liefert pro Bildsegment Label + Klassifikationssicherheit

---

### 2. **Pattern-Schicht – Mustererkennung und Kalibrierung**

* **Ziel:** Erkennung vordefinierter Regex-Muster und Validierung kompletter Sequenzen.
* **Registry:** JSON-Datei mit Pattern-Definitionen:

  ```json
  [
    {
      "name": "Beleg/Pic-Muster",
      "pattern": "\\d{17,21}",
      "enabled": true,
      "model_id_char": "char_sgd_v1",
      "model_id_pattern": "pattern_Beleg-Pic_v1"
    }
  ]
  ```
* **Training:**

  * Positiv-/Negativbeispiele unter `data_patterns/<pattern_name>/`
  * Optional: Kalibrator-Modell (z. B. LogReg oder SGD) zur Qualitätsbewertung
* **Ergebnis:** `models/pattern_<pattern_name>_vX.joblib`

---

### 3. **API-Schicht – FastAPI-Schnittstelle**

* **Endpunkte:**

  * `POST /extract-id`: empfängt ein Bild, erkennt Muster und gibt Treffer zurück
  * `GET /patterns`: listet bekannte Patterns
* **Beispielausgabe:**

  ```json
  {
    "results": [
      {
        "name": "Beleg/Pic-Muster",
        "matches": [
          {
            "value": "1234567890123456789",
            "margin_mean": 1.42,
            "quality_score": 0.93
          }
        ]
      }
    ]
  }
  ```
* **Sicherheit:**

  * Zugriff nur über Auth-Header (`Authorization: Bearer <token>`)
  * Keine externe Kommunikation, keine Cloud-Schnittstellen

---

## 🔁 Trainings- und Datenpipeline

### 1. **Feature-Caching (Effizienz)**

* Einmalige Extraktion von HOG-Features, Speicherung als Cache (`.npz` oder Zarr)
* Spart I/O bei späteren Trainingsläufen
* Automatische Erkennung von Cache-Versionen über Hash-Key

### 2. **Trainingsphasen**

| Phase   | Ziel                  | Datengrundlage | Modelltyp              | Lernart      |
| ------- | --------------------- | -------------- | ---------------------- | ------------ |
| Basis   | Zeichenklassifikation | data_digits    | SGDClassifier (hinge)  | inkrementell |
| Pattern | Musterkalibrierung    | data_patterns  | LogReg / SGDClassifier | inkrementell |

### 3. **Trainingskommandos (CLI)**

```bash
# Volltraining Basis
poetry run mini-ocr-train --ask-path --epochs 10 --batch 2048 --regex "\\d{17,21}"

# Fine-Tuning
poetry run mini-ocr-train --continue-from models/char_sgd_v1.joblib --epochs 5

# Pattern-Training
poetry run mini-ocr-train-pattern --pattern "Beleg/Pic-Muster" --epochs 5
```

---

## 🧩 Projektstruktur

```
ocr-helper/
├─ src/ocrhelper/
│  ├─ api.py               # FastAPI-Endpunkte
│  ├─ train.py             # Zeichen-Training (Basis)
│  ├─ train_pattern.py     # Pattern-Kalibratoren
│  ├─ cache_build.py       # Feature-Caching
│  ├─ synth.py             # Datengenerator
│  ├─ utils/               # Preprocessing, HOG, Logging
│  └─ models/
│      ├─ char_sgd_v1.joblib
│      ├─ pattern_Beleg-Pic_v1.joblib
│      └─ regex_patterns.json
├─ data_digits/
│  └─ 0..9/
├─ data_patterns/
│  └─ Beleg-Pic-Muster/
├─ feature_cache/
├─ tests/
└─ pyproject.toml
```

---

## 🔒 Datenschutz und Sicherheit

* Vollständig **lokale Verarbeitung**, keine Cloud-Abhängigkeiten
* Zugriffsschutz durch API-Token
* Modelle und Trainingsdaten versioniert, rückverfolgbar und auditierbar
* Speicherung auf internen Systemen oder Netzlaufwerken (z. B. UNC-Pfade)

---

## 📈 Erweiterungsperspektiven

* Integration kleiner CNN-Modelle für Handschrift oder unregelmäßige Layouts
* TorchScript-Export für Embedded-Einsatz
* Erweiterte Pattern-Registry (Priorisierung, Ablaufregeln)
* Automatisierte Evaluierung (Accuracy, CER, Sequence Confidence)

---

## 🧠 Kerngedanke

> **Zwei Ebenen, klare Verantwortung:**
>
> * Ebene A erkennt **Zeichen**.
> * Ebene B erkennt **Regeln/Muster** und bewertet deren Qualität.
>
> Damit bleibt das System modular, nachvollziehbar und datenschutzkonform erweiterbar.
