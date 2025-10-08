import numpy as np, cv2, re, joblib
from skimage.feature import hog
from typing import List, Tuple

class DigitOCR:
    def __init__(self, model_path="models/digit_svm.joblib"):
        bundle = joblib.load(model_path)
        self.clf = bundle["clf"]
        self.regex = re.compile(bundle.get("regex", r"\d{17,21}"))

    @staticmethod
    def preprocess(img_bytes: bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Ungültiges Bild.")
        scale = 2.0 if min(gray.shape[:2]) < 600 else 1.5
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,31,10)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8), iterations=1)
        return gray, thr

    @staticmethod
    def find_boxes(thr) -> List[Tuple[int,int,int,int]]:
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if h < 12 or w < 5: continue
            ar = h / max(w,1)
            if 0.8 <= ar <= 6.0:
                boxes.append((x,y,w,h))
        boxes.sort(key=lambda b: b[0])
        return boxes

    def classify_patch(self, gray, box):
        x,y,w,h = box
        crop = gray[y:y+h, x:x+w]
        crop = cv2.copyMakeBorder(crop, 2,2,2,2, cv2.BORDER_CONSTANT, value=255)
        crop = cv2.resize(crop, (32,32), interpolation=cv2.INTER_AREA)
        feat = hog(crop/255.0, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        pred = self.clf.predict([feat])[0]
        try:
            margin = float(self.clf.decision_function([feat]).max())
        except Exception:
            margin = 0.0
        return str(int(pred)), margin

    def read_sequence(self, gray, thr):
        boxes = self.find_boxes(thr)
        if not boxes: return "", 0.0
        chars, margins = [], []
        avg_w = np.mean([w for _,_,w,_ in boxes]) if boxes else 10
        buf, confs, prev_right = [], [], None
        for x,y,w,h in boxes:
            if prev_right is not None and (x - prev_right) > avg_w*1.7:
                chars.append(("".join(buf), np.mean(confs) if confs else 0.0)); buf, confs = [], []
            ch, m = self.classify_patch(gray, (x,y,w,h))
            buf.append(ch); confs.append(m)
            prev_right = x + w
        if buf:
            chars.append(("".join(buf), np.mean(confs) if confs else 0.0))
        # wähle die „beste“ Gruppe (höchste Margin)
        best = max(chars, key=lambda t: t[1]) if chars else ("", 0.0)
        return best

    def extract_matches(self, img_bytes: bytes):
        gray, thr = self.preprocess(img_bytes)
        seq, conf = self.read_sequence(gray, thr)
        digits_only = re.sub(r"\D", "", seq)
        hits = self.regex.findall(digits_only)
        # dedupe
        uniq = sorted(set(hits))
        return [{"value": h, "confidence_proxy": round(conf, 3)} for h in uniq]
    
    def extract_matches_for_patterns(self, img_bytes: bytes, patterns: list[dict]):
        gray, thr = self.preprocess(img_bytes)
        seq, conf = self.read_sequence(gray, thr)
        digits_only = re.sub(r"\D", "", seq)
        out = []
        for p in patterns:
            comp = p["_compiled"]
            hits = comp.findall(digits_only)
            uniq = sorted(set(hits))
            out.append({
            "name": p["name"],
            "pattern": p["pattern"],
            "enabled": p.get("enabled", True),
            "matches": [{"value": h, "confidence_proxy": round(conf, 3)} for h in uniq]
        })
        return out
