import json, re, pathlib
from typing import Dict, List

class RegexRegistry:
    def __init__(self, store_path: str = "models/regex_patterns.json"):
        self.path = pathlib.Path(store_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"patterns": []})
        self._cache = self._read()

    def _read(self) -> Dict:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict):
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def list(self) -> List[Dict]:
        self._cache = self._read()
        return self._cache["patterns"]

    def upsert(self, name: str, pattern: str, enabled: bool = True):
        # validieren
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Ungültiges Regex: {e}")

        pats = self.list()
        for p in pats:
            if p["name"] == name:
                p["pattern"] = pattern
                p["enabled"] = bool(enabled)
                self._write({"patterns": pats})
                return p
        pats.append({"name": name, "pattern": pattern, "enabled": bool(enabled)})
        self._write({"patterns": pats})
        return {"name": name, "pattern": pattern, "enabled": bool(enabled)}

    def remove(self, name: str):
        pats = self.list()
        newp = [p for p in pats if p["name"] != name]
        if len(newp) == len(pats):
            raise KeyError("Pattern nicht gefunden")
        self._write({"patterns": newp})

    def set_enabled(self, name: str, enabled: bool):
        pats = self.list()
        for p in pats:
            if p["name"] == name:
                p["enabled"] = bool(enabled)
                self._write({"patterns": pats})
                return p
        raise KeyError("Pattern nicht gefunden")

    def get_active(self, names: List[str] | None = None) -> List[Dict]:
        pats = self.list()
        if names:
            name_set = set(names)
            sel = [p for p in pats if p["name"] in name_set]
        else:
            sel = [p for p in pats if p.get("enabled")]
        # compile hinzufügen
        for p in sel:
            p["_compiled"] = re.compile(p["pattern"])
        return sel
