from fastapi import FastAPI, File, UploadFile, Header, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from .infer import DigitOCR
from .registry import RegexRegistry

API_TOKEN = "supergeheim"
ocr = DigitOCR(model_path="models/digit_svm.joblib")
reg = RegexRegistry("models/regex_patterns.json")
app = FastAPI(title="Mini OCR (mehrere benannte Regex, lokal)")

def auth_or_401(authorization: str | None):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/healthz")
def healthz(): return {"ok": True}

# ---------- Pattern-Management ----------
@app.get("/patterns")
def list_patterns(authorization: str | None = Header(None)):
    auth_or_401(authorization)
    return {"patterns": reg.list()}

@app.post("/patterns")
def add_pattern(
    name: str, pattern: str, enabled: bool = True,
    authorization: str | None = Header(None)
):
    auth_or_401(authorization)
    try:
        p = reg.upsert(name=name, pattern=pattern, enabled=enabled)
        return {"pattern": p}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.patch("/patterns/{name}")
def patch_pattern(
    name: str, pattern: str | None = None, enabled: bool | None = None,
    authorization: str | None = Header(None)
):
    auth_or_401(authorization)
    if pattern is not None:
        try:
            p = reg.upsert(name=name, pattern=pattern, enabled=enabled if enabled is not None else True)
            return {"pattern": p}
        except ValueError as e:
            raise HTTPException(400, str(e))
    if enabled is not None:
        try:
            p = reg.set_enabled(name, enabled)
            return {"pattern": p}
        except KeyError:
            raise HTTPException(404, "Pattern nicht gefunden")
    raise HTTPException(400, "Nichts zu ändern")

@app.delete("/patterns/{name}")
def delete_pattern(name: str, authorization: str | None = Header(None)):
    auth_or_401(authorization)
    try:
        reg.remove(name)
        return JSONResponse(status_code=204, content=None)
    except KeyError:
        raise HTTPException(404, "Pattern nicht gefunden")

# ---------- OCR ----------
@app.post("/extract-id")
async def extract_id(
    file: UploadFile = File(...),
    authorization: str | None = Header(None),
    pattern: List[str] = Query(default=None, description="Optionale Liste von Pattern-Namen, kommasepariert oder mehrfach")
):
    auth_or_401(authorization)
    try:
        content = await file.read()
        pats = reg.get_active(pattern)  # spezifische oder enabled
        if not pats:
            raise HTTPException(400, "Keine Patterns aktiv oder gefunden")
        res = ocr.extract_matches_for_patterns(content, pats)
        return {"results": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def cli():
    import uvicorn
    uvicorn.run("ocrhelper.api:app", host="0.0.0.0", port=8000, reload=True)
