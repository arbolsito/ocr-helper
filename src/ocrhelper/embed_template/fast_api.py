
# server.py
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse

app = FastAPI()

API_TOKEN = "supergeheim"  # gleich wie beim OCR-Endpoint


@app.get("/snip/start", response_class=HTMLResponse)
def snip_start(authorization: str | None = Header(None)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Unauthorized")

    # liefert ein self-cleaning Overlay + JS; lebt genau für eine Session
    return """
    """.replace("%(token)s", API_TOKEN)


# Optional: ein Stop-Endpoint, falls du via HTMX „Abbrechen“ serverseitig loggen willst
@app.post("/snip/stop")
def snip_stop():
    return Response(status_code=204)
