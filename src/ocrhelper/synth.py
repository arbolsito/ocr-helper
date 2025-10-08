from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os, sys, glob, random, pathlib
import numpy as np



def _windows_font_dirs():
    windir = os.environ.get("WINDIR", r"C:\Windows")
    return [os.path.join(windir, "Fonts")]

def _linux_font_dirs():
    # reicht für die üblichen Docker/Ubuntu/ Debian-Fälle
    return ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]

def _mac_font_dirs():
    return ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]

def discover_fonts(prefer=None, include_ttc=False):
    """
    Sucht ein paar brauchbare Fonts auf dem System.
    prefer: Liste von Namen (case-insensitiv) die bevorzugt werden, z.B. ["consola", "arial", "dejavu"]
    include_ttc: True, wenn .ttc Font Collections erlaubt sind (Pillow braucht dann index)
    """
    dirs = []
    if os.name == "nt":
        dirs += _windows_font_dirs()
    elif sys.platform == "darwin":
        dirs += _mac_font_dirs()
    else:
        dirs += _linux_font_dirs()

    exts = ["ttf", "otf"] + (["ttc"] if include_ttc else [])
    cand = []
    for d in dirs:
        for ext in exts:
            cand += glob.glob(os.path.join(d, f"*.{ext}"))

    # optional bevorzugte Fonts nach vorne sortieren
    if prefer:
        pl = [p.lower() for p in prefer]
        cand.sort(key=lambda p: min((p.lower().find(x) for x in pl if x in p.lower()), default=9999))
    else:
        cand.sort()

    # dedupe + existenz prüfen
    seen, out = set(), []
    for p in cand:
        if p.lower() not in seen and os.path.isfile(p):
            seen.add(p.lower()); out.append(p)
    return out

def load_font_any(size=34, prefer=None):
    """Versucht systemweite Fonts; fällt auf Pillow-Default zurück."""
    fonts = discover_fonts(prefer=prefer, include_ttc=False)  # .ttc weglassen, keep it simple
    for fp in fonts:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
    # letzter Ausweg
    return ImageFont.load_default()

def render_digit(ch: str, font: ImageFont.FreeTypeFont, size=42) -> np.ndarray:
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=font)
    w, h = x1 - x0, y1 - y0
    d.text(((size - w) // 2, (size - h) // 2), ch, fill=0, font=font)
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return np.array(img)

def generate(out_dir: str = "data_digits", n_per_digit: int = 1500, seed=13, preferred=None):
    """
    preferred: Liste von Teilstrings, um Fonts zu priorisieren (z.B. ["consola","arial","dejavu"])
    """
    random.seed(seed)
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for d in range(10):
        (out / str(d)).mkdir(parents=True, exist_ok=True)

    # mehrere Fonts laden (mono + sans, wenn möglich)
    fonts = []
    # versucht zuerst Consolas/Arial/DejaVu/Ubuntu-Mono zu finden
    for name in (preferred or ["consola", "consolas", "dejavu", "arial", "ubuntu"]):
        f = load_font_any(size=34, prefer=[name])
        # load_default() hat keine path-Eigenschaft; markieren wir ihn per Name
        fonts.append((f, name))
        

    # wenn alles nur default ist, wenigstens zwei Varianten mit unterschiedlichen Größen
    if all(f.__class__ is ImageFont.ImageFont for f, _ in fonts):
        fonts = [(load_font_any(size=s), f"default_{s}") for s in (32, 36, 40)]

    for fp,font_name in fonts:
        font_path = fp.path if isinstance(fp, ImageFont.FreeTypeFont) else font_name
        font = ImageFont.truetype(font_path, size=34)
        for digit in map(str, range(10)):
            for i in range(n_per_digit // len(fonts)):
                arr = render_digit(digit, font)
                ang = random.uniform(-6, 6)
                im = Image.fromarray(arr).rotate(ang, expand=False, fillcolor=255)
                scale = random.uniform(0.85, 1.15)
                sz = int(42 * scale)
                im = im.resize((sz, sz), resample=Image.Resampling.BICUBIC).resize((42, 42), Image.Resampling.BICUBIC)
                im.save(out / digit / f"{font_name}_{digit}_{i}.png")
