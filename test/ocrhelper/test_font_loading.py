# debug_font.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter, Image
import os, glob

def discover():
    font_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    paths = []
    for ext in ("ttf", "otf"):
        paths += glob.glob(os.path.join(font_dir, f"*.{ext}"))
    return sorted(paths)

fonts = discover()[:10]
print("Kandidaten:", len(fonts))
for fp in fonts:
    try:
        f = ImageFont.truetype(fp, size=34)
        print("OK:", fp)
        img = Image.new("L", (100, 100), 255)
        d = ImageDraw.Draw(img)
        x0,y0,x1,y1 = d.textbbox((0,0), "7", font=f)
        d.text(((100-(x1-x0))//2,(100-(y1-y0))//2), "7", fill=0, font=f)
        img = img.filter(ImageFilter.GaussianBlur(0.3))
        img.resize((42,42), resample=Image.Resampling.BICUBIC).save("debug_7.png")
        break
    except Exception as e:
        print("FAIL:", fp, "->", e)