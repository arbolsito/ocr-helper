from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, random, os, pathlib

FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
OUT = pathlib.Path("data_digits"); OUT.mkdir(parents=True, exist_ok=True)
random.seed(13)

def render_digit(ch, font, size=42):
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)    
    x0, y0, x1, y1 = draw.textbbox((0, 0), ch, font=font)
    w, h = x1 - x0, y1 - y0
    draw.text(((size-w)//2, (size-h)//2), ch, fill=0, font=font)
    # leichte Störungen
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    arr = np.array(img)
    return arr

def main():
    for d in range(10):
        cls = OUT/str(d); cls.mkdir(exist_ok=True, parents=True)
    for font_path in FONTS:
        font = ImageFont.truetype(font_path, size=34)
        for d in range(10):
            for i in range(800):  # ~1600/Buchstabe bei 2 Fonts
                arr = render_digit(str(d), font)
                # zufällige Affintransformationen
                ang = random.uniform(-6, 6)
                im = Image.fromarray(arr).rotate(ang, expand=False, fillcolor=255)
                # leichtes Resizing
                scale = random.uniform(0.85, 1.15)
                sz = int(42*scale)
                im = im.resize((sz, sz), resample=Image.Resampling.BICUBIC).resize((42,42), resample=Image.Resampling.BICUBIC)
                im.save(OUT/str(d)/f"{os.path.basename(font_path)}_{d}_{i}.png")
    print("done")

if __name__ == "__main__":
    main()
