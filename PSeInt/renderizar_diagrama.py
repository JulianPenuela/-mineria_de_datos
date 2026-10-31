"""Convierte el diagrama de flujo exportado por PSeInt (SVG) a PNG y lo corta en partes legibles.
resvg no interpreta 'font-size:16pt' de wxSVG, así que se pasa a px (1 pt = 4/3 px) antes de renderizar. Requiere: pip install resvg-py pillow numpy"""
import io
import re
import sys

import numpy as np
import resvg_py
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
PARTES = 6
FUENTE = ["C:/Program Files (x86)/PSeInt/Inconsolata-Regular.ttf"]  # fuente que usa PSeInt
svg = open(sys.argv[1] if len(sys.argv) > 1 else "diagrama_flujo_pseint.svg", encoding="utf-8").read()
svg = svg.replace('width="113.10cm" height="365.09cm"', 'width="3206" height="10349"', 1)
svg = re.sub(r"font-size:([\d.]+)pt", lambda m: f"font-size:{float(m.group(1)) * 4 / 3:.2f}px", svg)
im = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", font_files=FUENTE)))).convert("RGB")
im.save("diagrama_flujo_pseint.png", optimize=True)

# cortes donde la franja tiene menos tinta (entre cajas), para no partir ninguna caja ni línea de texto
tinta = (np.asarray(im.convert("L")) < 200).sum(axis=1)
alto = im.size[1]; cortes = [0]
for i in range(1, PARTES):
    objetivo = i * alto // PARTES
    ventana = range(objetivo - 150, objetivo + 150)
    cortes.append(min(ventana, key=lambda y: tinta[y - 3:y + 4].sum()))
cortes.append(alto)
for i in range(PARTES):
    parte = im.crop((0, cortes[i], im.size[0], cortes[i + 1]))
    xs = np.where((np.asarray(parte.convert("L")) < 200).any(axis=0))[0]  # recorta el blanco sobrante a la derecha
    parte.crop((max(0, xs.min() - 20), 0, min(parte.size[0], xs.max() + 20), parte.size[1])).save(f"diagrama_flujo_parte{i + 1}.png", optimize=True)
print(im.size, "cortes en", cortes)
