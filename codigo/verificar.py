"""Verifica la reproducibilidad: vuelve a ejecutar el análisis en una carpeta temporal y compara los resultados (CSV y JSON) con los entregados.

Uso:  python verificar.py        (desde la carpeta codigo/)
Termina con código 0 si todo coincide byte a byte, 1 si hay diferencias.
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import reproducir

base = Path(__file__).resolve().parent.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

with tempfile.TemporaryDirectory() as tmp:
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):  # silencia el detalle; solo importa la comparación
        reproducir.main(base / "datos" / "clientes.csv", Path(tmp))
    esperados = sorted((base / "resultados").glob("*"))
    malos = [p.name for p in esperados if not (Path(tmp) / "resultados" / p.name).exists() or sha(p) != sha(Path(tmp) / "resultados" / p.name)]
    for p in esperados:
        print(("DIFERENTE " if p.name in malos else "igual     ") + p.name)

print(f"\n{len(esperados) - len(malos)} de {len(esperados)} archivos de resultados coinciden con los entregados.")
sys.exit(1 if malos else 0)
