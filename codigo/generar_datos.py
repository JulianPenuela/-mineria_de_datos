"""Genera clientes.csv: datos sintéticos (semilla fija) con suciedad deliberada para la Fase 2."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1200
CIUDADES = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Cartagena"]
PRODUCTOS = ["Electrónica", "Hogar", "Ropa", "Alimentos", "Deportes", "Belleza"]
PAGOS = ["Tarjeta crédito", "Tarjeta débito", "Efectivo", "Transferencia"]

# 3 comportamientos latentes: frecuentes (~35 %), medios (~40 %), ocasionales (~25 %)
g = rng.choice(3, N, p=[0.35, 0.40, 0.25])
compras = np.select([g == 0, g == 1, g == 2],
                    [rng.poisson(28, N), rng.poisson(12, N), rng.poisson(3, N)]).clip(1)
ticket = np.select([g == 0, g == 1, g == 2],
                   [rng.normal(60_000, 15_000, N), rng.normal(150_000, 30_000, N), rng.normal(260_000, 50_000, N)]).clip(15_000)
valor = (compras * ticket).round(-2)
frecuencia = (compras / rng.uniform(10, 14, N)).round(2)  # compras por mes
edad = np.select([g == 0, g == 1, g == 2],
                 [rng.normal(30, 6, N), rng.normal(40, 8, N), rng.normal(48, 10, N)]).clip(18, 80).round()
ultima = pd.Timestamp("2026-10-31") - pd.to_timedelta(
    np.select([g == 0, g == 1, g == 2], [rng.integers(0, 40, N), rng.integers(10, 120, N), rng.integers(90, 400, N)]), unit="D")
estado = np.where((pd.Timestamp("2026-10-31") - ultima).days > 180, "Inactivo", "Activo")
tipo = np.where(valor > 2_500_000, "Premium", np.where(valor > 800_000, "Regular", "Básico"))

df = pd.DataFrame({
    "id_cliente": np.arange(1, N + 1), "edad": edad, "ciudad": rng.choice(CIUDADES, N, p=[.3, .2, .15, .15, .1, .1]),
    "tipo_cliente": tipo, "num_compras": compras, "valor_total": valor, "frecuencia_mensual": frecuencia,
    "producto": rng.choice(PRODUCTOS, N), "medio_pago": rng.choice(PAGOS, N, p=[.35, .3, .15, .2]),
    "ultima_compra": ultima.strftime("%Y-%m-%d"), "estado": estado})

# --- suciedad ---
df.loc[rng.choice(N, 40, replace=False), "edad"] = np.nan
df.loc[rng.choice(N, 30, replace=False), "ciudad"] = np.nan
df.loc[rng.choice(N, 25, replace=False), "medio_pago"] = np.nan
df.loc[rng.choice(N, 35, replace=False), "valor_total"] = np.nan
df.loc[rng.choice(N, 40, replace=False), "ciudad"] = df["ciudad"].str.upper()          # inconsistencia de mayúsculas
df.loc[rng.choice(N, 20, replace=False), "producto"] = df["producto"].str.lower() + " "  # minúsculas + espacio
df.loc[rng.choice(N, 8, replace=False), "edad"] = rng.choice([-5, 0, 150, 999], 8)       # edades imposibles
df.loc[rng.choice(N, 6, replace=False), "valor_total"] *= 15                              # atípicos extremos
df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True)                   # 30 duplicados exactos
df.to_csv("clientes.csv", index=False, encoding="utf-8-sig")
print(df.shape)
