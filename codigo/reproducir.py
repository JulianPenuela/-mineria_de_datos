"""Reproduce el análisis completo sin interfaz: limpia, segmenta, evalúa modelos y escribe resultados y visualizaciones.

Uso:  python reproducir.py [--datos ../datos/clientes.csv] [--salida ../]
Escribe en <salida>/resultados (CSV y JSON) y <salida>/visualizaciones (PNG).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import motor

COLORES = {"Frecuentes": "#1B998B", "Intermedios": "#3D5A80", "Ocasionales": "#F4A259"}


def guardar_ejes(fig, ejes, nombres, carpeta):
    """Guarda cada gráfico de una figura con varios ejes como PNG independiente."""
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    for ax, n in zip(ejes, nombres):
        bb = ax.get_tightbbox(rend).transformed(fig.dpi_scale_trans.inverted()).padded(0.1)
        fig.savefig(carpeta / n, bbox_inches=bb, dpi=130)
        print(f"  visualizaciones/{n}")


def main(datos, salida):
    res, vis = salida / "resultados", salida / "visualizaciones"
    res.mkdir(parents=True, exist_ok=True); vis.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Leyendo {datos}")
    raw = pd.read_csv(datos)
    print(f"  {len(raw)} registros, {raw.shape[1]} columnas")

    print("[2/6] Preparando los datos (Fase 2)")
    limpio, log = motor.limpiar(raw)
    for linea in log:
        print("  -", linea)
    tr = motor.trazabilidad(raw, limpio); rs = tr["resumen"]
    print(f"  Registros: {rs['registros_original']} -> {rs['registros_limpio']}; celdas modificadas: {rs['celdas_modificadas']}")
    limpio.assign(ultima_compra=limpio["ultima_compra"].dt.strftime("%Y-%m-%d")).to_csv(res / "clientes_limpio.csv", index=False, encoding="utf-8-sig")
    tr["cambios"].to_csv(res / "cambios_trazabilidad.csv", index=False, encoding="utf-8-sig")
    motor.tabla_traza(tr).to_csv(res / "comparacion_original_vs_limpio.csv", index=False, encoding="utf-8-sig")
    ind = motor.indicadores(limpio)
    for a, b in ind.items():
        print(f"  Indicador - {a}: {b:,.2f}")

    print("[3/6] Segmentando clientes con K-Means (Fase 4, alternativa B)")
    out, perfil, sil, k = motor.kmeans(limpio)
    orden = perfil["frecuencia"].sort_values(ascending=False).index.tolist()
    nombre = dict(zip(orden, ["Frecuentes", "Intermedios", "Ocasionales"]))
    out["segmento"] = out["cluster"].map(nombre)
    perfil.index = [nombre[i] for i in perfil.index]
    perfil = perfil.loc[["Frecuentes", "Intermedios", "Ocasionales"]]
    out.assign(ultima_compra=out["ultima_compra"].dt.strftime("%Y-%m-%d")).sort_values("id_cliente").to_csv(res / "clientes_segmentados.csv", index=False, encoding="utf-8-sig")
    perfil.rename_axis("segmento").to_csv(res / "perfil_segmentos.csv", encoding="utf-8-sig")
    print(f"  k elegido = {k}; silhouette = {sil[k]:.3f} (k=2: {sil[2]:.3f}, k=4: {sil[4]:.3f})")
    print(perfil.to_string())

    print("[4/6] Evaluando clasificación y regresión (alternativas A y C)")
    ca, imp = motor.clasificar(limpio); re_, desv = motor.regresion(limpio)
    pd.DataFrame({m: {a: v for a, v in x.items() if a != "matriz"} for m, x in ca.items()}).T.rename_axis("modelo").to_csv(res / "metricas_clasificacion.csv", encoding="utf-8-sig")
    pd.DataFrame(re_).T.rename_axis("modelo").to_csv(res / "metricas_regresion.csv", encoding="utf-8-sig")
    imp.rename_axis("variable").rename("importancia").to_csv(res / "importancia_variables.csv", encoding="utf-8-sig")
    for m, x in ca.items():
        print(f"  Clasificación {m}: exactitud {x['accuracy']:.3f}, F1 {x['f1']:.3f}")
    for m, x in re_.items():
        print(f"  Regresión {m}: R2 {x['R2']:.3f}, RMSE {x['RMSE']:,.0f}")

    print("[4b/6] Validación adicional: validación cruzada (5 particiones) y estabilidad de K-Means")
    val = motor.validacion(limpio, k)
    val["cruzada"].to_csv(res / "validacion_cruzada.csv", index=False, encoding="utf-8-sig")
    val["estabilidad"].to_csv(res / "estabilidad_kmeans.csv", index=False, encoding="utf-8-sig")
    for f in val["cruzada"].itertuples():
        print(f"  {f.tarea} - {f.modelo} - {f.metrica}: {f.media:,.3f} +/- {f.desviacion:,.3f}")
    for f in val["estabilidad"].itertuples():
        print(f"  Estabilidad ({f.prueba}): Rand ajustado medio {f.media:.3f}, minimo {f.minimo:.3f}")

    print("[5/6] Generando visualizaciones")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True); motor.graficos(limpio, axs.ravel())
    guardar_ejes(fig, axs.ravel(), ["eda_ventas_por_producto.png", "eda_frecuencia_por_ciudad.png", "eda_frecuencia_vs_valor.png", "eda_distribucion_edad.png"], vis)
    fig, ax = plt.subplots(figsize=(7, 4.5), tight_layout=True)
    for g, d in out.groupby("segmento"):
        ax.scatter(d["frecuencia_mensual"], d["valor_total"] / 1e6, s=9, alpha=.6, color=COLORES[g], label=g)
    ax.set(xlabel="Frecuencia mensual", ylabel="Valor total (millones $)", title="Segmentos de clientes (K-Means)"); ax.legend()
    fig.savefig(vis / "clustering_segmentos.png", dpi=130); print("  visualizaciones/clustering_segmentos.png")
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.plot(list(sil), list(sil.values()), "o-", color="#3D5A80"); ax.axvline(k, ls="--", color="#F4A259")
    ax.set(xlabel="Número de grupos (k)", ylabel="Silhouette", title="Elección de k"); fig.savefig(vis / "clustering_eleccion_k.png", dpi=130); print("  visualizaciones/clustering_eleccion_k.png")
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), tight_layout=True)
    for ax, (col, tit) in zip(axs, [("frecuencia", "Compras por mes"), ("ticket", "Ticket ($)"), ("dias_sin_comprar", "Días sin comprar"), ("inactivos", "% inactivos")]):
        ax.bar(perfil.index, perfil[col], color=[COLORES[s] for s in perfil.index]); ax.set_title(tit); ax.tick_params(axis="x", rotation=20)
    fig.savefig(vis / "perfil_segmentos.png", dpi=130); print("  visualizaciones/perfil_segmentos.png")
    motivos = pd.Series({**{"Duplicados eliminados": rs["duplicados_eliminados"]}, **rs["por_motivo"]}).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    ax.barh(motivos.index, motivos.values, color="#3D5A80"); ax.set_title("Preparación: cambios por motivo")
    for i, v in enumerate(motivos.values):
        ax.text(v, i, f" {v}", va="center")
    fig.savefig(vis / "calidad_cambios_por_motivo.png", dpi=130); print("  visualizaciones/calidad_cambios_por_motivo.png")
    mc = ca["Random Forest"]["matriz"]
    fig, ax = plt.subplots(figsize=(4.5, 4), tight_layout=True)
    ax.imshow(mc, cmap="Blues"); ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Bajo", "Alto"], yticklabels=["Bajo", "Alto"], xlabel="Predicho", ylabel="Real", title="Clasificación: matriz de confusión")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mc[i, j]), ha="center", va="center", color="white" if mc[i, j] > mc.max() / 2 else "black", fontsize=14)
    fig.savefig(vis / "clasificacion_matriz_confusion.png", dpi=130); print("  visualizaciones/clasificacion_matriz_confusion.png")
    fig, ax = plt.subplots(figsize=(7, 3.5), tight_layout=True)
    ax.barh(imp.index[::-1], imp.values[::-1], color="#1B998B"); ax.set_title("Importancia de variables (Random Forest)")
    fig.savefig(vis / "clasificacion_importancia_variables.png", dpi=130); print("  visualizaciones/clasificacion_importancia_variables.png")
    plt.close("all")

    print("[6/6] Escribiendo resumen")
    resumen = dict(registros_original=rs["registros_original"], registros_limpio=rs["registros_limpio"], duplicados_eliminados=rs["duplicados_eliminados"],
                   celdas_modificadas=rs["celdas_modificadas"], cambios_por_motivo=rs["por_motivo"], indicadores=ind, k=k, silhouette_por_k=sil,
                   perfil_segmentos=perfil.round(3).to_dict("index"), clasificacion={m: {a: v for a, v in x.items() if a != "matriz"} for m, x in ca.items()},
                   matriz_confusion_random_forest=mc.tolist(), validacion_cruzada=val["cruzada"].to_dict("records"), estabilidad_kmeans=val["estabilidad"].to_dict("records"), regresion=re_, desviacion_valor_total=desv, semilla_modelos=0)
    (res / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print("  resultados/resumen.json\nListo.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent.parent
    ap.add_argument("--datos", type=Path, default=base / "datos" / "clientes.csv")
    ap.add_argument("--salida", type=Path, default=base)
    a = ap.parse_args()
    main(a.datos, a.salida)
