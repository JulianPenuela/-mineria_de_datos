"""Pruebas del software. Ejecutar: python pruebas.py  (cada prueba lleva el id del caso CP-xx del documento)."""
import unittest

import numpy as np
import pandas as pd

import motor

from pathlib import Path

# el dataset está junto al código o en ../datos (estructura de evidencia_tecnica)
RUTA = next(p for p in (Path("clientes.csv"), Path(__file__).resolve().parent.parent / "datos" / "clientes.csv") if p.exists())
RAW = pd.read_csv(RUTA)


class Pruebas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.log = motor.limpiar(RAW)

    def test_cp01_duplicados(self):
        self.assertEqual(len(RAW) - len(self.df), 30)
        self.assertFalse(self.df.duplicated().any())

    def test_cp02_sin_faltantes(self):
        self.assertEqual(int(self.df[motor.NUM + motor.CAT].isna().sum().sum()), 0)

    def test_cp03_edades_validas(self):
        self.assertTrue(self.df["edad"].between(18, 100).all())

    def test_cp04_texto_unificado(self):
        self.assertEqual(self.df["ciudad"].nunique(), 6)
        self.assertEqual(self.df["producto"].nunique(), 6)

    def test_cp05_atipicos_limitados(self):
        q1, q3 = self.df["valor_total"].quantile([.25, .75])
        self.assertLessEqual(self.df["valor_total"].max(), RAW["valor_total"].max())
        self.assertLess(self.df["valor_total"].max(), 5_000_000)

    def test_cp06_no_modifica_original(self):
        antes = RAW.copy(); motor.limpiar(RAW)
        pd.testing.assert_frame_equal(antes, RAW)

    def test_cp07_indicadores(self):
        ind = motor.indicadores(self.df)
        self.assertAlmostEqual(ind["% clientes inactivos"], 17.33, places=1)
        self.assertGreater(ind["Ticket promedio por compra"], 0)

    def test_cp08_kmeans_automatico(self):
        out, perfil, sil, k = motor.kmeans(self.df)
        self.assertEqual(k, 3)
        self.assertEqual(int(perfil["clientes"].sum()), len(self.df))
        self.assertGreater(sil[k], 0.4)

    def test_cp09_kmeans_manual(self):
        _, perfil, _, k = motor.kmeans(self.df, 4)
        self.assertEqual((k, len(perfil)), (4, 4))

    def test_cp10_clasificacion(self):
        res, _ = motor.clasificar(self.df)
        self.assertGreater(res["Random Forest"]["accuracy"], 0.7)

    def test_cp11_regresion(self):
        res, _ = motor.regresion(self.df)
        self.assertTrue(0.3 < res["Random Forest"]["R2"] < 1)

    def test_cp12_dato_nuevo_cuenta(self):
        nuevo = RAW.iloc[[0]].copy(); nuevo["id_cliente"] = 99999
        d, _ = motor.limpiar(pd.concat([RAW, nuevo], ignore_index=True))
        self.assertEqual(len(d), len(self.df) + 1)
        self.assertIn(99999, d["id_cliente"].values)

    def test_cp14_trazabilidad_resumen(self):
        tr = motor.trazabilidad(RAW, self.df)
        r = tr["resumen"]
        self.assertEqual((r["registros_original"], r["registros_limpio"], r["duplicados_eliminados"], r["ids_trazables"]), (1230, 1200, 30, 1200))
        nulos = int(RAW.drop_duplicates()[motor.NUM + motor.CAT].isna().sum().sum())
        m = r["por_motivo"]
        self.assertEqual(m["Faltante imputado (mediana)"] + m["Faltante imputado (moda)"], nulos)
        self.assertEqual((m["Edad fuera de 18-100 (imputada)"], m["Atípico limitado a Q3+3·IQR"]), (8, 6))

    def test_cp15_cambios_apuntan_al_original(self):
        c = motor.trazabilidad(RAW, self.df)["cambios"]
        self.assertTrue(c["id_cliente"].isin(RAW["id_cliente"]).all())
        self.assertEqual(len(c), len(c.drop_duplicates(["id_cliente", "campo"])))  # una fila por celda modificada

    def test_cp16_crud_crear(self):
        f = dict(zip(motor.CAMPOS, ["90001", "30", "Cali", "Regular", "5", "100000", "0.5", "Hogar", "Efectivo", "2026-10-01", "Activo"]))
        d = motor.crear(RAW, f)
        self.assertEqual(len(d), len(RAW) + 1)
        self.assertEqual(d.iloc[-1]["ciudad"], "Cali")
        with self.assertRaises(ValueError):
            motor.crear(d, f)                                   # id repetido
        for malo in ({"edad": "abc"}, {"valor_total": "-1"}, {"ultima_compra": "01/10/2026"}, {"id_cliente": "x"}):
            with self.assertRaises(ValueError):
                motor.crear(RAW, {**f, "id_cliente": "90002", **malo})
        self.assertEqual(len(RAW), 1230)                        # el original no se toca

    def test_cp17_crud_actualizar(self):
        f = dict(zip(motor.CAMPOS, ["1", "41", "Bogotá", "Premium", "9", "500000", "1.5", "Ropa", "Efectivo", "2026-09-09", "Activo"]))
        d = motor.actualizar(RAW, 1, f)
        fila = d[d["id_cliente"] == 1].iloc[0]
        self.assertEqual((fila["edad"], fila["ciudad"], fila["tipo_cliente"]), (41.0, "Bogotá", "Premium"))
        self.assertEqual(len(d), len(RAW))
        self.assertNotEqual(RAW[RAW["id_cliente"] == 1].iloc[0]["tipo_cliente"], "Premium")  # copia, no modifica el original
        with self.assertRaises(ValueError):
            motor.actualizar(RAW, 999999, f)                    # id inexistente

    def test_cp18_crud_eliminar(self):
        d = motor.eliminar(RAW, 2)
        self.assertNotIn(2, d["id_cliente"].values)
        self.assertEqual(len(d), len(RAW) - 1)
        with self.assertRaises(ValueError):
            motor.eliminar(RAW, 999999)

    def test_cp19_crud_buscar(self):
        self.assertEqual(len(motor.buscar(RAW, "  ")), len(RAW))
        r = motor.buscar(RAW, "CALI")
        self.assertTrue(0 < len(r) < len(RAW))
        self.assertTrue(r.astype(str).apply(lambda c: c.str.lower().str.contains("cali")).any(axis=1).all())
        self.assertEqual(len(motor.buscar(RAW, "zzz-no-existe")), 0)

    def test_cp20_interfaz_crud(self):
        import app
        from unittest import mock
        a = app.App(); a.withdraw()
        try:
            a.df = RAW[app.CAMPOS].copy(); a.limpiar(); a.cluster()
            self.assertIsNotNone(a.limpio)
            a.filtro.set("cali"); a.update()                    # búsqueda filtra la tabla
            self.assertEqual(len(a.tabla.get_children()), min(1000, len(motor.buscar(a.limpio[app.CAMPOS], "cali"))))
            a.filtro.set(""); a.vista.set("Originales"); a.refrescar()
            primero = a.tabla.get_children()[0]; a.tabla.selection_set(primero)
            i = a.seleccionado()
            self.assertEqual(i, int(float(a.tabla.item(primero, "values")[0])))
            with mock.patch("app.messagebox.askyesno", return_value=True), mock.patch("app.messagebox.showinfo"):
                a.eliminar()
            self.assertNotIn(i, a.df["id_cliente"].values)
            self.assertIsNone(a.limpio); self.assertIsNone(a.out); self.assertIsNone(a.traza)  # resultados invalidados
            self.assertEqual(a.vista.get(), "Originales")
        finally:
            a.destroy()

    def test_cp21_validacion_cruzada(self):
        c = motor.validacion(self.df)["cruzada"].set_index(["tarea", "modelo", "metrica"])
        self.assertGreater(c.loc[("Clasificación", "Random Forest", "accuracy"), "media"], 0.75)
        self.assertLess(c.loc[("Clasificación", "Random Forest", "accuracy"), "desviacion"], 0.05)   # resultado estable entre particiones
        self.assertGreater(c.loc[("Clasificación", "Random Forest", "accuracy"), "media"], c.loc[("Clasificación", "Regresión logística", "accuracy"), "media"])
        self.assertGreater(c.loc[("Regresión", "Random Forest", "r2"), "media"], c.loc[("Regresión", "Regresión lineal", "r2"), "media"])

    def test_cp22_estabilidad_kmeans(self):
        e = motor.validacion(self.df)["estabilidad"].set_index("prueba")
        self.assertGreater(e.loc["20 semillas distintas", "minimo"], 0.95)       # mismas agrupaciones con cualquier semilla
        self.assertGreater(e.loc["20 submuestras del 80 %", "media"], 0.9)       # y con el 80 % de los clientes

    def test_cp13_interfaz(self):
        import app
        a = app.App(); a.withdraw()
        try:
            a.df = RAW[app.CAMPOS].copy(); a.limpiar(); a.cluster(); a.supervisado(); a.update()
            self.assertEqual(a.out["cluster"].nunique(), 3)
            self.assertEqual(len(a.tabla.get_children()), 1000)  # tope de filas mostradas
            self.assertEqual(a.traza["resumen"]["registros_limpio"], 1200)
            self.assertIn("1230 → 1200", a.texto_traza())
        finally:
            a.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
