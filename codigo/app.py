"""Minería de Datos con IA - aplicación de escritorio (tkinter). CRUD de clientes, limpieza, trazabilidad, gráficos, K-Means y modelos."""
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import motor

CAMPOS = motor.CAMPOS
MAX_FILAS = 1000  # tope de filas mostradas en la tabla, por rendimiento


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Minería de Datos con IA - Análisis de clientes")
        self.geometry("1100x720")
        self.df = None      # datos originales (los que se editan)
        self.limpio = None  # datos tras la limpieza
        self.out = None     # limpio + columna cluster
        self.traza = None
        bar = ttk.Frame(self); bar.pack(fill="x", padx=6, pady=(4, 0))
        for txt, cmd in [("Cargar CSV", self.cargar), ("Limpiar datos", self.limpiar), ("Guardar CSV limpio", self.guardar),
                         ("Exportar cambios", self.exportar_cambios)]:
            ttk.Button(bar, text=txt, command=cmd).pack(side="left", padx=2)
        ttk.Label(bar, text="  k (0 = automático):").pack(side="left")
        self.k = tk.IntVar(value=0)
        ttk.Spinbox(bar, from_=0, to=8, width=3, textvariable=self.k).pack(side="left")
        ttk.Button(bar, text="Ejecutar K-Means", command=self.cluster).pack(side="left", padx=6)
        ttk.Button(bar, text="Clasificación y regresión", command=self.supervisado).pack(side="left")
        crud = ttk.LabelFrame(self, text="Clientes (crear, leer, editar, eliminar)"); crud.pack(fill="x", padx=6, pady=4)
        ttk.Label(crud, text="Buscar:").pack(side="left", padx=(6, 2))
        self.filtro = tk.StringVar(); self.filtro.trace_add("write", lambda *_: self.refrescar())
        ttk.Entry(crud, textvariable=self.filtro, width=22).pack(side="left")
        ttk.Label(crud, text="  Vista:").pack(side="left")
        self.vista = ttk.Combobox(crud, values=["Originales", "Limpios"], state="readonly", width=10)
        self.vista.set("Originales"); self.vista.pack(side="left", padx=2)
        self.vista.bind("<<ComboboxSelected>>", lambda _: self.refrescar())
        for txt, cmd in [("Agregar", self.agregar), ("Editar seleccionado", self.editar), ("Eliminar seleccionado", self.eliminar),
                         ("Guardar datos (CSV)", self.guardar_datos)]:
            ttk.Button(crud, text=txt, command=cmd).pack(side="left", padx=2, pady=3)
        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True, padx=6, pady=4)
        self.t_datos, self.t_graf, self.t_clu, self.t_traza, self.t_log = (ttk.Frame(self.nb) for _ in range(5))
        for t, n in [(self.t_datos, "Datos"), (self.t_graf, "Exploración"), (self.t_clu, "Clustering"), (self.t_traza, "Trazabilidad"), (self.t_log, "Bitácora")]:
            self.nb.add(t, text=n)
        self.tabla = ttk.Treeview(self.t_datos, columns=CAMPOS, show="headings", selectmode="browse")
        for c in CAMPOS:
            self.tabla.heading(c, text=c); self.tabla.column(c, width=90)
        self.tabla.bind("<Double-1>", lambda _: self.editar())
        sb = ttk.Scrollbar(self.t_datos, command=self.tabla.yview); self.tabla.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.tabla.pack(fill="both", expand=True)
        self.txt = tk.Text(self.t_log, wrap="word"); self.txt.pack(fill="both", expand=True)
        self.txt_traza = tk.Text(self.t_traza, wrap="none", font=("Consolas", 9)); self.txt_traza.pack(fill="both", expand=True)
        self.estado = ttk.Label(self, text="Cargue un archivo CSV para comenzar."); self.estado.pack(fill="x", padx=6)

    # --- archivos ---
    def cargar(self):
        ruta = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not ruta:
            return
        try:
            df = pd.read_csv(ruta)
            faltan = [c for c in CAMPOS if c not in df.columns]
            if faltan:
                raise ValueError(f"Faltan columnas: {faltan}")
        except Exception as e:
            return messagebox.showerror("Error", str(e))
        self.df = df[CAMPOS]
        self.invalidar(); self.estado.config(text=f"{len(df)} registros cargados.")

    def guardar_datos(self):
        if self.df is None:
            return messagebox.showinfo("Aviso", "Primero cargue un CSV.")
        ruta = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="clientes_editado.csv", filetypes=[("CSV", "*.csv")])
        if ruta:
            self.df.to_csv(ruta, index=False, encoding="utf-8-sig"); self.estado.config(text=f"Datos guardados: {len(self.df)} registros.")

    def guardar(self):
        d = self.out if self.out is not None else self.limpio
        if d is None:
            return messagebox.showinfo("Aviso", "No hay datos limpios.")
        ruta = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if ruta:
            d.to_csv(ruta, index=False, encoding="utf-8-sig")

    def exportar_cambios(self):
        if self.traza is None:
            return messagebox.showinfo("Aviso", "Primero ejecute 'Limpiar datos'.")
        ruta = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="cambios_trazabilidad.csv", filetypes=[("CSV", "*.csv")])
        if ruta:
            self.traza["cambios"].to_csv(ruta, index=False, encoding="utf-8-sig")

    # --- CRUD ---
    def agregar(self):
        if self.df is None:
            return messagebox.showinfo("Aviso", "Primero cargue un CSV.")
        nuevo = {c: "" for c in CAMPOS}
        nuevo["id_cliente"] = int(pd.to_numeric(self.df["id_cliente"]).max()) + 1
        nuevo["ultima_compra"] = "2026-10-31"
        self.formulario("Nuevo cliente", nuevo, lambda f: motor.crear(self.df, f), "Cliente agregado.")

    def editar(self):
        i = self.seleccionado()
        if i is None:
            return
        fila = self.df[self.df["id_cliente"] == i].iloc[0]
        vals = {c: ("" if pd.isna(fila[c]) else (int(fila[c]) if c in ("id_cliente", "num_compras") else fila[c])) for c in CAMPOS}
        self.formulario(f"Editar cliente {i}", vals, lambda f: motor.actualizar(self.df, i, f), f"Cliente {i} actualizado.", bloquear_id=True)

    def eliminar(self):
        i = self.seleccionado()
        if i is not None and messagebox.askyesno("Eliminar", f"¿Eliminar el cliente {i}? Se borran todas las filas con ese id."):
            self.df = motor.eliminar(self.df, i)
            self.invalidar(f"Cliente {i} eliminado.")

    def seleccionado(self):
        """id_cliente de la fila elegida. Siempre opera sobre los datos originales."""
        sel = self.tabla.selection()
        if self.df is None or not sel:
            messagebox.showinfo("Aviso", "Cargue datos y seleccione una fila de la tabla.")
            return None
        return int(float(self.tabla.item(sel[0], "values")[0]))

    def formulario(self, titulo, valores, accion, mensaje, bloquear_id=False):
        w = tk.Toplevel(self); w.title(titulo); w.transient(self); ent = {}
        for r, c in enumerate(CAMPOS):
            ttk.Label(w, text=c).grid(row=r, column=0, sticky="e", padx=4, pady=2)
            ent[c] = ttk.Entry(w, width=26); ent[c].insert(0, valores[c]); ent[c].grid(row=r, column=1, padx=4)
        if bloquear_id:
            ent["id_cliente"].state(["readonly"])

        def ok():
            try:
                self.df = accion({c: e.get() for c, e in ent.items()})
            except ValueError as e:
                return messagebox.showerror("Error", str(e), parent=w)
            w.destroy(); self.invalidar(mensaje)
        ttk.Button(w, text="Guardar", command=ok).grid(row=len(CAMPOS), columnspan=2, pady=6)
        w.grab_set()

    def invalidar(self, mensaje=None):
        """Los datos originales cambiaron: los resultados derivados dejan de valer y hay que limpiar de nuevo."""
        self.limpio = self.out = self.traza = None
        self.txt_traza.delete("1.0", "end")
        self.vista.set("Originales"); self.refrescar()
        if mensaje:
            self.estado.config(text=f"{mensaje} Ejecute 'Limpiar datos' para actualizar resultados.")

    # --- análisis ---
    def limpiar(self):
        if self.df is None:
            return messagebox.showinfo("Aviso", "Primero cargue un CSV.")
        self.limpio, log = motor.limpiar(self.df)
        self.out = None
        ind = motor.indicadores(self.limpio)
        self.txt.delete("1.0", "end")
        self.txt.insert("end", "BITÁCORA DE PREPARACIÓN (Fase 2)\n\n" + "\n".join("• " + l for l in log))
        self.txt.insert("end", "\n\nINDICADORES\n" + "\n".join(f"• {k}: {v:,.2f}" for k, v in ind.items()))
        self.traza = motor.trazabilidad(self.df, self.limpio)
        self.txt_traza.delete("1.0", "end"); self.txt_traza.insert("end", self.texto_traza())
        self.vista.set("Limpios"); self.refrescar(); self.explorar()
        self.estado.config(text=f"Limpieza lista: {len(self.limpio)} registros.")

    def texto_traza(self):
        r, cb = self.traza["resumen"], self.traza["cambios"]
        motivos = "".join(f"  - {m}: {n}\n" for m, n in r["por_motivo"].items())
        return (f"TRAZABILIDAD: ORIGINAL → LIMPIO\n\n"
                f"Registros: {r['registros_original']} → {r['registros_limpio']} ({r['duplicados_eliminados']} duplicados eliminados)\n"
                f"Clientes del resultado presentes en el original (por id_cliente): {r['ids_trazables']} de {r['registros_limpio']}\n"
                f"Celdas modificadas: {r['celdas_modificadas']}\n{motivos}\n"
                + motor.tabla_traza(self.traza).to_string(index=False)
                + "\n\nPrimeros cambios (todos se exportan con 'Exportar cambios'):\n" + cb.head(15).to_string(index=False))

    def explorar(self):
        fig = Figure(figsize=(10, 6), tight_layout=True)
        motor.graficos(self.limpio, [fig.add_subplot(2, 2, i + 1) for i in range(4)])
        self.poner(self.t_graf, fig)

    def cluster(self):
        if self.limpio is None:
            return messagebox.showinfo("Aviso", "Primero ejecute 'Limpiar datos'.")
        out, perfil, sil, k = motor.kmeans(self.limpio, self.k.get() or None)
        fig = Figure(figsize=(10, 3.2), tight_layout=True)
        motor.graf_clusters(out, sil, [fig.add_subplot(1, 2, i + 1) for i in range(2)])
        self.poner(self.t_clu, fig, texto=f"k = {k}   |   Silhouette = {sil[k]:.3f}\n\n{perfil.to_string()}")
        self.limpio = out.drop(columns="cluster")
        self.out = out
        self.nb.select(self.t_clu)

    def supervisado(self):
        if self.limpio is None:
            return messagebox.showinfo("Aviso", "Primero ejecute 'Limpiar datos'.")
        self.txt.delete("1.0", "end"); self.txt.insert("end", motor.texto_supervisado(self.limpio)); self.nb.select(self.t_log)

    # --- utilidades de UI ---
    def refrescar(self):
        """Redibuja la tabla según la vista elegida y el texto de búsqueda."""
        base = self.limpio if (self.vista.get() == "Limpios" and self.limpio is not None) else self.df
        self.tabla.delete(*self.tabla.get_children())
        if base is None:
            return
        d = motor.buscar(base[CAMPOS], self.filtro.get())
        if base is self.limpio:  # en la vista limpia la fecha es datetime: se muestra sin hora
            d = d.assign(ultima_compra=pd.to_datetime(d["ultima_compra"], errors="coerce").dt.strftime("%Y-%m-%d").fillna(""))
        for r in d.head(MAX_FILAS).astype(str).itertuples(index=False):
            self.tabla.insert("", "end", values=list(r))
        if self.df is not None:
            self.estado.config(text=f"Mostrando {min(len(d), MAX_FILAS)} de {len(d)} registros ({self.vista.get().lower()}).")

    def poner(self, tab, fig, texto=None):
        for w in tab.winfo_children():
            w.destroy()
        if texto:  # el texto se empaqueta primero abajo para que el gráfico no lo desplace fuera de la ventana
            tk.Label(tab, text=texto, font=("Consolas", 9), justify="left", anchor="w").pack(side="bottom", fill="x")
        c = FigureCanvasTkAgg(fig, tab); c.draw(); c.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    if "--prueba" in sys.argv:  # auto-prueba sin ventana: verifica que el motor corre dentro del ejecutable
        d, _ = motor.limpiar(pd.read_csv(sys.argv[sys.argv.index("--prueba") + 1]))
        print("OK", len(d), motor.kmeans(d)[3]); sys.exit(0)
    App().mainloop()
