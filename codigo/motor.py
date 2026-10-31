"""Lógica de análisis (sin GUI): limpieza, indicadores, gráficos y K-Means. La usan app.py y informe.py."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

NUM = ["edad", "num_compras", "valor_total", "frecuencia_mensual"]
CAT = ["ciudad", "tipo_cliente", "producto", "medio_pago", "estado"]
FEATURES = ["num_compras", "valor_total", "frecuencia_mensual", "dias_sin_comprar"]
HOY = pd.Timestamp("2026-10-31")


def limpiar(df):
    """Devuelve (df_limpio, bitácora). La bitácora documenta cada cambio y su porqué (Fase 2)."""
    log, n0 = [], len(df)
    df = df.copy()
    log.append(f"Faltantes antes: " + ", ".join(f"{c}={v}" for c, v in df.isna().sum().items() if v))
    df = df.drop_duplicates(); log.append(f"Duplicados eliminados: {n0 - len(df)} (registros idénticos sesgan conteos).")
    for c in ["ciudad", "producto", "medio_pago", "tipo_cliente", "estado"]:
        df[c] = df[c].astype("string").str.strip().str.title()
    log.append("Texto normalizado (espacios y mayúsculas) en variables categóricas: 'BOGOTÁ' y 'Bogotá' eran la misma ciudad.")
    malas = ~df["edad"].between(18, 100) & df["edad"].notna()
    df.loc[malas, "edad"] = np.nan; log.append(f"Edades imposibles (<18 o >100) pasadas a faltante: {malas.sum()}.")
    for c in NUM:  # imputar numéricos con la mediana (robusta a atípicos)
        df[c] = df[c].fillna(df[c].median())
    for c in CAT:  # categóricos con la moda
        df[c] = df[c].fillna(df[c].mode()[0])
    log.append("Faltantes imputados: mediana en numéricas, moda en categóricas (se conserva el 100 % de registros).")
    q1, q3 = df["valor_total"].quantile([.25, .75]); lim = q3 + 3 * (q3 - q1)
    out = df["valor_total"] > lim
    log.append(f"Atípicos extremos en valor_total (> Q3+3·IQR = {lim:,.0f}): {out.sum()}; se limitan (winsorizan) a ese tope.")
    df.loc[out, "valor_total"] = lim
    df["ultima_compra"] = pd.to_datetime(df["ultima_compra"], errors="coerce")
    df["dias_sin_comprar"] = (HOY - df["ultima_compra"]).dt.days.fillna((HOY - df["ultima_compra"]).dt.days.median())
    log.append("Conversión: ultima_compra → fecha; nueva variable dias_sin_comprar (recencia).")
    log.append("Estandarización (StandardScaler) se aplica dentro de K-Means: las variables tienen escalas muy distintas.")
    return df.reset_index(drop=True), log


def indicadores(df):
    return {
        "Ticket promedio por compra": df["valor_total"].sum() / df["num_compras"].sum(),
        "% clientes inactivos": 100 * (df["estado"] == "Inactivo").mean(),
        "Valor total por cliente (media)": df["valor_total"].mean(),
        "Frecuencia mensual media": df["frecuencia_mensual"].mean(),
    }


def graficos(df, axs):
    """Dibuja 4 gráficos en la lista de ejes `axs`."""
    p = df.groupby("producto")["valor_total"].sum().sort_values() / 1e6
    axs[0].barh(p.index, p.values, color="#4C78A8"); axs[0].set_title("Ventas por producto (millones)")
    c = df.groupby("ciudad")["frecuencia_mensual"].mean().sort_values()
    axs[1].barh(c.index, c.values, color="#59A14F"); axs[1].set_title("Frecuencia mensual media por ciudad")
    axs[2].scatter(df["frecuencia_mensual"], df["valor_total"] / 1e6, s=8, alpha=.5, color="#E15759")
    axs[2].set_xlabel("Frecuencia mensual"); axs[2].set_ylabel("Valor total (M)")
    axs[2].set_title(f"Frecuencia vs valor (r = {df['frecuencia_mensual'].corr(df['valor_total']):.2f})")
    axs[3].hist(df["edad"], bins=20, color="#B07AA1"); axs[3].set_title("Distribución de edad")


def kmeans(df, k=None):
    """K-Means estandarizado. Si k es None elige el de mayor silhouette entre 2 y 8."""
    X = StandardScaler().fit_transform(df[FEATURES])
    sil = {i: silhouette_score(X, KMeans(i, n_init=10, random_state=0).fit_predict(X)) for i in range(2, 9)}
    k = k or max(sil, key=sil.get)
    etiquetas = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    out = df.assign(cluster=etiquetas)
    perfil = out.groupby("cluster").agg(clientes=("id_cliente", "count"), compras=("num_compras", "mean"),
                                       valor=("valor_total", "mean"), frecuencia=("frecuencia_mensual", "mean"),
                                       dias_sin_comprar=("dias_sin_comprar", "mean"), edad=("edad", "mean"),
                                       inactivos=("estado", lambda s: 100 * (s == "Inactivo").mean())).round(1)
    perfil["ticket"] = (out.groupby("cluster")["valor_total"].sum() / out.groupby("cluster")["num_compras"].sum()).round(0)
    return out, perfil, sil, k


def graf_clusters(out, sil, axs):
    for c, g in out.groupby("cluster"):
        axs[0].scatter(g["frecuencia_mensual"], g["valor_total"] / 1e6, s=8, alpha=.6, label=f"Grupo {c}")
    axs[0].set_xlabel("Frecuencia mensual"); axs[0].set_ylabel("Valor total (M)"); axs[0].legend(); axs[0].set_title("Clusters K-Means")
    axs[1].plot(list(sil), list(sil.values()), "o-"); axs[1].set_xlabel("k"); axs[1].set_ylabel("Silhouette"); axs[1].set_title("Elección de k")


def _xy(df):
    X = pd.get_dummies(df[["edad", "num_compras", "frecuencia_mensual", "dias_sin_comprar", "ciudad", "producto", "medio_pago"]], dtype=float)
    return X  # valor_total queda fuera de X: es el objetivo (evita fuga de datos)


def clasificar(df):
    """Alternativa A: ¿cliente de alto valor (valor_total > mediana)? Compara regresión logística y Random Forest."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    X, y = _xy(df), (df["valor_total"] > df["valor_total"].median()).astype(int)
    a, b, ya, yb = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
    res = {}
    for n, m in [("Regresión logística", LogisticRegression(max_iter=3000)), ("Random Forest", RandomForestClassifier(300, random_state=0))]:
        p = m.fit(a, ya).predict(b)
        res[n] = dict(accuracy=accuracy_score(yb, p), precision=precision_score(yb, p), recall=recall_score(yb, p),
                      f1=f1_score(yb, p), matriz=confusion_matrix(yb, p))
    imp = pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False).head(5)
    return res, imp


def regresion(df):
    """Alternativa C: estimar valor_total. Compara regresión lineal y Random Forest."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    X = _xy(df)
    a, b, ya, yb = train_test_split(X, df["valor_total"], test_size=.25, random_state=0)
    res = {}
    for n, m in [("Regresión lineal", LinearRegression()), ("Random Forest", RandomForestRegressor(300, random_state=0))]:
        p = m.fit(a, ya).predict(b)
        res[n] = dict(MAE=mean_absolute_error(yb, p), RMSE=mean_squared_error(yb, p) ** .5, R2=r2_score(yb, p))
    return res, yb.std()


def texto_supervisado(df):
    ca, imp = clasificar(df)
    re, desv = regresion(df)
    s = "ALTERNATIVA A - Clasificación (alto vs bajo valor, 25 % de prueba)\n"
    for n, r in ca.items():
        s += f"  {n}: accuracy {r['accuracy']:.3f} | precision {r['precision']:.3f} | recall {r['recall']:.3f} | F1 {r['f1']:.3f}\n    matriz [[VN, FP],[FN, VP]] = {r['matriz'].tolist()}\n"
    s += "  Variables más importantes: " + ", ".join(f"{k} ({v:.2f})" for k, v in imp.items()) + "\n\n"
    s += f"ALTERNATIVA C - Regresión de valor_total (desv. estándar del valor = {desv:,.0f})\n"
    for n, r in re.items():
        s += f"  {n}: MAE {r['MAE']:,.0f} | RMSE {r['RMSE']:,.0f} | R² {r['R2']:.3f}\n"
    return s


def trazabilidad(raw, limpio):
    """Compara original vs limpio. Devuelve dict: resumen, comparativo (por variable) y cambios (uno por celda modificada, con id_cliente y motivo)."""
    base = raw.drop_duplicates()
    base = base[~base["id_cliente"].duplicated()].set_index("id_cliente")
    fin = limpio[~limpio["id_cliente"].duplicated()].set_index("id_cliente")
    fin = fin.loc[fin.index.intersection(base.index)]
    cambios = []
    for c in [c for c in raw.columns if c not in ("id_cliente", "ultima_compra")]:
        a, b = base[c].reindex(fin.index), fin[c]
        num = pd.api.types.is_numeric_dtype(raw[c])
        cambio = a.isna() | (((a - b).abs() > 1e-9) if num else (a.astype(str) != b.astype(str)))
        for i in fin.index[cambio.values]:
            if pd.isna(a[i]):
                m = "Faltante imputado (mediana)" if num else "Faltante imputado (moda)"
            elif c == "edad":
                m = "Edad fuera de 18-100 (imputada)"
            elif c == "valor_total":
                m = "Atípico limitado a Q3+3·IQR"
            else:
                m = "Texto normalizado"
            cambios.append((i, c, a[i], b[i], m))
    cambios = pd.DataFrame(cambios, columns=["id_cliente", "campo", "valor_original", "valor_limpio", "motivo"])
    filas = []
    for c in NUM + CAT:
        num = c in NUM
        f = dict(variable=c, nulos_antes=int(raw[c].isna().sum()), nulos_despues=int(limpio[c].isna().sum()))
        if num:
            f.update(media_antes=raw[c].mean(), media_despues=limpio[c].mean(), min_antes=raw[c].min(), min_despues=limpio[c].min(),
                     max_antes=raw[c].max(), max_despues=limpio[c].max())
        else:
            f.update(distintos_antes=int(raw[c].nunique()), distintos_despues=int(limpio[c].nunique()))
        filas.append(f)
    resumen = dict(registros_original=len(raw), registros_limpio=len(limpio), duplicados_eliminados=len(raw) - len(raw.drop_duplicates()),
                   ids_trazables=int(limpio["id_cliente"].isin(raw["id_cliente"]).sum()), celdas_modificadas=len(cambios),
                   por_motivo=cambios["motivo"].value_counts().to_dict())
    return dict(resumen=resumen, comparativo=pd.DataFrame(filas), cambios=cambios)


def tabla_traza(tr):
    """Comparativo original → limpio en texto, listo para pantalla o documento."""
    f = lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}"
    out = []
    for r in tr["comparativo"].itertuples(index=False):
        if pd.isna(r.media_antes if hasattr(r, "media_antes") else float("nan")) and pd.notna(getattr(r, "distintos_antes", float("nan"))):
            det = f"{int(r.distintos_antes)} → {int(r.distintos_despues)} valores distintos"
        else:
            det = f"media {f(r.media_antes)} → {f(r.media_despues)}; rango {f(r.min_antes)}-{f(r.max_antes)} → {f(r.min_despues)}-{f(r.max_despues)}"
        out.append((r.variable, r.nulos_antes, r.nulos_despues, det))
    return pd.DataFrame(out, columns=["Variable", "Nulos antes", "Nulos después", "Antes → después"])


# ---------- CRUD sobre los datos originales (puro, sin interfaz; app.py solo lo invoca) ----------
CAMPOS = ["id_cliente", "edad", "ciudad", "tipo_cliente", "num_compras", "valor_total",
          "frecuencia_mensual", "producto", "medio_pago", "ultima_compra", "estado"]


def validar_fila(fila):
    """Convierte y valida un registro (dict de textos). Numéricos obligatorios; texto vacío = faltante. Lanza ValueError con el motivo."""
    v = {}
    try:
        v["id_cliente"] = int(float(str(fila["id_cliente"]).strip()))
    except ValueError:
        raise ValueError("id_cliente debe ser un número entero.")
    for c in ["edad", "num_compras", "valor_total", "frecuencia_mensual"]:
        try:
            v[c] = float(str(fila[c]).strip().replace(",", "."))
        except ValueError:
            raise ValueError(f"{c} debe ser un número.")
        if v[c] < 0:
            raise ValueError(f"{c} no puede ser negativo.")
    try:
        v["ultima_compra"] = pd.to_datetime(str(fila["ultima_compra"]).strip(), format="%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError("ultima_compra debe tener formato AAAA-MM-DD.")
    for c in ["ciudad", "tipo_cliente", "producto", "medio_pago", "estado"]:
        t = str(fila[c]).strip()
        v[c] = t if t else np.nan
    return v


def _pos(df, id_):
    i = df.index[df["id_cliente"] == id_]
    if len(i) == 0:
        raise ValueError(f"No existe el cliente {id_}.")
    return i


def crear(df, fila):
    v = validar_fila(fila)
    if (df["id_cliente"] == v["id_cliente"]).any():
        raise ValueError(f"Ya existe un cliente con id {v['id_cliente']}.")
    return pd.concat([df, pd.DataFrame([v])[CAMPOS]], ignore_index=True)


def actualizar(df, id_, fila):
    """El id_cliente no se modifica (es la clave de trazabilidad)."""
    i = _pos(df, id_)
    v = validar_fila({**fila, "id_cliente": id_})
    df = df.copy()
    for c in CAMPOS[1:]:
        df.loc[i, c] = v[c]
    return df


def eliminar(df, id_):
    return df.drop(_pos(df, id_)).reset_index(drop=True)


def buscar(df, texto):
    """Filtra filas cuyo texto (en cualquier columna) contiene `texto`, sin distinguir mayúsculas."""
    t = texto.strip().lower()
    if not t:
        return df
    return df[df.astype(str).apply(lambda col: col.str.lower().str.contains(t, regex=False)).any(axis=1)]


def validacion(df, k=3):
    """Validación adicional: (1) validación cruzada de 5 particiones para la clasificación y la regresión;
    (2) estabilidad de K-Means: 20 semillas y 20 submuestras del 80 %, comparadas con la solución de referencia mediante el índice de Rand ajustado (1 = grupos idénticos)."""
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import adjusted_rand_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
    X, y = _xy(df), (df["valor_total"] > df["valor_total"].median()).astype(int)
    filas = []
    for tarea, nombre, m, cv, objetivo, metricas in [
            ("Clasificación", "Regresión logística", LogisticRegression(max_iter=3000), StratifiedKFold(5, shuffle=True, random_state=0), y, ["accuracy", "f1"]),
            ("Clasificación", "Random Forest", RandomForestClassifier(300, random_state=0), StratifiedKFold(5, shuffle=True, random_state=0), y, ["accuracy", "f1"]),
            ("Regresión", "Regresión lineal", LinearRegression(), KFold(5, shuffle=True, random_state=0), df["valor_total"], ["r2", "neg_root_mean_squared_error"]),
            ("Regresión", "Random Forest", RandomForestRegressor(300, random_state=0), KFold(5, shuffle=True, random_state=0), df["valor_total"], ["r2", "neg_root_mean_squared_error"])]:
        r = cross_validate(m, X, objetivo, cv=cv, scoring=metricas)
        for met in metricas:
            v = r[f"test_{met}"] * (-1 if met.startswith("neg_") else 1)
            filas.append(dict(tarea=tarea, modelo=nombre, metrica=met.replace("neg_root_mean_squared_error", "rmse"), media=v.mean(), desviacion=v.std(), minimo=v.min(), maximo=v.max()))
    Z = StandardScaler().fit_transform(df[FEATURES])
    ref = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    semillas = [adjusted_rand_score(ref, KMeans(k, n_init=10, random_state=s).fit_predict(Z)) for s in range(1, 21)]
    rng = np.random.default_rng(0); sub = []
    for _ in range(20):
        i = rng.choice(len(Z), int(.8 * len(Z)), replace=False)
        sub.append(adjusted_rand_score(ref[i], KMeans(k, n_init=10, random_state=0).fit_predict(Z[i])))
    est = pd.DataFrame([dict(prueba="20 semillas distintas", media=np.mean(semillas), minimo=np.min(semillas), maximo=np.max(semillas)),
                        dict(prueba="20 submuestras del 80 %", media=np.mean(sub), minimo=np.min(sub), maximo=np.max(sub))])
    return dict(cruzada=pd.DataFrame(filas), estabilidad=est)
