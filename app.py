"""
Desarrollo de un Sistema Inteligente para la Predicción del Abandono de Clientes mediante Técnicas de Machine Learning para la empresa XYZ
Empresa hipotética de seguros de automóviles (datos simulados).

Cómo se ejecuta:  streamlit run app.py
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Predicción de abandono - Seguros", page_icon="🚗", layout="wide")

SEMILLA = 42
CARPETA = Path(__file__).parent
ARCHIVO_EXCEL = CARPETA / "Base_Datos_Simulada_Seguros_Auto.xlsx"

COLS_NUM = ["edad", "antiguedad_anios", "prima_mensual", "pagos_atrasados",
            "accidentes", "reclamaciones", "satisfaccion", "contactos_servicio",
            "renovaciones"]
COLS_CAT = ["tipo_cobertura"]
COLS_X = ["edad", "antiguedad_anios", "tipo_cobertura", "prima_mensual", "pagos_atrasados",
          "accidentes", "reclamaciones", "satisfaccion", "contactos_servicio", "renovaciones"]


# ----------------------------------------------------------------------------
# 1. Lectura de datos
# ----------------------------------------------------------------------------
def leer_tabla(archivo):
    """Lee un Excel o CSV y localiza solo la fila de encabezados (la que dice 'edad')."""
    nombre = getattr(archivo, "name", str(archivo)).lower()
    if nombre.endswith(".csv"):
        bruto = pd.read_csv(archivo, header=None, dtype=str)
    else:
        bruto = pd.read_excel(archivo, sheet_name=0, header=None, dtype=str)
    fila = None
    for i in range(min(len(bruto), 15)):
        celdas = [str(c).strip().lower() for c in bruto.iloc[i].tolist()]
        if "edad" in celdas:
            fila = i
            break
    if fila is None:
        raise ValueError("No encontré la fila de encabezados (debe existir una columna llamada 'edad').")
    df = bruto.iloc[fila + 1:].copy()
    df.columns = [str(c).strip().lower().replace("ñ", "n") for c in bruto.iloc[fila]]
    df = df.dropna(how="all")
    for c in df.columns:
        if c != "tipo_cobertura":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def generar_datos(n=3000):
    """Datos simulados (por si no existe el Excel)."""
    rng = np.random.default_rng(SEMILLA)
    edad = rng.integers(18, 76, n)
    antig = rng.integers(1, 16, n)
    cob = rng.choice(["Basica", "Intermedia", "Completa"], size=n, p=[0.30, 0.45, 0.25])
    prima = np.clip(rng.normal(55 + (cob == "Completa") * 35 + (cob == "Intermedia") * 15, 18, n), 20, 180).round(2)
    pagos = rng.poisson(0.7, n)
    acc = rng.poisson(0.45, n)
    recl = rng.poisson(0.55, n)
    sat = rng.integers(1, 6, n)
    cont = rng.poisson(1.4, n)
    renov = np.maximum(antig - 1, 0)
    z = (-2.15 + 0.48 * pagos + 0.22 * acc + 0.18 * recl + 0.55 * (5 - sat) + 0.16 * cont
         + 0.012 * (prima - 70) - 0.075 * antig - 0.025 * renov + 0.20 * (cob == "Basica"))
    ab = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame(dict(edad=edad, antiguedad_anios=antig, tipo_cobertura=cob, prima_mensual=prima,
                             pagos_atrasados=pagos, accidentes=acc, reclamaciones=recl, satisfaccion=sat,
                             contactos_servicio=cont, renovaciones=renov, abandono=ab))


@st.cache_data
def cargar_base():
    if ARCHIVO_EXCEL.exists():
        return leer_tabla(ARCHIVO_EXCEL), "Excel del proyecto"
    return generar_datos(), "datos simulados generados por el programa"


# ----------------------------------------------------------------------------
# 2. Entrenamiento (se hace una sola vez y queda en memoria)
# ----------------------------------------------------------------------------
def nuevo_preprocesador():
    return ColumnTransformer([
        ("num", StandardScaler(), COLS_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLS_CAT),
    ])


def candidatos():
    return {
        "Regresión Logística": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                                random_state=SEMILLA, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=SEMILLA),
    }


@st.cache_resource(show_spinner="Entrenando el modelo con la base de clientes (solo la primera vez)...")
def entrenar():
    df, origen = cargar_base()
    df = df.dropna(subset=COLS_X + ["abandono"])
    X, y = df[COLS_X], df["abandono"].astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)

    puntajes = {}
    for nombre, algoritmo in candidatos().items():
        pipe = Pipeline([("pre", nuevo_preprocesador()), ("m", algoritmo)])
        p = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        puntajes[nombre] = (p.mean(), p.std())
    mejor = max(puntajes, key=lambda k: puntajes[k][0])

    base = Pipeline([("pre", nuevo_preprocesador()), ("m", candidatos()[mejor])])
    calibrado = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    # Probabilidades "fuera de muestra": cada cliente se predice con un modelo que NO lo vio entrenar.
    prob_oof = cross_val_predict(calibrado, X, y, cv=cv, method="predict_proba")[:, 1]
    calibrado.fit(X, y)

    imp = permutation_importance(calibrado, X, y, scoring="roc_auc", n_repeats=5,
                                 random_state=SEMILLA, n_jobs=1)
    importancia = pd.Series(imp.importances_mean, index=COLS_X).sort_values(ascending=False)

    referencia = {c: (X[c].median() if c in COLS_NUM else X[c].mode()[0]) for c in COLS_X}
    return dict(modelo=calibrado, mejor=mejor, puntajes=puntajes, prob_oof=prob_oof, y=y.values,
                prima=X["prima_mensual"].values, importancia=importancia, referencia=referencia,
                origen=origen, n=len(df), tasa=float(y.mean()),
                auc_oof=float(roc_auc_score(y, prob_oof)))


def explicar_cliente(modelo, fila, referencia, prob_actual):
    """Para cada variable: ¿cuánto cambia la probabilidad si ese dato fuera 'típico'?
    Positivo = ese dato está AUMENTANDO el riesgo de este cliente."""
    efectos = {}
    for col in COLS_X:
        alterno = fila.copy()
        alterno[col] = referencia[col]
        efectos[col] = prob_actual - float(modelo.predict_proba(alterno)[0, 1])
    return pd.Series(efectos).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# 3. Reglas de negocio
# ----------------------------------------------------------------------------
def nivel_riesgo(p):
    if p >= 0.60:
        return "Alto"
    if p >= 0.35:
        return "Medio"
    return "Bajo"


ACCIONES = {
    "Alto": "Llamada personal de un asesor en las próximas 48 horas y oferta de retención.",
    "Medio": "Enviar encuesta de satisfacción o un beneficio (descuento en renovación).",
    "Bajo": "Sin acción especial; mantener comunicación habitual.",
}
ETIQUETAS = {
    "edad": "edad", "antiguedad_anios": "antigüedad", "tipo_cobertura": "tipo de cobertura",
    "prima_mensual": "prima mensual", "pagos_atrasados": "pagos atrasados", "accidentes": "accidentes",
    "reclamaciones": "reclamaciones", "satisfaccion": "satisfacción", "contactos_servicio": "contactos con servicio",
    "renovaciones": "renovaciones",
}


def beneficio_neto(prob, real, prima, umbral, costo_contacto, exito):
    """Ahorro estimado si se contacta a quienes tienen prob >= umbral."""
    contactados = prob >= umbral
    ingreso = np.sum(real[contactados] * exito * prima[contactados] * 12)
    gasto = contactados.sum() * costo_contacto
    return ingreso - gasto, int(contactados.sum())


def a_excel(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Clientes en riesgo")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# 4. Interfaz
# ----------------------------------------------------------------------------
M = entrenar()
modelo = M["modelo"]

st.title("🚗 Sistema inteligente de predicción de abandono de clientes")
st.caption("Empresa hipotética de seguros de automóviles · Proyecto académico con datos simulados "
           "(no representa clientes reales).")

with st.sidebar:
    st.header("Supuestos del negocio")
    st.write("Puedes cambiarlos; afectan el cálculo de ahorro estimado.")
    costo_contacto = st.number_input("Costo de contactar a un cliente (US$)", 0.0, 500.0, 60.0, 5.0)
    exito = st.slider("Probabilidad de retener a un cliente contactado", 0.05, 0.90, 0.20, 0.05)
    st.divider()
    st.markdown(f"**Modelo elegido:** {M['mejor']}")
    st.markdown(f"**Datos:** {M['n']:,} clientes ({M['origen']})")
    st.markdown(f"**Abandono en la base:** {M['tasa']:.1%}")

tab1, tab2, tab3 = st.tabs(["👤 Un cliente", "📋 Lista de clientes (Excel)", "📊 Rendimiento y valor para el negocio"])

# ---------------------------- Pestaña 1 ----------------------------
with tab1:
    st.subheader("Ingresa los datos del cliente o las respuestas de su encuesta")
    c1, c2, c3 = st.columns(3)
    with c1:
        edad = st.slider("Edad", 18, 75, 40)
        antig = st.slider("Años como cliente", 1, 15, 3)
        cobertura = st.selectbox("Tipo de cobertura", ["Basica", "Intermedia", "Completa"], index=1)
    with c2:
        prima = st.number_input("Prima mensual (US$)", 20.0, 180.0, 70.0, 1.0)
        pagos = st.number_input("Pagos atrasados (último año)", 0, 12, 0)
        contactos = st.number_input("Veces que contactó al servicio al cliente", 0, 15, 1)
    with c3:
        accidentes = st.number_input("Accidentes registrados", 0, 8, 0)
        reclam = st.number_input("Reclamaciones presentadas", 0, 8, 0)
        sat = st.slider("Satisfacción (encuesta: 1 = muy mala, 5 = excelente)", 1, 5, 3)

    fila = pd.DataFrame([dict(edad=edad, antiguedad_anios=antig, tipo_cobertura=cobertura, prima_mensual=prima,
                              pagos_atrasados=pagos, accidentes=accidentes, reclamaciones=reclam,
                              satisfaccion=sat, contactos_servicio=contactos, renovaciones=max(antig - 1, 0))])

    if st.button("Calcular riesgo de abandono", type="primary"):
        p = float(modelo.predict_proba(fila[COLS_X])[0, 1])
        nivel = nivel_riesgo(p)
        a, b = st.columns(2)
        a.metric("Probabilidad de abandono", f"{p:.1%}")
        b.metric("Nivel de riesgo", nivel)
        st.progress(min(max(p, 0.0), 1.0))
        {"Alto": st.error, "Medio": st.warning, "Bajo": st.success}[nivel](f"**Acción sugerida:** {ACCIONES[nivel]}")

        efectos = explicar_cliente(modelo, fila[COLS_X], M["referencia"], p)
        sube = efectos[efectos > 0.005].head(3)
        baja = efectos[efectos < -0.005].tail(3).iloc[::-1]
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Factores que más AUMENTAN su riesgo**")
            if len(sube):
                for k, v in sube.items():
                    st.write(f"• {ETIQUETAS[k]} (+{v * 100:.1f} puntos)")
            else:
                st.write("Ninguno destacado.")
        with e2:
            st.markdown("**Factores que más lo REDUCEN**")
            if len(baja):
                for k, v in baja.items():
                    st.write(f"• {ETIQUETAS[k]} ({v * 100:.1f} puntos)")
            else:
                st.write("Ninguno destacado.")
        st.caption("Cada factor se compara contra un cliente 'típico'. La probabilidad es una estimación, "
                   "no una certeza de que el cliente se irá.")

# ---------------------------- Pestaña 2 ----------------------------
with tab2:
    st.subheader("Evalúa a muchos clientes a la vez")
    st.write("Sube un archivo Excel (.xlsx) o CSV con las mismas columnas de la base de ejemplo. "
             "La columna `abandono` es opcional.")
    subido = st.file_uploader("Archivo de clientes", type=["xlsx", "csv"])
    usar_demo = st.checkbox("No tengo archivo: usar 200 clientes de la base de ejemplo")

    datos = None
    try:
        if subido is not None:
            datos = leer_tabla(subido)
        elif usar_demo:
            datos = cargar_base()[0].sample(200, random_state=1).reset_index(drop=True)
    except Exception as e:
        st.error(f"No pude leer el archivo: {e}")

    if datos is not None:
        faltan = [c for c in COLS_X if c not in datos.columns]
        if faltan:
            st.error("Faltan estas columnas en tu archivo: " + ", ".join(faltan))
        else:
            validos = datos.dropna(subset=COLS_X).copy()
            if len(validos) < len(datos):
                st.warning(f"Se omitieron {len(datos) - len(validos)} filas con datos vacíos.")
            validos["probabilidad_abandono"] = modelo.predict_proba(validos[COLS_X])[:, 1].round(4)
            validos["nivel_riesgo"] = validos["probabilidad_abandono"].apply(nivel_riesgo)
            validos["accion_sugerida"] = validos["nivel_riesgo"].map(ACCIONES)
            validos = validos.sort_values("probabilidad_abandono", ascending=False).reset_index(drop=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Clientes evaluados", f"{len(validos):,}")
            m2.metric("Riesgo alto", int((validos["nivel_riesgo"] == "Alto").sum()))
            m3.metric("Riesgo medio", int((validos["nivel_riesgo"] == "Medio").sum()))

            contactar = validos[validos["probabilidad_abandono"] >= 0.35]
            esperado = float((contactar["probabilidad_abandono"] * exito * contactar["prima_mensual"] * 12).sum()
                             - len(contactar) * costo_contacto)
            st.info(f"Si contactas a los {len(contactar)} clientes con riesgo medio o alto, el ahorro neto "
                    f"estimado es de **US$ {esperado:,.0f}** al año (según los supuestos de la barra lateral).")
            st.dataframe(validos)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Descargar en Excel", a_excel(validos), "clientes_en_riesgo.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            d2.download_button("⬇️ Descargar en CSV", validos.to_csv(index=False).encode("utf-8-sig"),
                               "clientes_en_riesgo.csv", "text/csv")

# ---------------------------- Pestaña 3 ----------------------------
with tab3:
    st.subheader("¿Qué tan bueno es el modelo?")
    tabla = pd.DataFrame({k: {"ROC-AUC promedio": v[0], "Variación (±)": v[1]} for k, v in M["puntajes"].items()}).T
    st.write("Comparación con validación cruzada de 5 grupos (el dato se evalúa en clientes que el modelo no vio):")
    st.dataframe(tabla.round(3))
    st.caption("ROC-AUC: 0.5 = azar, 1.0 = perfecto. Como los datos son simulados con una regla sencilla, "
               "es normal que los modelos simples rindan igual o mejor que los complejos.")

    st.subheader("¿Qué variables influyen más?")
    st.bar_chart(M["importancia"].rename(index=ETIQUETAS))

    st.subheader("¿A quién conviene contactar? (valor para el negocio)")
    umbral = st.slider("Contactar a los clientes con probabilidad de abandono mayor o igual a:", 0.05, 0.95, 0.35, 0.05)
    grid = np.arange(0.05, 0.96, 0.05)
    curva = pd.DataFrame({"Umbral": grid,
                          "Ahorro neto (US$)": [beneficio_neto(M["prob_oof"], M["y"], M["prima"], u,
                                                               costo_contacto, exito)[0] for u in grid]}).set_index("Umbral")
    neto, n_cont = beneficio_neto(M["prob_oof"], M["y"], M["prima"], umbral, costo_contacto, exito)
    pred = M["prob_oof"] >= umbral
    tp = int((pred & (M["y"] == 1)).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int((M["y"] == 1).sum()), 1)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes a contactar", f"{n_cont:,}")
    k2.metric("De cada 100 contactados, se iban", f"{precision * 100:.0f}")
    k3.metric("Abandonos detectados", f"{recall:.0%}")
    k4.metric("Ahorro neto estimado", f"US$ {neto:,.0f}")
    st.line_chart(curva)
    st.caption("Subir el umbral = menos llamadas pero más abandonos sin detectar. Bajarlo = más llamadas, "
               "más abandonos detectados. El mejor punto depende del costo de contacto y del valor del cliente. "
               "Estimación sobre la base simulada; con datos reales los números serían distintos.")
