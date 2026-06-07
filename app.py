from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# Configuración general
# ============================================================

CSV_PATH = Path("clientes_conectel_v2.csv")
UMBRAL_DECISION = 0.30

st.set_page_config(
    page_title="ConecTel – Predictor de Mora",
    page_icon="📡",
    layout="wide",
)

# ============================================================
# Estilos
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e94560;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #a8b2d8;
        margin: 0.4rem 0 0;
        font-size: 1rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .risk-alto {
        background:#ffe5e5;
        color:#c0392b;
        border:2px solid #e74c3c;
    }
    .risk-medio {
        background:#fff8e1;
        color:#e67e22;
        border:2px solid #f39c12;
    }
    .risk-bajo {
        background:#e8f8f0;
        color:#1e8449;
        border:2px solid #27ae60;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        border-left: 4px solid #e94560;
        padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Funciones auxiliares
# ============================================================


def convertir_columnas_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas tipo texto que en realidad contienen números."""
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        serie_limpia = (
            df[col]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )

        serie_convertida = pd.to_numeric(serie_limpia, errors="coerce")
        proporcion_numerica = serie_convertida.notna().mean()

        if proporcion_numerica > 0.90:
            df[col] = serie_convertida

    return df


def normalizar_binarias(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza variables binarias Sí/No a 1/0 cuando corresponde."""
    df = df.copy()

    columnas_binarias = [
        "descuento_activo",
        "tiene_internet",
        "tiene_tv",
        "tiene_linea_movil",
    ]

    mapa_binario = {
        "sí": 1,
        "si": 1,
        "s": 1,
        "yes": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }

    for col in columnas_binarias:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map(mapa_binario)
                    .fillna(df[col])
                )

            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas simples de imputación usadas para el despliegue."""
    df = df.copy()

    # Género
    if "genero" in df.columns:
        if df["genero"].notna().any():
            df["genero"] = df["genero"].fillna(df["genero"].mode(dropna=True)[0])
        else:
            df["genero"] = df["genero"].fillna("Prefiero no decir")

    # Velocidad de internet
    if {"tiene_internet", "velocidad_mbps"}.issubset(df.columns):
        df.loc[df["tiene_internet"] == 0, "velocidad_mbps"] = (
            df.loc[df["tiene_internet"] == 0, "velocidad_mbps"].fillna(0)
        )

        mediana_velocidad = df.loc[df["tiene_internet"] == 1, "velocidad_mbps"].median()

        if pd.isna(mediana_velocidad):
            mediana_velocidad = df["velocidad_mbps"].median()

        df.loc[df["tiene_internet"] == 1, "velocidad_mbps"] = (
            df.loc[df["tiene_internet"] == 1, "velocidad_mbps"].fillna(mediana_velocidad)
        )

        df["velocidad_mbps"] = df["velocidad_mbps"].fillna(0)

    columnas_numericas = [
        "ingreso_estimado_clp",
        "meses_sin_reajuste",
        "nps",
        "edad",
        "llamadas_soporte_6m",
        "reclamos_12m",
        "dias_mora_hist",
        "cambios_plan_12m",
        "antiguedad_meses",
        "factura_mensual_clp",
        "num_servicios",
    ]

    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mediana = df[col].median()

            if pd.isna(mediana):
                mediana = 0

            df[col] = df[col].fillna(mediana)

    # Si quedan columnas categóricas con nulos, se deja una categoría explícita.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Sin información")

    return df


def agregar_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables de feature engineering usadas por el modelo."""
    df = df.copy()

    df["ratio_factura_ingreso"] = (
        df["factura_mensual_clp"] / df["ingreso_estimado_clp"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    df["indice_conflictividad"] = (
        df["llamadas_soporte_6m"] + df["reclamos_12m"]
    )

    df["antiguedad_tramo"] = pd.cut(
        df["antiguedad_meses"],
        bins=[-1, 12, 36, np.inf],
        labels=["Nueva", "Intermedia", "Antigua"],
    ).astype(str)

    return df


def preparar_features(df: pd.DataFrame, columnas_modelo: list[str] | None = None):
    """
    Prepara datos para entrenamiento o predicción.

    Si columnas_modelo es None, retorna X, y y columnas.
    Si columnas_modelo viene definido, retorna X alineado al modelo.
    """
    df = df.copy()

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    y = None
    if "mora_90d" in df.columns:
        y = df["mora_90d"].astype(int)
        df = df.drop(columns=["mora_90d"])

    df = agregar_variables_derivadas(df)

    columnas_categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)

    # Asegura que todas las columnas sean numéricas
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if columnas_modelo is not None:
        for col in columnas_modelo:
            if col not in df.columns:
                df[col] = 0

        df = df[columnas_modelo]
        return df

    return df, y, list(df.columns)


# ============================================================
# Entrenamiento del modelo
# ============================================================


@st.cache_resource(show_spinner="Cargando y entrenando modelo... ⏳")
def entrenar_modelo():
    if not CSV_PATH.exists():
        st.error(
            "No se encontró el archivo `clientes_conectel_v2.csv`. "
            "Déjalo en la misma carpeta que `app.py` y vuelve a ejecutar la aplicación."
        )
        st.stop()

    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    if "mora_90d" not in df.columns:
        st.error("El dataset debe incluir la columna objetivo `mora_90d`.")
        st.stop()

    df = convertir_columnas_numericas(df)
    df = normalizar_binarias(df)
    df = imputar_nulos(df)

    X, y, columnas = preparar_features(df)

    if y.nunique() < 2:
        st.error("La variable `mora_90d` debe tener al menos dos clases: 0 y 1.")
        st.stop()

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    modelo = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=1000,
    )
    modelo.fit(X_train_scaled, y_train)

    return modelo, scaler, columnas


modelo, scaler, columnas_modelo = entrenar_modelo()

# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>📡 ConecTel – Predictor de Riesgo de Mora</h1>
        <p>Ingresa los datos del cliente para estimar la probabilidad de mora mayor a 90 días</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Formulario
# ============================================================

with st.form("formulario_cliente"):
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown(
            '<div class="section-title">👤 Datos Personales</div>',
            unsafe_allow_html=True,
        )

        edad = st.number_input(
            "Edad (años)",
            min_value=18,
            max_value=100,
            value=35,
        )

        genero = st.selectbox(
            "Género",
            ["Femenino", "Masculino", "No binario", "Prefiero no decir"],
        )

        region = st.selectbox(
            "Región",
            [
                "Antofagasta",
                "Araucanía",
                "Atacama",
                "Biobío",
                "Coquimbo",
                "Los Lagos",
                "Maule",
                "Metropolitana",
                "O'Higgins",
                "Valparaíso",
            ],
        )

        ingreso_estimado_clp = st.number_input(
            "Ingreso estimado (CLP)",
            min_value=0,
            max_value=10_000_000,
            value=600_000,
            step=50_000,
        )

        st.markdown(
            '<div class="section-title">📋 Datos del Contrato</div>',
            unsafe_allow_html=True,
        )

        tipo_contrato = st.selectbox(
            "Tipo de contrato",
            ["Mensual", "Anual", "Bianual"],
        )

        plan = st.selectbox(
            "Plan contratado",
            ["Básico", "Estándar", "Premium"],
        )

        antiguedad_meses = st.number_input(
            "Antigüedad (meses)",
            min_value=0,
            max_value=360,
            value=24,
        )

        factura_mensual_clp = st.number_input(
            "Factura mensual (CLP)",
            min_value=0,
            max_value=500_000,
            value=25_000,
            step=1_000,
        )

        meses_sin_reajuste = st.number_input(
            "Meses sin reajuste de precio",
            min_value=0,
            max_value=120,
            value=6,
        )

    with col_der:
        st.markdown(
            '<div class="section-title">💳 Método de Pago y Descuentos</div>',
            unsafe_allow_html=True,
        )

        metodo_pago = st.selectbox(
            "Método de pago",
            ["Cheque", "Débito automático", "Efectivo", "Transferencia", "WebPay"],
        )

        descuento_activo = st.radio(
            "¿Tiene descuento activo?",
            ["No", "Sí"],
            horizontal=True,
        )

        st.markdown(
            '<div class="section-title">📡 Servicios Contratados</div>',
            unsafe_allow_html=True,
        )

        num_servicios = st.slider(
            "Número de servicios",
            min_value=1,
            max_value=5,
            value=2,
        )

        tiene_internet = st.radio(
            "¿Tiene internet?",
            ["No", "Sí"],
            horizontal=True,
        )

        velocidad_mbps = st.number_input(
            "Velocidad internet (Mbps)",
            min_value=0,
            max_value=1000,
            value=100,
            disabled=(tiene_internet == "No"),
        )

        tiene_tv = st.radio(
            "¿Tiene TV?",
            ["No", "Sí"],
            horizontal=True,
        )

        tiene_linea_movil = st.radio(
            "¿Tiene línea móvil?",
            ["No", "Sí"],
            horizontal=True,
        )

        st.markdown(
            '<div class="section-title">📞 Historial de Servicio</div>',
            unsafe_allow_html=True,
        )

        llamadas_soporte_6m = st.number_input(
            "Llamadas a soporte (últimos 6 meses)",
            min_value=0,
            max_value=50,
            value=1,
        )

        reclamos_12m = st.number_input(
            "Reclamos (últimos 12 meses)",
            min_value=0,
            max_value=50,
            value=0,
        )

        dias_mora_hist = st.number_input(
            "Días de mora histórica",
            min_value=0,
            max_value=365,
            value=0,
        )

        cambios_plan_12m = st.number_input(
            "Cambios de plan (últimos 12 meses)",
            min_value=0,
            max_value=20,
            value=0,
        )

        nps = st.slider(
            "NPS del cliente (1 a 10)",
            min_value=1,
            max_value=10,
            value=6,
        )

    st.markdown("---")

    predecir = st.form_submit_button(
        "🔍 Calcular Riesgo de Mora",
        use_container_width=True,
        type="primary",
    )

# ============================================================
# Predicción
# ============================================================

if predecir:
    tiene_internet_bin = 1 if tiene_internet == "Sí" else 0
    tiene_tv_bin = 1 if tiene_tv == "Sí" else 0
    tiene_linea_movil_bin = 1 if tiene_linea_movil == "Sí" else 0
    descuento_activo_bin = 1 if descuento_activo == "Sí" else 0
    velocidad_final = velocidad_mbps if tiene_internet_bin == 1 else 0

    datos_raw = {
        "edad": edad,
        "genero": genero,
        "region": region,
        "ingreso_estimado_clp": ingreso_estimado_clp,
        "tipo_contrato": tipo_contrato,
        "plan": plan,
        "antiguedad_meses": antiguedad_meses,
        "factura_mensual_clp": factura_mensual_clp,
        "meses_sin_reajuste": meses_sin_reajuste,
        "metodo_pago": metodo_pago,
        "descuento_activo": descuento_activo_bin,
        "num_servicios": num_servicios,
        "tiene_internet": tiene_internet_bin,
        "velocidad_mbps": velocidad_final,
        "tiene_tv": tiene_tv_bin,
        "tiene_linea_movil": tiene_linea_movil_bin,
        "llamadas_soporte_6m": llamadas_soporte_6m,
        "reclamos_12m": reclamos_12m,
        "dias_mora_hist": dias_mora_hist,
        "cambios_plan_12m": cambios_plan_12m,
        "nps": nps,
    }

    df_input_raw = pd.DataFrame([datos_raw])
    df_input_raw = normalizar_binarias(df_input_raw)
    df_input_raw = imputar_nulos(df_input_raw)

    df_input = preparar_features(
        df_input_raw,
        columnas_modelo=columnas_modelo,
    )

    df_input_scaled = scaler.transform(df_input)

    prob = modelo.predict_proba(df_input_scaled)[0][1]
    prob_pct = prob * 100
    pred_bin = int(prob >= UMBRAL_DECISION)

    if prob_pct >= 60:
        nivel, clase_css, emoji = "ALTO", "risk-alto", "🔴"
    elif prob_pct >= 30:
        nivel, clase_css, emoji = "MEDIO", "risk-medio", "🟡"
    else:
        nivel, clase_css, emoji = "BAJO", "risk-bajo", "🟢"

    # ========================================================
    # Resultados
    # ========================================================

    st.markdown("---")
    st.markdown("## 📊 Resultado de la Evaluación")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(
            f'<div class="risk-card {clase_css}">{emoji} Riesgo {nivel}</div>',
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Probabilidad de Mora", f"{prob_pct:.1f}%")
        st.progress(min(float(prob), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(
            "Decisión del Modelo",
            "⚠️ En riesgo" if pred_bin else "✅ Sin riesgo",
        )
        st.caption(f"Umbral de decisión: {UMBRAL_DECISION:.0%}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 💡 Recomendación Comercial")

    if nivel == "ALTO":
        st.error(
            "**Acción inmediata recomendada.** Este cliente presenta un perfil de alto riesgo. "
            "Se sugiere contacto preventivo, revisión de deuda activa y, si aplica, ofrecer "
            "facilidades de pago antes de que se genere la mora."
        )
    elif nivel == "MEDIO":
        st.warning(
            "**Monitoreo cercano.** El cliente muestra señales moderadas de riesgo. "
            "Se recomienda hacer seguimiento en los próximos 30 días y considerar "
            "incentivos de fidelización o descuentos para mejorar su compromiso de pago."
        )
    else:
        st.success(
            "**Cliente de bajo riesgo.** No se requieren acciones preventivas inmediatas. "
            "Buen candidato para ofertas de upgrade de plan o fidelización a largo plazo."
        )

    with st.expander("📋 Ver resumen de variables ingresadas"):
        ratio_factura_ingreso = (
            factura_mensual_clp / ingreso_estimado_clp
            if ingreso_estimado_clp > 0
            else 0
        )
        indice_conflictividad = llamadas_soporte_6m + reclamos_12m

        if antiguedad_meses <= 12:
            antiguedad_tramo = "Nueva"
        elif antiguedad_meses <= 36:
            antiguedad_tramo = "Intermedia"
        else:
            antiguedad_tramo = "Antigua"

        resumen = pd.DataFrame(
            {
                "Variable": [
                    "Edad",
                    "Género",
                    "Región",
                    "Ingreso estimado",
                    "Factura mensual",
                    "Antigüedad",
                    "Tramo de antigüedad",
                    "NPS",
                    "Días mora histórica",
                    "Llamadas soporte (6m)",
                    "Reclamos (12m)",
                    "Cambios de plan (12m)",
                    "Ratio factura/ingreso",
                    "Índice conflictividad",
                ],
                "Valor": [
                    edad,
                    genero,
                    region,
                    f"${ingreso_estimado_clp:,.0f}",
                    f"${factura_mensual_clp:,.0f}",
                    f"{antiguedad_meses} meses",
                    antiguedad_tramo,
                    nps,
                    dias_mora_hist,
                    llamadas_soporte_6m,
                    reclamos_12m,
                    cambios_plan_12m,
                    f"{ratio_factura_ingreso:.4f}",
                    f"{indice_conflictividad:.0f}",
                ],
            }
        )

        st.dataframe(
            resumen,
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption(
    "ConecTel · Modelo: Regresión Logística · "
    "Umbral operativo: 30% · IICG514 Business Intelligence"
)

