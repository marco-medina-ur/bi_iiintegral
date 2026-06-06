# ============================================================
# app.py  –  ConecTel · Predictor de Mora (Streamlit Cloud)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ── Configuración de la página ───────────────────────────────
st.set_page_config(
    page_title="ConecTel – Predictor de Mora",
    page_icon="📡",
    layout="wide",
)

# ── Estilos ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #a8b2d8; margin: 0.4rem 0 0; font-size: 1rem; }

    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .risk-alto  { background:#ffe5e5; color:#c0392b; border:2px solid #e74c3c; }
    .risk-medio { background:#fff8e1; color:#e67e22; border:2px solid #f39c12; }
    .risk-bajo  { background:#e8f8f0; color:#1e8449; border:2px solid #27ae60; }

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
""", unsafe_allow_html=True)


# ── Entrenamiento del modelo (se cachea; solo corre una vez) ──
@st.cache_resource(show_spinner="Cargando modelo... un momento ⏳")
def entrenar_modelo():
    df = pd.read_csv('clientes_conectel_v2.csv')

    # Errores de formato → numérico
    cols_formato = [
        col for col in df.select_dtypes(include='object').columns
        if pd.to_numeric(df[col], errors='coerce').notna().mean() > 0.9
    ]
    for col in cols_formato:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Duplicados
    df = df.drop_duplicates(keep='first').reset_index(drop=True)

    # Nulos
    df.loc[df['tiene_internet'] == 0, 'velocidad_mbps'] = \
        df.loc[df['tiene_internet'] == 0, 'velocidad_mbps'].fillna(0)
    mediana_vel = df.loc[df['tiene_internet'] == 1, 'velocidad_mbps'].median()
    df.loc[df['tiene_internet'] == 1, 'velocidad_mbps'] = \
        df.loc[df['tiene_internet'] == 1, 'velocidad_mbps'].fillna(mediana_vel)

    df['genero'] = df['genero'].fillna(df['genero'].mode()[0])
    for col in ['ingreso_estimado_clp','meses_sin_reajuste','nps','edad',
                'llamadas_soporte_6m','reclamos_12m','dias_mora_hist','cambios_plan_12m']:
        df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    df_modelo = df.drop(columns=['customer_id'])

    cols_binarias = [
        col for col in df_modelo.select_dtypes(include='object').columns
        if df_modelo[col].dropna().isin(['Sí', 'No']).all()
    ]
    for col in cols_binarias:
        df_modelo[col] = (df_modelo[col] == 'Sí').astype(int)

    cols_ohe = df_modelo.select_dtypes(include=['object','category'])\
                        .columns.difference(cols_binarias + ['mora_90d']).tolist()
    df_modelo = pd.get_dummies(df_modelo, columns=cols_ohe, drop_first=True)

    df_modelo['ratio_factura_ingreso'] = (
        df_modelo['factura_mensual_clp'] / df_modelo['ingreso_estimado_clp'].replace(0, np.nan)
    ).fillna(0)
    df_modelo['indice_conflictividad'] = (
        df_modelo['llamadas_soporte_6m'] + df_modelo['reclamos_12m']
    )

    X = df_modelo.drop(columns=['mora_90d'])
    y = df_modelo['mora_90d']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    return modelo, list(X.columns)


modelo, columnas_modelo = entrenar_modelo()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 ConecTel – Predictor de Riesgo de Mora</h1>
    <p>Ingresa los datos del cliente para estimar la probabilidad de mora mayor a 90 días</p>
</div>
""", unsafe_allow_html=True)

# ── Formulario ───────────────────────────────────────────────
with st.form("formulario_cliente"):

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown('<div class="section-title">👤 Datos Personales</div>', unsafe_allow_html=True)
        edad                 = st.number_input("Edad (años)", min_value=18, max_value=100, value=35)
        genero               = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
        region               = st.selectbox("Región", [
            "Región Metropolitana", "Valparaíso", "Biobío",
            "La Araucanía", "Los Lagos", "Antofagasta", "Otra"
        ])
        ingreso_estimado_clp = st.number_input(
            "Ingreso estimado (CLP)", min_value=0, max_value=10_000_000,
            value=600_000, step=50_000
        )

        st.markdown('<div class="section-title">📋 Datos del Contrato</div>', unsafe_allow_html=True)
        tipo_contrato        = st.selectbox("Tipo de contrato", ["Mensual", "Anual", "Bianual"])
        plan                 = st.selectbox("Plan contratado", ["Básico", "Estándar", "Premium", "Empresarial"])
        antiguedad_meses     = st.number_input("Antigüedad (meses)", min_value=0, max_value=360, value=24)
        factura_mensual_clp  = st.number_input(
            "Factura mensual (CLP)", min_value=0, max_value=500_000,
            value=25_000, step=1_000
        )
        meses_sin_reajuste   = st.number_input("Meses sin reajuste de precio", min_value=0, max_value=120, value=6)

    with col_der:
        st.markdown('<div class="section-title">💳 Método de Pago y Descuentos</div>', unsafe_allow_html=True)
        metodo_pago      = st.selectbox("Método de pago", ["PAT", "Transferencia", "Efectivo", "Tarjeta de crédito"])
        descuento_activo = st.radio("¿Tiene descuento activo?", ["No", "Sí"], horizontal=True)

        st.markdown('<div class="section-title">📡 Servicios Contratados</div>', unsafe_allow_html=True)
        num_servicios     = st.slider("Número de servicios", 1, 5, 2)
        tiene_internet    = st.radio("¿Tiene internet?",    ["No", "Sí"], horizontal=True)
        velocidad_mbps    = st.number_input(
            "Velocidad internet (Mbps)", min_value=0, max_value=1000, value=100,
            disabled=(tiene_internet == "No")
        )
        tiene_tv          = st.radio("¿Tiene TV?",          ["No", "Sí"], horizontal=True)
        tiene_linea_movil = st.radio("¿Tiene línea móvil?", ["No", "Sí"], horizontal=True)

        st.markdown('<div class="section-title">📞 Historial de Servicio</div>', unsafe_allow_html=True)
        llamadas_soporte_6m = st.number_input("Llamadas a soporte (últimos 6 meses)",  min_value=0, max_value=50, value=1)
        reclamos_12m        = st.number_input("Reclamos (últimos 12 meses)",           min_value=0, max_value=50, value=0)
        dias_mora_hist      = st.number_input("Días de mora histórica",                min_value=0, max_value=365, value=0)
        cambios_plan_12m    = st.number_input("Cambios de plan (últimos 12 meses)",    min_value=0, max_value=20, value=0)
        nps                 = st.slider("NPS del cliente (-100 a 100)", -100, 100, 30)

    st.markdown("---")
    predecir = st.form_submit_button(
        "🔍 Calcular Riesgo de Mora",
        use_container_width=True,
        type="primary"
    )

# ── Predicción ───────────────────────────────────────────────
if predecir:

    tiene_internet_bin    = 1 if tiene_internet    == "Sí" else 0
    tiene_tv_bin          = 1 if tiene_tv          == "Sí" else 0
    tiene_linea_movil_bin = 1 if tiene_linea_movil == "Sí" else 0
    descuento_activo_bin  = 1 if descuento_activo  == "Sí" else 0
    velocidad_final       = velocidad_mbps if tiene_internet_bin == 1 else 0

    datos_raw = {
        'edad':                 edad,
        'genero':               genero,
        'region':               region,
        'ingreso_estimado_clp': ingreso_estimado_clp,
        'tipo_contrato':        tipo_contrato,
        'plan':                 plan,
        'antiguedad_meses':     antiguedad_meses,
        'factura_mensual_clp':  factura_mensual_clp,
        'meses_sin_reajuste':   meses_sin_reajuste,
        'metodo_pago':          metodo_pago,
        'descuento_activo':     descuento_activo_bin,
        'num_servicios':        num_servicios,
        'tiene_internet':       tiene_internet_bin,
        'velocidad_mbps':       velocidad_final,
        'tiene_tv':             tiene_tv_bin,
        'tiene_linea_movil':    tiene_linea_movil_bin,
        'llamadas_soporte_6m':  llamadas_soporte_6m,
        'reclamos_12m':         reclamos_12m,
        'dias_mora_hist':       dias_mora_hist,
        'cambios_plan_12m':     cambios_plan_12m,
        'nps':                  nps,
    }
    df_input = pd.DataFrame([datos_raw])

    # OHE
    cols_ohe_app = ['genero', 'region', 'tipo_contrato', 'plan', 'metodo_pago']
    df_input = pd.get_dummies(df_input, columns=cols_ohe_app)

    # Variables derivadas
    df_input['ratio_factura_ingreso'] = (
        df_input['factura_mensual_clp'] / df_input['ingreso_estimado_clp'].replace(0, np.nan)
    ).fillna(0)
    df_input['indice_conflictividad'] = (
        df_input['llamadas_soporte_6m'] + df_input['reclamos_12m']
    )

    # Alinear columnas
    for col in columnas_modelo:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[columnas_modelo]

    # Predicción
    prob     = modelo.predict_proba(df_input)[0][1]
    prob_pct = prob * 100
    pred_bin = int(prob >= 0.5)

    if prob_pct >= 60:
        nivel, clase_css, emoji = "ALTO",  "risk-alto",  "🔴"
    elif prob_pct >= 30:
        nivel, clase_css, emoji = "MEDIO", "risk-medio", "🟡"
    else:
        nivel, clase_css, emoji = "BAJO",  "risk-bajo",  "🟢"

    # ── Resultados ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Resultado de la Evaluación")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(
            f'<div class="risk-card {clase_css}">{emoji} Riesgo {nivel}</div>',
            unsafe_allow_html=True
        )
    with r2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Probabilidad de Mora", f"{prob_pct:.1f}%")
        st.progress(min(prob, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Decisión del Modelo", "⚠️ En riesgo" if pred_bin else "✅ Sin riesgo")
        st.caption("Umbral de decisión: 50%")
        st.markdown('</div>', unsafe_allow_html=True)

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
        resumen = pd.DataFrame({
            'Variable': [
                'Edad', 'Ingreso estimado', 'Factura mensual', 'Antigüedad',
                'NPS', 'Días mora histórica', 'Llamadas soporte (6m)',
                'Reclamos (12m)', 'Cambios de plan (12m)',
                'Ratio factura/ingreso', 'Índice conflictividad'
            ],
            'Valor': [
                edad, f"${ingreso_estimado_clp:,}", f"${factura_mensual_clp:,}",
                f"{antiguedad_meses} meses", nps, dias_mora_hist,
                llamadas_soporte_6m, reclamos_12m, cambios_plan_12m,
                f"{df_input['ratio_factura_ingreso'].values[0]:.4f}",
                f"{df_input['indice_conflictividad'].values[0]:.0f}"
            ]
        })
        st.dataframe(resumen, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("ConecTel · Modelo: Random Forest · IICG514 Business Intelligence")
