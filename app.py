import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ==============================
# Cargar paquete del modelo
# ==============================

with open("paquete_streamlit_v1.pkl", "rb") as f:
    paquete = pickle.load(f)

modelo = paquete["modelo"]
nombre_modelo = paquete["nombre_modelo"]
features_all = paquete["features_all"]
features_num = paquete["features_num"]
features_bin = paquete["features_bin"]
scaler = paquete["scaler"]
target_map = paquete["target_map"]

# ==============================
# Funciones auxiliares
# ==============================

def clasificar_facilidad(nombre):
    n = str(nombre).upper()

    if "FILTRO" in n:
        return "Filtro"
    if "CELDA" in n:
        return "Celda"
    if "CPI" in n:
        return "CPI"
    if "TANQUE" in n:
        return "Tanque"
    if "CPL" in n:
        return "CPL"
    if "CPN" in n or "CP " in n or "CPS" in n:
        return "CPN"
    if "ENTRADA" in n:
        return "Entrada"
    if "SALIDA" in n:
        return "Salida"
    if "STAP" in n:
        return "STAP"

    return "Otro"


def clasificar_regla(sst, grasas, turbidez):
    if (sst <= 2.5) and (grasas <= 5) and (turbidez <= 2):
        return "Buena"

    if (sst > 2.6) or (grasas > 6) or (turbidez > 3):
        return "Mala"

    return "Intermedia"


def nivel_riesgo(prob_mala):
    if prob_mala < 0.40:
        return "Bajo"
    elif prob_mala < 0.60:
        return "Intermedio"
    return "Alto"


def preparar_dato(fecha, facilidad, sst, grasas, turbidez):
    tipo = clasificar_facilidad(facilidad)

    promedio_target = np.mean(list(target_map.values()))

    dato = pd.DataFrame(
        np.zeros((1, len(features_all))),
        columns=features_all
    )

    valores_base = {
        "SST": sst,
        "grasas": grasas,
        "turbidez": turbidez,
        "SST_log": np.log1p(sst),
        "grasas_log": np.log1p(grasas),
        "turbidez_log": np.log1p(turbidez),
        "facilidad_target_enc": target_map.get(facilidad, promedio_target),
        "año": fecha.year,
        "mes": fecha.month,
        "grasas_flag_NR": 0,
        "grasas_flag_ND": 0,
        "turbidez_flag_NR": 0,
    }

    for col, val in valores_base.items():
        if col in dato.columns:
            dato.loc[0, col] = val

    col_tipo = f"tipo_{tipo}"
    if col_tipo in dato.columns:
        dato.loc[0, col_tipo] = 1

    dato = dato[features_all]

    columnas_scaler = list(scaler.feature_names_in_)

    dato[columnas_scaler] = scaler.transform(
        dato[columnas_scaler]
    )

    return dato

# ==============================
# Configuración de página
# ==============================

st.set_page_config(
    page_title="Modelo de Calidad de Agua",
    layout="wide"
)

st.title("Modelo predictivo de calidad de agua de producción")

st.write(
    "Prototipo funcional para evaluar la calidad del agua en sistemas de tratamiento de agua de producción "
    "a partir de variables fisicoquímicas y condiciones básicas del punto de muestreo."
)

# ==============================
# Sidebar - Entrada de datos
# ==============================

st.sidebar.header("Ingreso de datos")

fecha = st.sidebar.date_input("Fecha de medición")
facilidad = st.sidebar.text_input("Facilidad", value="Salida STAP")

sst = st.sidebar.number_input("SST", min_value=0.0, value=2.0, step=0.1)
grasas = st.sidebar.number_input("Grasas y aceites", min_value=0.0, value=4.0, step=0.1)
turbidez = st.sidebar.number_input("Turbidez", min_value=0.0, value=1.5, step=0.1)

evaluar = st.sidebar.button("Evaluar muestra")

# ==============================
# Ejecución
# ==============================

if evaluar:
    calidad_regla = clasificar_regla(sst, grasas, turbidez)

    X_usuario = preparar_dato(
        fecha=fecha,
        facilidad=facilidad,
        sst=sst,
        grasas=grasas,
        turbidez=turbidez
    )

    pred = modelo.predict(X_usuario)[0]
    prob = modelo.predict_proba(X_usuario)[0]

    calidad_modelo = "Mala" if pred == 1 else "Buena"
    prob_buena = prob[0]
    prob_mala = prob[1]
    riesgo = nivel_riesgo(prob_mala)

    # ==============================
    # Métricas principales
    # ==============================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Regla operacional", calidad_regla)

    with col2:
        st.metric(f"Predicción ML ({nombre_modelo})", calidad_modelo)

    with col3:
        st.metric("Nivel de riesgo", riesgo)

    st.divider()

    # ==============================
    # Gráficas
    # ==============================

    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Probabilidad de clasificación")

        df_prob = pd.DataFrame({
            "Clase": ["Buena", "Mala"],
            "Probabilidad": [prob_buena, prob_mala]
        })

        fig, ax = plt.subplots()

        colores = ["#2E8B57", "#B22222"]

        barras = ax.bar(
            df_prob["Clase"],
            df_prob["Probabilidad"],
            color=colores
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilidad")
        ax.set_title("Probabilidad estimada por el modelo")

        for barra in barras:
            altura = barra.get_height()
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                altura + 0.02,
                f"{altura:.1%}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        st.pyplot(fig)

    with col5:
        st.subheader("Comparación contra límites operacionales")

        df_limites = pd.DataFrame({
            "Variable": ["SST", "Grasas", "Turbidez"],
            "Valor ingresado": [sst, grasas, turbidez],
            "Límite buena calidad": [2.5, 5, 2]
        })

        fig2, ax2 = plt.subplots()

        barras2 = ax2.bar(
            df_limites["Variable"],
            df_limites["Valor ingresado"],
            color="#4682B4",
            label="Valor ingresado"
        )

        ax2.plot(
            df_limites["Variable"],
            df_limites["Límite buena calidad"],
            marker="o",
            linestyle="--",
            color="#B22222",
            label="Límite buena calidad"
        )

        ax2.set_ylabel("Valor")
        ax2.set_title("Valores ingresados vs límites")
        ax2.legend()

        for barra in barras2:
            altura = barra.get_height()
            ax2.text(
                barra.get_x() + barra.get_width() / 2,
                altura + 0.05,
                f"{altura:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        for i, valor in enumerate(df_limites["Límite buena calidad"]):
            ax2.text(
                i,
                valor + 0.05,
                f"{valor:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                color="#B22222"
            )

        st.pyplot(fig2)

    # ==============================
    # Interpretación
    # ==============================

    st.subheader("Interpretación operativa")

    if calidad_regla == "Buena" and calidad_modelo == "Buena":
        st.success(
            "El agua cumple los criterios operacionales y el modelo confirma una condición favorable para inyección."
        )

    elif calidad_regla == "Mala" and calidad_modelo == "Mala":
        st.error(
            "El agua presenta condiciones deficientes y el modelo confirma una alta probabilidad de mala calidad."
        )

    elif calidad_regla == "Buena" and calidad_modelo == "Mala":
        st.warning(
            "Aunque el agua cumple la regla operacional, el modelo identifica un posible riesgo según el comportamiento histórico aprendido."
        )

    elif calidad_regla == "Mala" and calidad_modelo == "Buena":
        st.warning(
            "La regla operacional clasifica el agua como mala, pero el modelo estima una condición favorable. Se recomienda revisar el contexto operativo."
        )

    else:
        st.info(
            "La muestra se encuentra en una zona intermedia. Se recomienda realizar análisis adicional."
        )

    # ==============================
    # Tabla resumen
    # ==============================

    st.subheader("Resumen de la muestra evaluada")

    resultado = pd.DataFrame({
        "fecha": [fecha],
        "facilidad": [facilidad],
        "SST": [sst],
        "grasas": [grasas],
        "turbidez": [turbidez],
        "calidad_regla": [calidad_regla],
        "calidad_modelo": [calidad_modelo],
        "probabilidad_buena": [round(prob_buena, 4)],
        "probabilidad_mala": [round(prob_mala, 4)],
        "riesgo": [riesgo]
    })

    st.dataframe(resultado)

else:
    st.info("Ingrese los valores de la muestra en el panel lateral y presione 'Evaluar muestra'.")