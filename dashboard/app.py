# =========================================================
# INDUSTRIAL PREDICTIVE MAINTENANCE DASHBOARD (FINAL UI)
# =========================================================

import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL
model = joblib.load("models/model.pkl")

# PAGE CONFIG
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# TITLE
st.title("🏭 Industrial Predictive Maintenance Dashboard")
st.caption("NASA CMAPSS-based AI System for Machine Health Monitoring")

# =========================================================
# SENSOR NAME MAPPING (INTERPRETED VIEW)
# =========================================================

sensor_map = {
    "s1": "Engine Temperature 🌡️",
    "s2": "Compressor Pressure 🧪",
    "s3": "Fan Speed ⚙️",
    "s4": "Fuel Flow Rate ⛽",
    "s5": "Exhaust Gas Pressure 💨",
    "s6": "Vibration Level 🎢",
    "s7": "Oil Pressure 🛢️",
    "s8": "Rotor Speed ⚙️",
    "s9": "Heat Flux 🔥",
    "s10": "Air Intake Pressure 🌬️",
    "s11": "Core Temperature 🔥",
    "s12": "Shaft Vibration 🎢",
    "s13": "Fuel Efficiency ⛽",
    "s14": "Compressor Efficiency 🧪",
    "s15": "Engine Load ⚙️",
    "s16": "Bearing Health 🧿",
    "s17": "Turbine Pressure 💨",
    "s18": "Oil Temperature 🛢️",
    "s19": "Speed Variation 📊",
    "s20": "Combustion Stability 🔥",
    "s21": "Air Pressure Ratio 🌬️",
    "s22": "Sensor A",
    "s23": "Sensor B",
    "s24": "Sensor C"
}

# =========================================================
# SIDEBAR INPUT
# =========================================================

st.sidebar.title("🔧 Engine Sensor Input Panel")
st.sidebar.markdown("Enter Sensor Values")

data = {}

for i in range(1, 25):
    key = f"s{i}"
    data[key] = st.sidebar.slider(sensor_map[key], -1.0, 1.0, 0.0)

input_df = pd.DataFrame([data])

# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# =========================================================
# MAIN STATUS SECTION
# =========================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Machine Status")

    if prediction == 1:
        st.error("⚠ CRITICAL FAILURE RISK DETECTED")
    else:
        st.success("✅ MACHINE OPERATING NORMALLY")

with col2:
    st.subheader("📊 Failure Risk Score")
    st.metric("Probability", f"{probability:.3f}")

# PROGRESS BAR
st.progress(int(probability * 100))

# =========================================================
# SENSOR TABLE (INTERPRETED)
# =========================================================

st.subheader("📡 Sensor Readings (Interpreted View)")

display_df = pd.DataFrame({
    "Sensor": [sensor_map[f"s{i}"] for i in range(1, 25)],
    "Value": [input_df[f"s{i}"].values[0] for i in range(1, 25)]
})

st.dataframe(display_df, use_container_width=True)

# =========================================================
# AI INSIGHT SECTION
# =========================================================

st.subheader("📉 AI Insight")

if probability < 0.3:
    st.info("System is stable. No immediate maintenance required.")
elif probability < 0.6:
    st.warning("Early warning detected. Monitor system closely.")
else:
    st.error("Immediate maintenance required! High failure probability.")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("🚀 AI-Based Predictive Maintenance System | Industry Simulation")