import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_autorefresh import st_autorefresh

import firebase_admin
from firebase_admin import credentials, db

# ================= PAGE CONFIG =================
st.set_page_config(page_title="IoT Based Crop Health Monitoring System", layout="wide")
st.title("IoT Based Crop Health Monitoring System 🌱")
st.divider()

# ================= AUTO REFRESH =================
st_autorefresh(interval=2000, key="sensor_refresh")

# ================= FIREBASE =================
cred = credentials.Certificate("firebase-key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://crop-health-monitoring-57ff1-default-rtdb.firebaseio.com/"
    })

# ================= SESSION STATE =================
for key in ["sensor_data", "leaf_status", "leaf_disease", "final_decision"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ================= THRESHOLDS =================
REALTIME_THRESHOLDS = {
    "Temperature": (18, 32),
    "Humidity": (40, 70),
    "Soil Moisture": (20, 80),
}

EXTRA_SENSOR_THRESHOLDS = {
    "Soil pH": (5.5, 7.5),
    "Soil Temperature (°C)": (18.0, 30.0),
    "Light Intensity": (300.0, 800.0),
    "Nitrogen Level": (40.0, 70.0),
    "Phosphorus Level": (30.0, 60.0),
    "Potassium Level": (40.0, 70.0),
}

# ================= SOLUTIONS =================
LIVE_SENSOR_SOLUTIONS_MODERATE = {
    "Temperature": "Use shade nets during peak hours",
    "Humidity": "Improve air circulation",
    "Soil Moisture": "Slightly increase irrigation",
}

LIVE_SENSOR_SOLUTIONS_SEVERE = {
    "Temperature": "Immediate cooling and shading required",
    "Humidity": "Forced ventilation required",
    "Soil Moisture": "Immediate irrigation correction",
}

# ================= HELPERS =================
def classify(value, low, high):
    if low <= value <= high:
        return "Healthy"
    elif (low * 0.8) <= value <= (high * 1.2):
        return "Moderate Stress"
    else:
        return "High Stress"

def overall_status(statuses):
    if statuses.count("High Stress") >= 2:
        return "High Stress"
    elif "High Stress" in statuses or statuses.count("Moderate Stress") >= 2:
        return "Moderate Stress"
    else:
        return "Healthy"

# ================= LOAD MODEL =================
@st.cache_resource
def load_leaf_model():
    return load_model("tomato_leaf_disease_1model.h5", compile=False)

leaf_model = load_leaf_model()

LEAF_CLASSES = [
    "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
    "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites",
    "Tomato_Target_Spot","Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus","Tomato_healthy"
]

# ================= LEAF SECTION =================
st.header("Leaf Disease Detection 🍃")

leaf = st.file_uploader("Upload tomato leaf image", ["jpg","jpeg","png"])

if leaf:
    img = Image.open(leaf).convert("RGB")
    st.image(img, width=250)

    arr = np.expand_dims(np.array(img.resize((224,224))) / 255.0, axis=0)

    pred = leaf_model.predict(arr, verbose=0)
    disease = LEAF_CLASSES[np.argmax(pred)]

    st.session_state.leaf_disease = disease
    st.session_state.leaf_status = "Healthy" if disease == "Tomato_healthy" else "Diseased"

    if disease == "Tomato_healthy":
        st.success("🌿 Healthy Leaf")
    else:
        st.error(f"🍂 {disease.replace('Tomato_','')}")

st.divider()

# ================= FIREBASE SENSOR DATA =================
st.header("Live Sensor Data 🌡️")

sensor_statuses = []
problem_sensors = {}

try:
    ref = db.reference("sensors")
    data = ref.get()

    if data:

        raw_soil = data.get("soil", 0)

        if raw_soil >= 4000:
            soil_percent = 0
        else:
            soil_percent = int((4095 - raw_soil) * 100 / 4095)

        soil_percent = max(0, min(100, soil_percent))

        st.session_state.sensor_data = {
            "Temperature": data.get("temperature", 0),
            "Humidity": data.get("humidity", 0),
            "Soil Moisture": soil_percent,
            "Light": data.get("light", 0),
        }

except:
    pass

# ================= DISPLAY SENSOR =================
if st.session_state.sensor_data:

    d = st.session_state.sensor_data

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("🌡 Temperature (°C)", d["Temperature"])
    c2.metric("💧 Humidity (%)", d["Humidity"])
    c3.metric("🌱 Soil Moisture (%)", d["Soil Moisture"])

    light_status = "ON" if d["Light"] == 0 else "OFF"
    c4.metric("💡 Light", light_status)

    st.subheader("📊 Sensor Health Status")

    for s,(lo,hi) in REALTIME_THRESHOLDS.items():

        stt = classify(d[s], lo, hi)
        sensor_statuses.append(stt)

        if stt != "Healthy":
            problem_sensors[s] = stt

        st.write(f"{s}: **{stt}**")

st.divider()

# ================= EXTRA SENSOR SLIDERS =================
st.header("Additional Sensor Inputs")

cols = st.columns(2)
i = 0

for s,(lo,hi) in EXTRA_SENSOR_THRESHOLDS.items():

    with cols[i%2]:

        v = st.slider(s, 0.0, float(hi*2), float((lo+hi)/2), 0.1)

        stt = classify(v, lo, hi)

        sensor_statuses.append(stt)

        if stt != "Healthy":
            problem_sensors[s] = stt

        if stt=="Healthy":
            st.success("Healthy")
        elif stt=="Moderate Stress":
            st.warning("Moderate Stress")
        else:
            st.error("High Stress")

    i+=1

overall = overall_status(sensor_statuses)

if overall=="Healthy":
    st.success("🌿 Overall Sensor Status: HEALTHY")
elif overall=="Moderate Stress":
    st.warning("⚠️ Overall Sensor Status: MODERATE STRESS")
else:
    st.error("🚨 Overall Sensor Status: HIGH STRESS")

st.divider()

# ================= FINAL DECISION =================
st.header("Final Plant Health Decision")

if st.button("Final Plant Health Decision ✅"):

    leaf = st.session_state.leaf_status
    sensor = overall

    if leaf == "Healthy" and sensor == "Healthy":
        st.success("🌿 Plant Healthy")
    else:
        st.warning("⚠️ Plant requires attention")
