import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from streamlit_autorefresh import st_autorefresh

import firebase_admin
from firebase_admin import credentials, db

import os
import json


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="IoT Based Crop Health Monitoring System",
    layout="centered"
)

# ================= MOBILE UI STYLE =================
st.markdown("""
<style>

.main-title{
font-size:42px;
font-weight:700;
text-align:center;
}

@media (max-width:768px){

.main-title{
font-size:30px;
}

h2{
font-size:22px;
}

h3{
font-size:20px;
}

.stMetric{
font-size:14px;
}

}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">IoT Based Crop Health Monitoring System 🌱</div>', unsafe_allow_html=True)
st.divider()


# ================= AUTO REFRESH =================
st_autorefresh(interval=5000, key="sensor_refresh")


# ================= FIREBASE INIT =================
@st.cache_resource
def init_firebase():

    firebase_dict = json.loads(os.environ["FIREBASE_KEY"])
    cred = credentials.Certificate(firebase_dict)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            cred,
            {"databaseURL": "https://crop-health-monitoring-57ff1-default-rtdb.firebaseio.com/"}
        )

    return db.reference("sensors")

sensor_ref = init_firebase()


# ================= SESSION STATE =================
for key in [
    "sensor_data",
    "leaf_status",
    "leaf_disease",
    "final_decision",
    "uploaded_leaf"
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ================= THRESHOLDS =================
REALTIME_THRESHOLDS = {
    "Temperature": (18, 32),
    "Humidity": (40, 70),
    "Soil Moisture": (30, 70),
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

SENSOR_SOLUTIONS_MODERATE = {
    "Soil pH": "Minor soil amendment",
    "Soil Temperature (°C)": "Apply light mulch",
    "Light Intensity": "Adjust shading",
    "Nitrogen Level": "Low dose Nitrogen fertilizer",
    "Phosphorus Level": "Low dose Phosphorus fertilizer",
    "Potassium Level": "Low dose Potassium fertilizer",
}

SENSOR_SOLUTIONS_SEVERE = {
    "Soil pH": "Apply lime or sulfur as required",
    "Soil Temperature (°C)": "Thick mulching + irrigation",
    "Light Intensity": "Reduce direct sunlight immediately",
    "Nitrogen Level": "Recommended Nitrogen dose",
    "Phosphorus Level": "Recommended Phosphorus dose",
    "Potassium Level": "Recommended Potassium dose",
}

SENSOR_SOLUTIONS_CRITICAL = {
    "Soil pH": "Soil testing before correction",
    "Soil Temperature (°C)": "Emergency cooling + heavy mulching",
    "Light Intensity": "Artificial shading immediately",
    "Nitrogen Level": "Expert-guided nitrogen correction",
    "Phosphorus Level": "Expert-guided phosphorus correction",
    "Potassium Level": "Expert-guided potassium correction",
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

LEAF_SOLUTIONS = {
"Tomato_Bacterial_spot": "Apply copper fungicide; avoid overhead irrigation",
"Tomato_Early_blight": "Apply Mancozeb; remove infected leaves",
"Tomato_Late_blight": "Immediate fungicide; destroy infected plants",
"Tomato_Leaf_Mold": "Improve ventilation; apply sulfur fungicide",
"Tomato_Septoria_leaf_spot": "Remove infected leaves; apply fungicide",
"Tomato_Spider_mites": "Apply neem oil or insecticidal soap",
"Tomato_Target_Spot": "Apply fungicide; reduce humidity",
"Tomato_Tomato_mosaic_virus": "Remove infected plants; disinfect tools",
"Tomato_Tomato_YellowLeaf_Curl_Virus": "Control whiteflies; remove infected plants",
}


# ================= LEAF =================
st.subheader("Leaf Disease Detection 🍃")

leaf = st.file_uploader("Upload tomato leaf image", ["jpg","jpeg","png"])

if leaf is not None:

    img = Image.open(leaf).convert("RGB")
    st.session_state.uploaded_leaf = img

    arr = np.expand_dims(np.array(img.resize((224,224))) / 255.0, axis=0)
    pred = leaf_model.predict(arr, verbose=0)

    disease = LEAF_CLASSES[np.argmax(pred)]

    st.session_state.leaf_disease = disease
    st.session_state.leaf_status = "Healthy" if disease=="Tomato_healthy" else "Diseased"


if st.session_state.uploaded_leaf is not None:

    st.image(st.session_state.uploaded_leaf,width=250)

    if st.session_state.leaf_status=="Healthy":
        st.success("🌿 Healthy Leaf")
    else:
        st.error(f"🍂 {st.session_state.leaf_disease.replace('Tomato_','')}")


st.divider()


# ================= LIVE SENSOR DATA =================
st.subheader("Live Sensor Data 🌡️")

sensor_statuses=[]
problem_sensors={}

try:

    data = sensor_ref.get()

    if data:

        raw_soil = data.get("soil",0)
        soil_percent = max(0,min(100,(4095-raw_soil)*100/4095))

        st.session_state.sensor_data={
            "Temperature":data.get("temperature",0),
            "Humidity":data.get("humidity",0),
            "Soil Moisture":round(soil_percent,1),
            "Light":data.get("light",0),
        }

except:
    pass


if st.session_state.sensor_data:

    d = st.session_state.sensor_data

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric("🌡 Temperature (°C)", d["Temperature"])
    col2.metric("💧 Humidity (%)", d["Humidity"])
    col3.metric("🌱 Soil Moisture (%)", d["Soil Moisture"])

    light_status = "ON" if d["Light"]==0 else "OFF"
    col4.metric("💡 Light", light_status)

    st.markdown("**Sensor Health Status**")

    for s,(lo,hi) in REALTIME_THRESHOLDS.items():

        stt = classify(d[s],lo,hi)

        sensor_statuses.append(stt)

        if stt!="Healthy":
            problem_sensors[s]=stt

        st.write(f"{s}: **{stt}**")


st.divider()


# ================= EXTRA SENSOR SLIDERS =================
st.subheader("Additional Sensor Inputs")

for s,(lo,hi) in EXTRA_SENSOR_THRESHOLDS.items():

    v = st.slider(
        s,
        0.0,
        float(hi*2),
        float((lo+hi)/2),
        0.1,
        key=s
    )

    stt = classify(v,lo,hi)

    sensor_statuses.append(stt)

    if stt!="Healthy":
        problem_sensors[s]=stt

    if stt=="Healthy":
        st.success("Healthy")
    elif stt=="Moderate Stress":
        st.warning("Moderate Stress")
    else:
        st.error("High Stress")


overall = overall_status(sensor_statuses)

if overall=="Healthy":
    st.success("🌿 Overall Sensor Status: HEALTHY")

elif overall=="Moderate Stress":
    st.warning("⚠️ Overall Sensor Status: MODERATE STRESS")

else:
    st.error("🚨 Overall Sensor Status: HIGH STRESS")


st.divider()


# ================= FINAL DECISION =================
st.subheader("Final Plant Health Decision")

if st.button("Final Plant Health Decision ✅"):

    leaf = st.session_state.leaf_status
    sensor = overall

    messages=[]

    if leaf=="Healthy" and sensor=="Healthy":
        st.session_state.final_decision=("success","🌿 Plant Healthy",["✅ No action required"])

    else:

        if leaf=="Diseased":
            messages.append(f"🦠 Leaf Disease → {LEAF_SOLUTIONS[st.session_state.leaf_disease]}")

        for s,stt in problem_sensors.items():

            if s in LIVE_SENSOR_SOLUTIONS_MODERATE:
                rec = LIVE_SENSOR_SOLUTIONS_SEVERE[s] if stt=="High Stress" else LIVE_SENSOR_SOLUTIONS_MODERATE[s]

            else:

                if leaf=="Diseased" and stt=="High Stress":
                    rec = SENSOR_SOLUTIONS_CRITICAL[s]
                elif stt=="High Stress":
                    rec = SENSOR_SOLUTIONS_SEVERE[s]
                else:
                    rec = SENSOR_SOLUTIONS_MODERATE[s]

            messages.append(f"{s} → {rec}")

        if leaf=="Healthy" and sensor=="Moderate Stress":
            level,title="warning","⚠️ Recoverable Environmental Stress"
        elif leaf=="Healthy" and sensor=="High Stress":
            level,title="error","🚨 Severe Environmental Stress"
        elif leaf=="Diseased" and sensor=="Healthy":
            level,title="error","🦠 Disease Detected"
        elif leaf=="Diseased" and sensor=="Moderate Stress":
            level,title="warning","🟠 HIGH RISK"
        else:
            level,title="error","🚨 CRITICAL CONDITION"

        st.session_state.final_decision=(level,title,messages)


# ================= OUTPUT =================
if st.session_state.final_decision:

    lvl,title,msgs = st.session_state.final_decision

    getattr(st,lvl)(title)

    for m in msgs:
        st.write(f"- {m}")

    if "CRITICAL" in title:
        st.write("🚜 Consult agricultural expert immediately.")
