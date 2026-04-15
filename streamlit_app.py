import streamlit as st
import numpy as np
import pickle
import os

# ================================
# LOAD MODEL
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "parkinsons_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Parkinson's AI Detection",
    page_icon="🧠",
    layout="wide"
)

# ================================
# CUSTOM CSS (🔥 UI UPGRADE)
# ================================
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #00C9A7;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
    }
    .stNumberInput>div>div>input {
        background-color: #1c1f26;
        color: white;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #1c1f26;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown("""
<h1 style='text-align: center; color: #00C9A7;'>
🧠 Parkinson's Disease Prediction
</h1>
<p style='text-align: center;'>
AI-powered early detection system
</p>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Prediction", "About"])

# ================================
# PREDICTION PAGE
# ================================
if page == "Prediction":

    st.markdown("### 🔬 Enter Patient Voice Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Voice Frequency")
        fo = st.number_input("Fo (Hz)")
        fhi = st.number_input("Fhi (Hz)")
        flo = st.number_input("Flo (Hz)")

        st.markdown("#### Jitter")
        jitter_percent = st.number_input("Jitter (%)")
        jitter_abs = st.number_input("Jitter (Abs)")
        rap = st.number_input("RAP")
        ppq = st.number_input("PPQ")

    with col2:
        st.markdown("#### Shimmer")
        ddp = st.number_input("DDP")
        shimmer = st.number_input("Shimmer")
        shimmer_db = st.number_input("Shimmer (dB)")
        apq3 = st.number_input("APQ3")
        apq5 = st.number_input("APQ5")
        apq = st.number_input("APQ")
        dda = st.number_input("DDA")

    with col3:
        st.markdown("#### Other Features")
        nhr = st.number_input("NHR")
        hnr = st.number_input("HNR")
        rpde = st.number_input("RPDE")
        dfa = st.number_input("DFA")
        spread1 = st.number_input("Spread1")
        spread2 = st.number_input("Spread2")
        d2 = st.number_input("D2")
        ppe = st.number_input("PPE")

    st.markdown("---")

    if st.button("🔍 Predict Now"):

        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                                shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                                rpde, dfa, spread1, spread2, d2, ppe]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        confidence = np.max(probability) * 100

        st.markdown("## 📊 Prediction Result")

        if prediction[0] == 1:
            st.markdown(f"""
            <div class="card">
                <h2 style='color:red;'>⚠️ Parkinson's Detected</h2>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card">
                <h2 style='color:green;'>✅ Healthy</h2>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

# ================================
# ABOUT PAGE
# ================================
elif page == "About":

    st.markdown("""
    ## 📘 About This Project

    This application uses Machine Learning models to predict Parkinson's Disease 
    based on voice measurements.

    ### 🚀 Features
    - High accuracy ML models
    - Clean and modern UI
    - Real-time prediction

    ### 🧠 Models Used
    - Support Vector Machine
    - Random Forest
    - XGBoost

    ### 👨‍💻 Developed By
    (Manasa Gunnam)
    """)
