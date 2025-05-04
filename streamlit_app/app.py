import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")
st.title("🩺 Multiple Disease Prediction System")

disease = st.sidebar.selectbox("Select Disease", ["Liver", "Kidney", "Parkinsons"])
model_type = st.sidebar.selectbox("Choose Model", ["LogisticRegression", "RandomForest"])

def predict(model, input_data):
    prediction = model.predict([input_data])[0]
    proba = model.predict_proba([input_data])[0]
    return prediction, proba


# ---------------- LIVER ----------------
if disease == "Liver":
    st.subheader("Liver Disease Prediction")

    age = st.number_input("Age", 0, 100)
    gender = st.radio("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin")
    db = st.number_input("Direct Bilirubin")
    ap = st.number_input("Alkaline Phosphotase")
    alt = st.number_input("Alamine Aminotransferase")
    ast = st.number_input("Aspartate Aminotransferase")
    tp = st.number_input("Total Proteins")
    alb = st.number_input("Albumin")
    agr = st.number_input("Albumin and Globulin Ratio")

    if st.button("Predict"):
        input_data = [
            age, 1 if gender == "Male" else 0, tb, db, ap,
            alt, ast, tp, alb, agr
        ]
        model = joblib.load(f"liver_{model_type}.pkl")
        result, prob = predict(model, input_data)
        st.success(f"Disease: {'Yes' if result==1 else 'No'} | Confidence: {max(prob)*100:.2f}%")


# ---------------- KIDNEY ----------------
elif disease == "Kidney":
    st.subheader("Kidney Disease Prediction")

    age = st.number_input("Age", 0, 100)
    bp = st.number_input("Blood Pressure (bp)")
    sg = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.selectbox("Albumin (al)", [0, 1, 2, 3, 4, 5])
    su = st.selectbox("Sugar (su)", [0, 1, 2, 3, 4, 5])
    pcv = st.number_input("Packed Cell Volume (pcv)", 0)
    wc = st.number_input("White Blood Cell Count (wc)", 0)
    rc = st.number_input("Red Blood Cell Count (rc)", 0.0)

    htn = st.selectbox("Hypertension", ["no", "yes"])
    dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
    appet = st.selectbox("Appetite", ["poor", "good"])
    ane = st.selectbox("Anemia", ["no", "yes"])

    if st.button("Predict"):
        input_data = [
            age, bp, sg, al, su,
            pcv, wc, rc,
            1 if htn == "yes" else 0,
            1 if dm == "yes" else 0,
            1 if appet == "good" else 0,
            1 if ane == "yes" else 0,
        ]
        model = joblib.load(f"kidney_{model_type}.pkl")
        result, prob = predict(model, input_data)
        st.success(f"Disease: {'Yes' if result==1 else 'No'} | Confidence: {max(prob)*100:.2f}%")


# ---------------- PARKINSONS ----------------
elif disease == "Parkinsons":
    st.subheader("Parkinson's Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_percent = st.number_input("MDVP:Jitter(%)")
    jitter_abs = st.number_input("MDVP:Jitter(Abs)")
    rap = st.number_input("MDVP:RAP")
    ppq = st.number_input("MDVP:PPQ")
    ddp = st.number_input("Jitter:DDP")
    shimmer = st.number_input("MDVP:Shimmer")
    shimmer_db = st.number_input("MDVP:Shimmer(dB)")
    apq3 = st.number_input("Shimmer:APQ3")
    apq5 = st.number_input("Shimmer:APQ5")
    apq = st.number_input("MDVP:APQ")
    dda = st.number_input("Shimmer:DDA")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("spread1")
    spread2 = st.number_input("spread2")
    d2 = st.number_input("D2")
    ppe = st.number_input("PPE")

    if st.button("Predict"):
        input_data = [
            fo, fhi, flo, jitter_percent, jitter_abs,
            rap, ppq, ddp, shimmer, shimmer_db,
            apq3, apq5, apq, dda, nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ]
        model = joblib.load(f"parkinsons_{model_type}.pkl")
        result, prob = predict(model, input_data)
        st.success(f"Disease: {'Yes' if result==1 else 'No'} | Confidence: {max(prob)*100:.2f}%")
