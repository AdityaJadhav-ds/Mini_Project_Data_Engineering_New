
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import base64
import traceback
from pathlib import Path
from datetime import datetime

# ---------------------------
# Configuration & constants
# ---------------------------
APP_TITLE = "🏦 Loan Approval Prediction"
MODEL_PATHS = ["loan_approval_model.pkl", "loan_approval_model.joblib", "model/loan_approval_model.pkl"]
GITHUB_URL = "https://github.com/AdityaJadhav-ds"
LINKEDIN_URL = "https://www.linkedin.com/in/aditya-jadhav-6775702b4"
ALLOWED_TERM_OPTIONS = [12, 36, 60, 120, 180, 240, 300, 360, 480]

# ---------------------------
# Page setup & CSS
# ---------------------------
st.set_page_config(page_title="Loan Approval - Aditya Jadhav", page_icon="🏦", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#fbfdff,#eef4f8); color:#0b3556; font-family: Inter, system-ui, -apple-system; }
    h1 { text-align:center; color:#0b3556; font-weight:800; }
    .card { background:#fff; padding:18px; border-radius:12px; box-shadow:0 10px 30px rgba(2,6,23,0.06); }
    .control-label { font-weight:600; color:#0b3556; }
    [data-testid="stSidebar"] { background: #0b1220; color:#e6eef6; }
    [data-testid="stSidebar"] a { color:#7dd3fc !important; }
    .result-card { background: linear-gradient(90deg,#ecfeff,#f0f9ff); padding:18px; border-radius:12px; text-align:center; box-shadow:0 10px 30px rgba(2,6,23,0.04); }
    .success-card { border-left: 6px solid #10b981; }
    .fail-card { border-left: 6px solid #ef4444; }
    .muted { color:#6b7280; }
    .small { font-size:13px; color:#475569; }
    .dev-footer { text-align:center; margin-top:18px; color:#475569; font-size:13px; }
    .download-button-wrap { margin-top:10px; }
    @media (max-width: 900px) {
        h1 { font-size: 1.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utilities
# ---------------------------
def try_load_model(paths=MODEL_PATHS):
    """
    Try loading model from candidate paths using joblib or pickle.
    Supports:
      - plain model (estimator)
      - tuple e.g. (model, scaler, encoders) or (pipeline, sklearn_version)
      - dict {'model':..., 'preprocessor':..., 'encoders':..., 'metadata':...}
    Returns: dict with keys: model, preprocessor, scaler, encoders, metadata
    """
    last_err = None
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            continue
        try:
            # try joblib first (handles sklearn objects well)
            try:
                obj = joblib.load(pth)
            except Exception:
                with open(pth, "rb") as f:
                    obj = pickle.load(f)

            result = {"model": None, "preprocessor": None, "scaler": None, "encoders": None, "metadata": {}}
            if isinstance(obj, dict):
                # common saved dict structure
                result["model"] = obj.get("model") or obj.get("estimator")
                result["preprocessor"] = obj.get("preprocessor")
                result["scaler"] = obj.get("scaler")
                result["encoders"] = obj.get("encoders") or obj.get("label_encoders")
                result["metadata"] = obj.get("metadata", {})
            elif isinstance(obj, tuple):
                # some people save (model, scaler, encoders) or (model, sklearn_version)
                if len(obj) == 3:
                    result["model"], result["scaler"], result["encoders"] = obj
                elif len(obj) == 2:
                    result["model"], result["metadata"]["sklearn_version"] = obj
                else:
                    result["model"] = obj[0]
            else:
                # assume plain model/pipeline
                result["model"] = obj
            return result
        except Exception as e:
            last_err = e
            continue

    # if none found, raise a helpful message
    raise FileNotFoundError(f"No model found in paths {paths}. Last error: {last_err}")


@st.cache_data(show_spinner=False)
def read_csv_safe(file) -> pd.DataFrame:
    return pd.read_csv(file)


def df_to_download_bytes(df: pd.DataFrame):
    csv = df.to_csv(index=False).encode()
    return csv


def safe_encode_inputs(df: pd.DataFrame, encoders):
    """
    If encoders provided, attempt to transform columns present.
    Encoders expected as dict: {col: encoder_object}
    """
    if not encoders:
        return df
    df_copy = df.copy()
    for col, enc in encoders.items():
        if col in df_copy.columns:
            try:
                # ensure string for categories
                df_copy[col] = enc.transform(df_copy[col].astype(str))
            except Exception:
                # maybe encoder expects numpy array shape
                try:
                    df_copy[col] = enc.transform(df_copy[col])
                except Exception:
                    # leave as-is if transform fails
                    pass
    return df_copy

# ---------------------------
# Load model
# ---------------------------
with st.spinner("Loading model..."):
    try:
        bundle = try_load_model()
        model = bundle.get("model")
        preprocessor = bundle.get("preprocessor")
        scaler = bundle.get("scaler")
        encoders = bundle.get("encoders")
        metadata = bundle.get("metadata", {})
        sklearn_version = metadata.get("sklearn_version", "unknown")
    except FileNotFoundError as e:
        st.error("⚠️ Model not found. Upload `loan_approval_model.pkl` (or joblib) to the app folder.")
        st.stop()
    except Exception as e:
        st.error("❌ Failed to load model - see details.")
        st.exception(e)
        st.stop()

# ---------------------------
# Header
# ---------------------------
st.title(APP_TITLE)
st.markdown(f"<div class='small muted' style='text-align:center'>Model: <b>{getattr(model,'__class__', type(model)).__name__}</b> • scikit-learn: {sklearn_version}</div>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.markdown("<h3 style='color:#e6eef6'>Controls</h3>", unsafe_allow_html=True)
    mode = st.radio("Mode", ("Single Input", "Batch CSV"), index=0)
    st.markdown("---")
    st.markdown("<div style='color:#e6eef6'><b>Developer</b></div>", unsafe_allow_html=True)
    st.markdown(f"<a href='{GITHUB_URL}' target='_blank'>GitHub</a> • <a href='{LINKEDIN_URL}' target='_blank'>LinkedIn</a>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Tip: For batch predictions upload a CSV with the exact column names used in training (see sample).")
    if st.button("Download sample CSV"):
        # create sample row and provide a download
        sample = pd.DataFrame([{
            "Gender":"Male","Married":"Yes","Dependents":"0","Education":"Graduate","Self_Employed":"No",
            "ApplicantIncome":5000,"CoapplicantIncome":0,"LoanAmount":120,"Loan_Amount_Term":360,"Credit_History":1,"Property_Area":"Urban"
        }])
        st.download_button("Download sample.csv", data=df_to_download_bytes(sample), file_name="loan_sample_input.csv", mime="text/csv")

# ---------------------------
# Input form area
# ---------------------------
st.markdown("## Applicant details")
st.markdown("Fill applicant information below. Hover over labels for tooltips.")

def single_input_form():
    col1, col2 = st.columns([1,1])
    with col1:
        gender = st.selectbox("Gender", ["Male","Female"], help="Applicant gender")
        married = st.selectbox("Married", ["Yes","No"], help="Marital status")
        dependents = st.selectbox("Dependents", ["0","1","2","3+"], help="Number of dependents")
        education = st.selectbox("Education", ["Graduate","Not Graduate"], help="Education level")
        self_employed = st.selectbox("Self Employed", ["No","Yes"], help="Is the applicant self-employed?")
    with col2:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500, help="Monthly applicant income")
        coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=500, help="Monthly coapplicant income")
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=120, step=1, help="Requested loan amount (in thousands or same units as trained model)")
        loan_amount_term = st.selectbox("Loan Amount Term (months)", ALLOWED_TERM_OPTIONS, index=7, help="Repayment term in months")
        credit_history = st.selectbox("Credit History (1 = good)", [1,0], index=0, help="1 indicates good credit history")
        property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"], index=0, help="Property area type")
    row = {
        "Gender": gender, "Married": married, "Dependents": dependents, "Education": education,
        "Self_Employed": self_employed, "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_amount_term, "Credit_History": credit_history,
        "Property_Area": property_area
    }
    return pd.DataFrame([row])

# Choose input mode
if st.sidebar.radio("Choose input", ["Single", "Batch"]) == "Batch":
    mode = "Batch CSV"
else:
    mode = "Single Input"

uploaded = None
if mode == "Batch CSV":
    uploaded = st.file_uploader("Upload CSV file for batch predictions", type=["csv"])
    if uploaded is not None:
        try:
            input_df = read_csv_safe(uploaded)
            st.success(f"Loaded {len(input_df)} rows")
            if st.checkbox("Show uploaded data"):
                st.dataframe(input_df)
        except Exception as e:
            st.error("Failed to read uploaded CSV.")
            st.exception(e)
            st.stop()
else:
    input_df = single_input_form()
    if st.checkbox("Show input data"):
        st.dataframe(input_df)

st.write("---")

# ---------------------------
# Prepare for prediction
# ---------------------------
def prepare_for_model(df: pd.DataFrame):
    X = df.copy()
    # If encoders exist, attempt to transform (non-destructive)
    X = safe_encode_inputs(X, encoders)
    # If scaler exists, attempt to scale numeric columns (non-destructive)
    if scaler is not None:
        try:
            # attempt to scale only numeric cols used during training
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_num = pd.DataFrame(scaler.transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
                X.update(X_num)
        except Exception:
            # fallback: scaler may be part of pipeline only, so ignore
            pass
    # If preprocessor pipeline exists, let it transform
    if preprocessor is not None:
        try:
            X_trans = preprocessor.transform(X)
            return X_trans, True  # transformed flag
        except Exception:
            # if transform fails, continue with X as-is
            pass
    return X, False

# ---------------------------
# Run prediction
# ---------------------------
predict_btn = st.button("🔮 Predict Loan Approval", type="primary")

if predict_btn:
    with st.spinner("Predicting..."):
        try:
            X_ready, transformed = prepare_for_model(input_df)
            # get probabilities if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_ready)
                preds = model.predict(X_ready)
                # get positive class index
                if probs.shape[1] == 2:
                    positive_probs = probs[:, 1]
                else:
                    positive_probs = probs.max(axis=1)
            else:
                preds = model.predict(X_ready)
                positive_probs = None

            # results frame
            results = input_df.copy().reset_index(drop=True)
            results["Loan_Approved"] = np.where(np.asarray(preds).astype(int) == 1, "Approved", "Rejected")
            if positive_probs is not None:
                results["Approval_Confidence"] = (positive_probs * 100).round(2)

            # display a nice result card for single input
            if len(results) == 1:
                status = results.loc[0, "Loan_Approved"]
                confidence = results.loc[0, "Approval_Confidence"] if "Approval_Confidence" in results.columns else None
                if status == "Approved":
                    st.markdown(f"<div class='result-card success-card'><h2>✅ Loan Approved</h2><p class='muted'>Confidence: <b>{confidence}%</b></p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-card fail-card'><h2>❌ Loan Rejected</h2><p class='muted'>Confidence: <b>{confidence if confidence is not None else '—'}%</b></p></div>", unsafe_allow_html=True)
            else:
                # batch summary
                approved_count = (results["Loan_Approved"] == "Approved").sum()
                total = len(results)
                st.markdown(f"<div class='result-card'><h2>Batch prediction completed</h2><p class='muted'>Approved: <b>{approved_count}</b> / {total}</p></div>", unsafe_allow_html=True)

            # show table & actions
            st.markdown("### Detailed results")
            st.dataframe(results)

            # allow download
            csv_bytes = df_to_download_bytes(results)
            st.download_button("⬇️ Download predictions (CSV)", data=csv_bytes, file_name=f"loan_predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

            # small explainability: feature importances or coefficients
            try:
                fi = None
                if hasattr(model, "feature_importances_"):
                    fi = np.asarray(model.feature_importances_)
                elif hasattr(model, "coef_"):
                    fi = np.abs(np.asarray(model.coef_)).ravel()

                if fi is not None:
                    # try to discover feature names
                    try:
                        if transformed:
                            # preprocessor may provide feature names
                            feature_names = preprocessor.get_feature_names_out()
                        else:
                            feature_names = input_df.columns
                    except Exception:
                        feature_names = input_df.columns

                    # align length
                    if len(feature_names) == len(fi):
                        fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
                        fi_df = fi_df.sort_values("importance", ascending=False).head(10).reset_index(drop=True)
                        st.markdown("---")
                        st.markdown("#### Top feature importances (approx)")
                        st.table(fi_df.style.hide_index())
            except Exception:
                # ignore explainability errors
                pass

        except Exception as e:
            st.error("Prediction failed. See details below.")
            st.exception(traceback.format_exc())

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.markdown(
    f"""
<div class='dev-footer'>
  Built with ❤️ by <b>Aditya Jadhav</b> ·
  <a href='{GITHUB_URL}' target='_blank'>GitHub</a> ·
  <a href='{LINKEDIN_URL}' target='_blank'>LinkedIn</a><br>
  <span class='small'>v1.2 • © {datetime.utcnow().year} Loan Approval ML</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# End
# ---------------------------
