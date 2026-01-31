import os
import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Student Dropout Early Warning System")

MODEL_PATH = "student_dropout_model.joblib"

model = None
def _ensure_sklearn_compat():
    try:
        import importlib
        cct = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(cct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            setattr(cct, "_RemainderColsList", _RemainderColsList)
    except Exception:
        # best-effort shim; if this fails we'll let joblib show the original error
        pass

if os.path.exists(MODEL_PATH):
    try:
        _ensure_sklearn_compat()
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}: {e}")

if model is None:
    st.warning(
        "Model file not found. Upload a trained model file here, or place 'student_dropout_model.joblib' in the project directory."
    )
    uploaded_model = st.file_uploader(
        "Upload trained model (joblib .joblib or .pkl)", type=["joblib", "pkl"]
    )
    if uploaded_model is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.getbuffer())
        try:
            _ensure_sklearn_compat()
            model = joblib.load(MODEL_PATH)
            st.success("Model uploaded and loaded successfully.")
        except Exception as e:
            st.error(f"Uploaded model could not be loaded: {e}")

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload Student CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Drop any previously computed columns that shouldn't be inputs
    for col in ("risk_score", "risk_label"):
        if col in data.columns:
            data = data.drop(columns=[col])

    # Select numeric columns only — the model expects numeric input
    numeric = data.select_dtypes(include="number")
    if hasattr(model, "n_features_in_"):
        n_in = int(model.n_features_in_)
    else:
        n_in = numeric.shape[1]

    if numeric.shape[1] < n_in:
        st.error(
            f"Uploaded CSV has {numeric.shape[1]} numeric columns but the model expects {n_in}.\n"
            "Please provide a CSV with the required numeric feature columns."
        )
    else:
        # If more numeric columns than expected, take the first n_in
        X = numeric.iloc[:, :n_in].to_numpy()
        probs = model.predict_proba(X)[:, 1]

        data["risk_score"] = probs
        data["risk_label"] = data["risk_score"].apply(
            lambda x: "High" if x >= 0.7 else "Medium" if x >= 0.4 else "Low"
        )

    # Only show results if predictions were computed
    if "risk_score" in data.columns:
        st.subheader("🚨 Top 20 High-Risk Students")
        st.dataframe(data.sort_values("risk_score", ascending=False).head(20))

        student = st.selectbox("Select Student", data.index)
        st.write("Risk Score:", data.loc[student, "risk_score"])
        st.write("Risk Level:", data.loc[student, "risk_label"])
    else:
        st.info("No predictions were made. Ensure your CSV contains the numeric feature columns expected by the model.")
