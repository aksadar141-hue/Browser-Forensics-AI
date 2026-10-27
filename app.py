import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from urllib.parse import urlparse
from scipy.sparse import hstack

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Browser Forensics AI",
    page_icon="🔎",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "phishing_url_model_v3.joblib"
VECTORIZER_PATH = ROOT / "models" / "phishing_url_vectorizer_v3.joblib"


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_url_features(url):
    url = str(url).strip()
    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    return [
        len(url),
        len(domain),
        len(path),
        len(query),
        domain.count("."),
        domain.count("-"),
        domain.count("_"),
        url.count("/"),
        url.count("?"),
        url.count("="),
        url.count("&"),
        url.count("%"),
        sum(c.isdigit() for c in url),
        sum(c.isalpha() for c in url),
        int(url.startswith("https://")),
        int(url.startswith("http://")),
        int("@" in url),
        int("://" in url),
        len(set(domain)),
    ]


# -----------------------------
# APP TITLE
# -----------------------------
st.title("🔎 Browser Forensics AI")
st.write(
    "Upload browser history in CSV format to analyse URLs "
    "using the V3 machine-learning model."
)

st.info(
    "Privacy: Upload only a browser-history CSV you are permitted to analyse. "
    "This cloud app cannot directly access your laptop's browser history."
)

st.warning(
    "Model label caution: this app assumes class 1 = PHISHING and other "
    "classes = BENIGN, matching the existing script. Confirm this mapping "
    "against the model's training labels before relying on the results."
)


# -----------------------------
# CSV UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload browser history CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]

        st.subheader("Uploaded data")
        st.write(f"Rows found: {len(df)}")
        st.dataframe(df.head(10), use_container_width=True)

        # Find URL column regardless of capitalization
        url_column = next(
            (
                col for col in df.columns
                if col.strip().lower() in ["url", "link"]
            ),
            None
        )

        if url_column is None:
            st.error(
                "URL column not found. Your CSV must contain a column "
                "named URL, url, Link, or link."
            )
            st.stop()

        if not MODEL_PATH.exists():
            st.error(
                "Model file is missing: models/phishing_url_model_v3.joblib"
            )
            st.stop()

        if not VECTORIZER_PATH.exists():
            st.error(
                "Vectorizer file is missing: "
                "models/phishing_url_vectorizer_v3.joblib"
            )
            st.stop()

        if st.button("Analyse Browser History", type="primary"):
            with st.spinner("Loading model and analysing URLs..."):
                model = joblib.load(MODEL_PATH)
                vectorizer = joblib.load(VECTORIZER_PATH)

                work = df.copy()
                work[url_column] = work[url_column].fillna("").astype(str).str.strip()

                # Keep valid HTTP and HTTPS URLs only
                valid_mask = work[url_column].str.startswith(
                    ("http://", "https://"), na=False
                )
                valid = work.loc[valid_mask].copy()

                if valid.empty:
                    st.error(
                        "No valid HTTP/HTTPS URLs were found in this CSV."
                    )
                    st.stop()

                # Analyse each unique URL only once
                unique_urls = valid[url_column].drop_duplicates().tolist()

                text_features = vectorizer.transform(unique_urls)
                numeric_features = np.asarray(
                    [extract_url_features(url) for url in unique_urls]
                )

                combined_features = hstack(
                    [text_features, numeric_features]
                ).tocsr()

                predictions = model.predict(combined_features)
                probabilities = model.predict_proba(combined_features)

                # Build results for unique URLs
                prediction_map = {}

                for url, prediction, probs in zip(
                    unique_urls, predictions, probabilities
                ):
                    class_labels = list(model.classes_)
                    predicted_index = class_labels.index(prediction)
                    confidence = float(probs[predicted_index]) * 100

                    if prediction == 1:
                        status = "PHISHING"
                    else:
                        status = "BENIGN"

                    prediction_map[url] = {
                        "Status": status,
                        "Predicted Class": str(prediction),
                        "Confidence (%)": round(confidence, 2),
                    }

                # Add predictions back to every matching history row
                results = valid.copy()
                results["Status"] = results[url_column].map(
                    lambda url: prediction_map[url]["Status"]
                )
                results["Predicted Class"] = results[url_column].map(
                    lambda url: prediction_map[url]["Predicted Class"]
                )
                results["Confidence (%)"] = results[url_column].map(
                    lambda url: prediction_map[url]["Confidence (%)"]
                )

                # Rename URL column for consistent display
                if url_column != "URL":
                    results = results.rename(columns={url_column: "URL"})

                st.session_state["browser_results"] = results

        # Show results after analysis
        if "browser_results" in st.session_state:
            results = st.session_state["browser_results"]

            st.subheader("Analysis Results")

            total = len(results)
            phishing_count = int((results["Status"] == "PHISHING").sum())
            benign_count = int((results["Status"] == "BENIGN").sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Total URLs analysed", total)
            col2.metric("Flagged as phishing", phishing_count)
            col3.metric("Classified as benign", benign_count)

            st.dataframe(results, use_container_width=True)

            csv_data = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Analysis Report (CSV)",
                data=csv_data,
                file_name="browser_forensics_report.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.caption(
            "If the error mentions scikit-learn or incompatible model "
            "versions, the installed library version may not match the "
            "version used to save the model."
        )

st.caption("Browser Forensics AI | V3 URL analysis")