"""
Streamlit app for Crop Recommendation System.

- Inference-only app (no dataset dependency)
- Uses trained model + feature_columns.json
- Cloud-safe & production-ready
"""

# -------------------------------------------------
# Make project root importable (IMPORTANT)
# -------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------
# Imports
# -------------------------------------------------

from typing import Dict, Any
import pandas as pd
import streamlit as st

from src.deployment import predict_single, predict_batch


# -------------------------------------------------
# Fixed feature schema (MODEL INPUTS)
# -------------------------------------------------

FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


# -------------------------------------------------
# Input form
# -------------------------------------------------

def build_input_form() -> Dict[str, Any]:
    st.subheader("🌾 Enter Soil & Environmental Parameters")

    input_data = {
        "N": st.number_input("Nitrogen (N)", min_value=0.0, value=50.0),
        "P": st.number_input("Phosphorus (P)", min_value=0.0, value=40.0),
        "K": st.number_input("Potassium (K)", min_value=0.0, value=40.0),
        "temperature": st.number_input("Temperature (°C)", value=25.0),
        "humidity": st.number_input("Humidity (%)", value=70.0),
        "ph": st.number_input("pH", value=6.5),
        "rainfall": st.number_input("Rainfall (mm)", value=100.0),
    }

    return input_data


# -------------------------------------------------
# Main app
# -------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Crop Recommendation System",
        page_icon="🌱",
        layout="centered",
    )

    st.title("🌱 Crop Recommendation System")

    st.markdown(
        """
        This application recommends the **most suitable crop** based on  
        **soil nutrients and environmental conditions**.

        **Outputs:**
        - Best crop
        - Confidence score
        - Top-3 alternative crops
        """
    )

    tab1, tab2 = st.tabs(["🌾 Single Recommendation", "📂 Batch Prediction"])

    # -------------------------------
    # Single Prediction
    # -------------------------------

    with tab1:
        input_data = build_input_form()

        st.write("### 🔍 Input Preview")
        st.json(input_data)

        if st.button("🌿 Recommend Crop"):
            try:
                result = predict_single(input_data, top_k=3)

                st.success(f"🌾 **Best Crop:** {result['best_crop']}")
                st.metric(
                    label="Confidence",
                    value=f"{result['confidence'] * 100:.2f}%",
                )

                st.subheader("🥈 Top-3 Recommended Crops")
                df_reco = pd.DataFrame(result["top_recommendations"])
                df_reco["probability"] *= 100

                st.dataframe(
                    df_reco.rename(
                        columns={
                            "crop": "Crop",
                            "probability": "Probability (%)",
                        }
                    ),
                    use_container_width=True,
                )

            except Exception as exc:
                st.error("Prediction failed.")
                st.write(str(exc))

    # -------------------------------
    # Batch Prediction
    # -------------------------------

    with tab2:
        st.subheader(
            "Upload a CSV for batch crop recommendations "
            "(must contain: N, P, K, temperature, humidity, ph, rainfall)"
        )

        uploaded = st.file_uploader("Choose a CSV", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())

            if st.button("Run Batch Recommendation"):
                try:
                    result_df = predict_batch(df, top_k=3)
                    st.dataframe(result_df.head(10))

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions",
                        data=csv,
                        file_name="crop_recommendations.csv",
                        mime="text/csv",
                    )

                except Exception as exc:
                    st.error("Batch prediction failed.")
                    st.write(str(exc))


# -------------------------------------------------
# Entry point
# -------------------------------------------------

if __name__ == "__main__":
    main()
