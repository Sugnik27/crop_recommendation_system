"""
Streamlit app for Crop Recommendation System.

Note:
- To avoid import issues when Streamlit runs the script,
  the project root is added to sys.path at runtime.
"""


# Make project root importable (IMPORTANT for Streamlit)


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Imports


from typing import Any, Dict
import pandas as pd
import streamlit as st

from src.config import CLEANED_DATA_PATH
from src.deployment import predict_single, predict_batch, get_feature_columns
from src.preprocessing import load_data


# Cache data loading (for UI defaults)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    df = load_data(CLEANED_DATA_PATH)
    return df



# Build input form dynamically


def build_input_form(df_sample: pd.DataFrame) -> Dict[str, Any]:
    st.subheader("🌾 Enter Soil & Environmental Parameters")

    model = load_data()
    feature_cols = get_feature_columns(model)

    input_data: Dict[str, Any] = {}

    for col in feature_cols:
        series = df_sample[col]

        default = float(series.median()) if not series.isna().all() else 0.0

        input_data[col] = st.number_input(
            label=col,
            value=default
        )

    return input_data



# Main app


def main() -> None:
    st.set_page_config(
        page_title="Crop Recommendation System",
        page_icon="🌱",
        layout="centered"
    )

    st.title("🌱 Crop Recommendation System")

    st.markdown(
        """
        This application recommends the **most suitable crop** based on  
        **soil nutrients and environmental conditions**.

        The model returns:
        - Best crop
        - Confidence score
        - Top-3 alternative crops
        """
    )

    # Load data
    df_sample = load_sample_data()

    st.sidebar.markdown("### 📊 Dataset Snapshot")
    st.sidebar.dataframe(df_sample.head(5))

    tab1, tab2 = st.tabs(["🌾 Single Recommendation", "📂 Batch Prediction"])

    
    # Single prediction tab
    

    with tab1:
        input_data = build_input_form(df_sample)

        st.write("### Input Preview")
        st.json(input_data)

        if st.button("🌿 Recommend Crop"):
            try:
                result = predict_single(input_data, top_k=3)

                st.success(f"🌾 **Best Crop:** {result['best_crop']}")
                st.metric(
                    label="Confidence",
                    value=f"{result['confidence']*100:.2f}%"
                )

                st.subheader("🥈 Top-3 Recommended Crops")
                df_reco = pd.DataFrame(result["top_recommendations"])
                df_reco["probability"] = df_reco["probability"] * 100

                st.dataframe(
                    df_reco.rename(
                        columns={
                            "crop": "Crop",
                            "probability": "Probability (%)"
                        }
                    ),
                    use_container_width=True
                )

            except Exception as exc:
                st.error("Prediction failed — check inputs or model.")
                st.write(f"Error: {str(exc)}")

    
    # Batch prediction tab
    

    with tab2:
        st.subheader(
            "Upload a CSV for batch crop recommendations "
            "(must contain the same feature columns)"
        )

        uploaded = st.file_uploader("Choose a CSV", type=["csv"])

        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(uploaded_df.head())

            if st.button("Run Batch Recommendation"):
                try:
                    result_df = predict_batch(uploaded_df, top_k=3)
                    st.write("Prediction results (first 10 rows):")
                    st.dataframe(result_df.head(10))

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions",
                        data=csv,
                        file_name="crop_recommendations.csv",
                        mime="text/csv"
                    )

                except Exception as exc:
                    st.error("Batch prediction failed — check input file.")
                    st.write(f"Error: {str(exc)}")



# Entry point


if __name__ == "__main__":
    main()
