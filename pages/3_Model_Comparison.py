import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Modellvergleich",
    page_icon="📊"
)

st.title("📊 Modellvergleich")

results = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "MAE": [
            13452.22,
            13459.48,
            13452.24
        ],
        "RMSE": [
            16423.23,
            16416.15,
            16423.20
        ],
        "R²": [
            0.6003,
            0.6007,
            0.6003
        ]
    }
)

st.subheader("Regression Models")

st.dataframe(
    results,
    use_container_width=True
)

best_model = results.loc[
    results["R²"].idxmax()
]

st.subheader("🏆 Bestes Model")

st.success(
    f"""
    Model: {best_model['Model']}
    
    R² = {best_model['R²']:.4f}
    
    MAE = {best_model['MAE']:.2f}
    """
)

