import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("📈 Model Performance Dashboard")

# Load metrics
metrics_path = "metrics/model_scores.csv"
predictions_dir = "predictions"

if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    st.subheader("📊 Model Evaluation Metrics")
    st.dataframe(metrics_df)
else:
    st.warning("Model metrics not found. Please run train.py.")

# Model list
models = ["random_forest", "xgboost", "linear_regression"]

st.subheader("📉 Model Predictions vs Actual")

# Dropdown to select a model
selected_model = st.selectbox("Select Model to View Predictions", models)

# Load and visualize predictions
pred_path = f"{predictions_dir}/{selected_model}_predicted_prices.csv"

if os.path.exists(pred_path):
    df_pred = pd.read_csv(pred_path)
    df_pred["Prediction_Time"] = pd.to_datetime(df_pred["Prediction_Time"])

    st.line_chart(
        df_pred.set_index("Prediction_Time")[["Actual_Close", "Predicted_Close"]],
        use_container_width=True
    )

    # Optional: More advanced matplotlib-style plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_pred["Prediction_Time"], df_pred["Actual_Close"], label="Actual", color='black', linestyle='--')
    ax.plot(df_pred["Prediction_Time"], df_pred["Predicted_Close"], label="Predicted", linestyle='-')
    ax.set_ylim(190, max(df_pred["Actual_Close"].max(), df_pred["Predicted_Close"].max()) * 1.01)
    ax.set_title(f"{selected_model.replace('_', ' ').title()} Predictions vs Actual Close Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)


else:
    st.warning(f"Prediction file for {selected_model} not found.")
