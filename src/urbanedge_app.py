import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import json
import tempfile
import io
from datetime import datetime
import plotly.express as px
from prophet import Prophet
import plotly.graph_objs as go
from fpdf import FPDF  # for PDF export
from data.tenant_datasets import tenant_datasets

# ------------------------------
# Mock Login (for tenants)
# ------------------------------
st.sidebar.title("UrbanEdge POC")
tenant = st.sidebar.selectbox("Select Tenant", list(tenant_datasets.keys()))
st.sidebar.markdown(f"**Logged in as:** {tenant}")

# ------------------------------
# File Upload (JSON data)
# ------------------------------
st.title("📊 Open Data Analytics POC")
uploaded_file = st.file_uploader("Upload your dataset (JSON)", type="json")

if uploaded_file is not None:
    data = json.load(uploaded_file)
    df = pd.DataFrame(data)
else:
    st.info(f"ℹ️ No dataset uploaded. Using default data for **{tenant}**.")
    df = pd.DataFrame(tenant_datasets[tenant])

# ------------------------------
# Data Preprocessing
# ------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])
metrics = df["metric"].unique().tolist()
selected_metric = st.sidebar.selectbox("Choose a metric", metrics)

metric_df = df[df["metric"] == selected_metric]

# ------------------------------
# KPIs
# ------------------------------
st.subheader(f"Key Performance Indicators – {selected_metric}")
total = metric_df["value"].sum()
avg = metric_df["value"].mean()
latest = metric_df.sort_values("timestamp").iloc[-1]["value"]

col1, col2, col3 = st.columns(3)
col1.metric("Total", f"{total:.2f}")
col2.metric("Average", f"{avg:.2f}")
col3.metric("Latest Value", f"{latest:.2f}")

# ------------------------------
# Charts
# ------------------------------
st.subheader("Trend Over Time")
fig = px.line(metric_df, x="timestamp", y="value", title=f"{selected_metric} over time")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Distribution")
fig2 = px.histogram(metric_df, x="value", nbins=10, title=f"{selected_metric} distribution")
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# Forecasting Module
# ------------------------------
st.subheader("🔮 Forecasting (Next 30 Days)")

try:
    forecast_df = metric_df.rename(columns={"timestamp": "ds", "value": "y"})
    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["y"], mode="lines+markers", name="Actual"))
    fig3.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
    fig3.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig3.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill="tonexty", line=dict(width=0), showlegend=False))

    fig3.update_layout(title=f"{selected_metric} – 30 Day Forecast", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig3, use_container_width=True)

except Exception as e:
    st.warning("⚠️ Not enough data for forecasting. Upload a larger dataset.")
    st.write(e)

# ------------------------------
# Data Preview
# ------------------------------
with st.expander("See raw data"):
    st.write(metric_df)

# ------------------------------
# Export Options
# ------------------------------
st.subheader("📥 Export Options")

# Export CSV
csv = metric_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, f"{tenant}_{selected_metric}.csv", "text/csv")

# Export PDF
def create_pdf_with_charts(tenant, metric, total, avg, latest, chart1, chart2, chart3=None):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"UrbanEdge Report - {tenant}", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)

    # KPIs
    pdf.cell(200, 10, f"Metric: {metric}", ln=True)
    pdf.cell(200, 10, f"Total: {total:.2f}", ln=True)
    pdf.cell(200, 10, f"Average: {avg:.2f}", ln=True)
    pdf.cell(200, 10, f"Latest Value: {latest:.2f}", ln=True)
    pdf.ln(10)

    # Save and insert charts
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart1.write_image(tmpfile.name, format="png")
        pdf.image(tmpfile.name, w=170)
    pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart2.write_image(tmpfile.name, format="png")
        pdf.image(tmpfile.name, w=170)
    pdf.ln(5)

    if chart3:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            chart3.write_image(tmpfile.name, format="png")
            pdf.image(tmpfile.name, w=170)

    # Return PDF as BytesIO
    pdf_bytes = bytes(pdf.output(dest="S"))
    return io.BytesIO(pdf_bytes)

if st.button("Generate PDF Report"):
    pdf_buffer = create_pdf_with_charts(
        tenant,
        selected_metric,
        total,
        avg,
        latest,
        fig,     # Trend chart
        fig2,    # Distribution chart
        fig3 if "fig3" in locals() else None  # Forecast chart if available
    )
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name=f"{tenant}_{selected_metric}_report.pdf",
        mime="application/pdf"
    )
    