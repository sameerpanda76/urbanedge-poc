import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import json
import plotly.io as pio
from io import BytesIO
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
st.title("📊 Open Data Analytics")
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
fig1 = px.line(metric_df, x="timestamp", y="value", title=f"{selected_metric} over time")
st.plotly_chart(fig1, use_container_width=True)

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

def fig_to_img(fig, scale=3):
    """Convert a Plotly figure into PNG bytes using Kaleido."""
    buf = BytesIO(pio.to_image(fig, format="png", scale=scale))
    buf.seek(0)
    return buf

# Export PDF
def create_pdf_with_charts(tenant, metric, total, avg, latest, chart1=None, chart2=None, chart3=None):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, f"UrbanEdge Report - {tenant}", ln=True, align="C")

    # Metrics
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Metric: {metric}", ln=True)
    pdf.cell(200, 10, f"Total: {total:.2f}", ln=True)
    pdf.cell(200, 10, f"Average: {avg:.2f}", ln=True)
    pdf.cell(200, 10, f"Latest: {latest:.2f}", ln=True)

    # Add charts if available
    for fig in [chart1, chart2, chart3]:
        if fig:
            buf = fig_to_img(fig, scale=3)   # High-quality PNG
            pdf.add_page()
            pdf.image(buf, x=10, y=20, w=180)

    # Return PDF as bytes for Streamlit download
    return bytes(pdf.output(dest="S"))

if st.button("Generate PDF Report"):
    pdf_buffer = create_pdf_with_charts(
        tenant,
        selected_metric,
        total,
        avg,
        latest,
        fig1 if "fig1" in locals() else None,
        fig2 if "fig2" in locals() else None,
        fig3 if "fig3" in locals() else None
    )   
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name=f"{tenant}_{selected_metric}_report.pdf",
        mime="application/pdf"
    )