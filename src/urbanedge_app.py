import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    train_df = metric_df.rename(columns={"timestamp": "ds", "value": "y"})
    model = Prophet()
    model.fit(train_df)

    future = model.make_future_dataframe(periods=30)
    forecast_df = model.predict(future)

    # Plot actual vs forecast
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=train_df["ds"], y=train_df["y"], mode="lines+markers", name="Actual"))
    fig3.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="Forecast"))
    fig3.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig3.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_lower"], mode="lines", fill="tonexty", line=dict(width=0), showlegend=False))

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

def format_date_axis(ax):
    """Clean up x-axis date labels to avoid overlap."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

def generate_matplotlib_chart(metric_df, metric, chart_type="line"):
    fig, ax = plt.subplots(figsize=(6,4))

    if chart_type == "line":
        ax.plot(metric_df["timestamp"], metric_df["value"], marker="o", color="green")
        ax.set_title(f"{metric} Trend Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

    elif chart_type == "hist":
        ax.hist(metric_df["value"], bins=10, color="skyblue", edgecolor="black")
        ax.set_title(f"{metric} Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    return fig

def fig_to_img(fig, dpi=200):
    """Convert a Matplotlib figure into an in-memory PNG buffer."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
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
    pdf.cell(200, 10, f"Latest Value: {latest:.2f}", ln=True)

    # Trend chart (Matplotlib line chart)
    trend_fig = generate_matplotlib_chart(metric_df, metric, chart_type="line")
    ax = trend_fig.axes[0]
    format_date_axis(ax)
    buf = fig_to_img(trend_fig, dpi=200)
    pdf.add_page()
    pdf.image(buf, x=10, y=20, w=180)

    # Distribution chart (Matplotlib histogram)
    hist_fig = generate_matplotlib_chart(metric_df, metric, chart_type="hist")
    buf = fig_to_img(hist_fig, dpi=200)
    pdf.add_page()
    pdf.image(buf, x=10, y=20, w=180)

    # Forecast chart (if provided)
    if forecast_df is not None and "yhat" in forecast_df:
        fig, ax = plt.subplots(figsize=(6,4))

        # Actual values
        ax.plot(metric_df["timestamp"], metric_df["value"], label="Actual", marker="o")

        # Forecast with confidence intervals
        ax.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", color="red")
        ax.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], 
                        color="pink", alpha=0.3, label="Confidence Interval")
        
        format_date_axis(ax)
        ax.set_title(f"{metric} – 30 Day Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        buf = fig_to_img(fig, dpi=200)
        pdf.add_page()
        pdf.image(buf, x=10, y=20, w=180)

    # Return PDF as bytes
    return bytes(pdf.output(dest="S"))

if st.button("Generate PDF Report"):
    pdf_buffer = create_pdf_with_charts(
        tenant,
        selected_metric,
        total,
        avg,
        latest,
        metric_df,
        forecast_df if "forecast" in locals() else None
    )   
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name=f"{tenant}_{selected_metric}_report.pdf",
        mime="application/pdf"
    )