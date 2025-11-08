import sys, os, shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import plotly.express as px
import plotly.graph_objs as go
import tempfile

from fpdf import FPDF  # for PDF export
from data.tenant_datasets import tenant_datasets

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from io import BytesIO

st.write("✅ Starting UrbanEdge...")

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
fig1.update_layout(
    autosize=True,
    margin=dict(l=80, r=40, t=40, b=80),  # more margin = smaller inside graph
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Distribution")
fig2 = px.histogram(metric_df, x="value", nbins=10, title=f"{selected_metric} distribution")
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# Forecasting Module
# ------------------------------
st.subheader("🔮 Forecasting (Next 30 Days)")
forecast_df = None

try:
    
    from neuralprophet import NeuralProphet

    train_df = metric_df.rename(columns={"timestamp": "ds", "value": "y"})
    train_df = train_df[["ds", "y"]] 

    # Clean old lightning logs to avoid checkpoint load errors
    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")

    model = NeuralProphet()

    model.fit(train_df,freq="D")
    future = model.make_future_dataframe(train_df,periods=30)
    forecast_df = model.predict(future)

    # Plot actual vs forecast
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=train_df["ds"], y=train_df["y"], mode="lines+markers", name="Actual"))

    forecast_col = "yhat1" if "yhat1" in forecast_df.columns else "yhat"

    fig3.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df[forecast_col],
        mode="lines",
        name="Forecast"
    ))

    # Confidence intervals may not always exist in NP, so check safely
    if "yhat1_upper" in forecast_df.columns and "yhat1_lower" in forecast_df.columns:
        fig3.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat1_upper"], mode="lines", line=dict(width=0), showlegend=False))
        fig3.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat1_lower"], mode="lines", fill="tonexty", line=dict(width=0), showlegend=False))

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

def fig_to_temp_png(fig, dpi=200):
    """Save a Matplotlib figure to a temporary PNG file and return its path."""
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp_file.name, format="png", dpi=dpi, bbox_inches="tight")
    tmp_file.close()
    return tmp_file.name


# Export PDF
def create_pdf_with_charts(tenant, metric, total, avg, latest, metric_df, forecast_df):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph(f"<b>UrbanEdge Report – {tenant}</b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))

    # Metrics section
    summary_text = f"""
    <b>Metric:</b> {metric}<br/>
    <b>Total:</b> {total:.2f}<br/>
    <b>Average:</b> {avg:.2f}<br/>
    <b>Latest Value:</b> {latest:.2f}<br/>
    """
    summary = Paragraph(summary_text, styles["BodyText"])
    story.append(summary)
    story.append(Spacer(1, 20))

    # Trend chart
    trend_fig = generate_matplotlib_chart(metric_df, metric, chart_type="line")
    ax = trend_fig.axes[0]
    format_date_axis(ax)
    buf1 = fig_to_temp_png(trend_fig, dpi=200)
    story.append(Image(buf1, width=450, height=250))
    story.append(Spacer(1, 20))

    # Histogram
    hist_fig = generate_matplotlib_chart(metric_df, metric, chart_type="hist")
    buf2 = fig_to_temp_png(hist_fig, dpi=200)
    story.append(Image(buf2, width=450, height=250))
    story.append(Spacer(1, 20))

    # Forecast chart
    if forecast_df is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(metric_df["timestamp"], metric_df["value"], label="Actual", marker="o")
        forecast_col = "yhat1" if "yhat1" in forecast_df.columns else "yhat"

        ax.plot(forecast_df["ds"], forecast_df[forecast_col], label="Forecast", color="red")

        upper_col = "yhat1_upper" if "yhat1_upper" in forecast_df.columns else "yhat_upper"
        lower_col = "yhat1_lower" if "yhat1_lower" in forecast_df.columns else "yhat_lower"

        if upper_col in forecast_df and lower_col in forecast_df:
            ax.fill_between(
                forecast_df["ds"],
                forecast_df[lower_col],
                forecast_df[upper_col],
                color="pink",
                alpha=0.3,
                label="Confidence"
            )

        format_date_axis(ax)
        ax.set_title(f"{metric} – 30 Day Forecast")
        ax.legend()
        fig.tight_layout()

        buf3 = fig_to_temp_png(fig, dpi=200)
        story.append(Image(buf3, width=450, height=250))

    doc.build(story)

    buffer.seek(0)
    return buffer.getvalue()

if st.button("Generate PDF Report"):
    pdf_buffer = create_pdf_with_charts(
        tenant,
        selected_metric,
        total,
        avg,
        latest,
        metric_df,
        forecast_df
    )   
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name=f"{tenant}_{selected_metric}_report.pdf",
        mime="application/pdf"
    )