UrbanEdge POC – Open Data Analytics Platform

🌍 UrbanEdge is a SaaS-based Open Data Analytics platform designed for municipalities and SMEs.
This Proof of Concept (POC) demonstrates how smart dashboards, sustainability KPIs, and forecasting can be delivered through a simple Streamlit app.

🔹 Features

✅ Multi-tenant support (e.g., City Tallinn, SMEs)
✅ Upload your own JSON datasets or use built-in mock data
✅ Prebuilt dashboards for energy, waste, traffic
✅ Key Performance Indicators (KPIs) summary
✅ Interactive charts & forecasts (30-day prediction using Prophet)
✅ Export data & reports: CSV + PDF with embedded charts
✅ Simple, cloud-ready SaaS demo

🔹 Tech Stack

Streamlit
 – app framework

Pandas
 – data processing

Plotly
 – interactive visualizations

Prophet
 – time-series forecasting

fpdf2
 – PDF report generation

Kaleido
 – export Plotly charts as images


🔹 Project Structure

 urbanedge-poc/
│── src/
│   ├── urbanedge_app.py         # main Streamlit app
│── data/
│   ├── tenant_datasets.py       # mock datasets per tenant
│── requirements.txt             # dependencies for Streamlit Cloud
│── README.md                    # project intro


🔹 Setup (Local)

# Clone repo
git clone https://github.com/<your-username>/urbanedge-poc.git
cd urbanedge-poc

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run src/urbanedge_app.py


🔹 Deploy on Streamlit Cloud

Push repo to GitHub.

Go to Streamlit Cloud

Create new app → point to:

src/urbanedge_app.py

Done ✅ → your app is live at:

https://urbanedge-poc.streamlit.app/


🔹 Roadmap

 Multi-metric dashboard (Energy + Waste + Traffic side-by-side)

 Live open data integration (Tallinn Open Data portal, EU Open Data)

 Role-based access & authentication

 More forecasting models (beyond Prophet)

 White-label SaaS model for municipalities


 🔹 About

This project is developed as a Proof of Concept (POC) for Beamline Accelerator funding application.
The vision is to transform open data into actionable insights, helping cities and businesses become more sustainable, efficient, and transparent.