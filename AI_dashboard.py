import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

# ---- Page Config ---- #
st.set_page_config(page_title="AI Sales Dashboard", layout="wide")
st.title("📈 AI-Powered Sales Insights Dashboard")

# ---- CSV Encoding Helper ---- #
def read_csv_auto(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding='ISO-8859-1')

# ---- File Upload ---- #
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

try:
    if uploaded_file is not None:
        df = read_csv_auto(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully")
    else:
        st.sidebar.info("Using default sample sales data")
        df = read_csv_auto("")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Debug: Display available columns
    st.write("📋 Columns in your file:", df.columns.tolist())

except Exception as e:
    st.error(f"❌ Error reading the CSV file: {e}")
    st.stop()

# ---- Data Preprocessing ---- #
if 'Month' not in df.columns or 'Sales' not in df.columns:
    st.error("⚠️ CSV must have 'Month' and 'Sales' columns (case-sensitive)")
    st.stop()

df = df[['Month', 'Sales']].copy()
df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df = df.dropna()

# ---- Forecasting ---- #
x = df[['Month']]
y = df['Sales']
model = LinearRegression()
model.fit(x, y)

future_months = pd.DataFrame({'Month': range(int(df['Month'].max()) + 1, int(df['Month'].max()) + 7)})
future_months['Forecast'] = model.predict(future_months[['Month']])

# ---- Anomaly Detection ---- #
iso = IsolationForest(contamination=0.15, random_state=42)
df['anomaly'] = iso.fit_predict(df[['Sales']])
anomalies = df[df['anomaly'] == -1]

# ---- Group for Clean Visualization ---- #
monthly_avg = df.groupby('Month')['Sales'].mean().reset_index()
monthly_anomalies = anomalies.groupby('Month')['Sales'].mean().reset_index()

# ---- Visualizations ---- #
st.subheader("📊 Average Monthly Sales with Anomalies Highlighted")

fig = px.line(
    monthly_avg, x='Month', y='Sales',
    title='Average Monthly Sales',
    markers=True, line_shape='spline'
)

fig.add_scatter(
    x=monthly_anomalies['Month'],
    y=monthly_anomalies['Sales'],
    mode='markers',
    marker=dict(color='red', size=12, symbol='circle'),
    name='Anomalies'
)

fig.update_traces(line=dict(width=3), selector=dict(type='scatter', mode='lines'))
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

st.plotly_chart(fig, use_container_width=True)

# ---- Forecast Plot ---- #
st.subheader("🔮 Forecasted Sales for Next 6 Months")

forecast_fig = px.line(future_months, x='Month', y='Forecast', markers=True, title="Sales Forecast")
forecast_fig.update_traces(line=dict(width=3), marker=dict(size=8, color='cyan'))
forecast_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

st.plotly_chart(forecast_fig, use_container_width=True)

# ---- Auto Summary ---- #
st.subheader("🧠 Auto-Generated Summary")
total_sales = df['Sales'].sum()
avg_sales = df['Sales'].mean()
peak_month = df.loc[df['Sales'].idxmax(), 'Month']
anomaly_count = len(anomalies)

st.markdown(f"""
- **Total sales:** ₹{total_sales:,.2f}  
- **Average monthly sales:** ₹{avg_sales:,.2f}  
- **Peak sales occurred in month:** {int(peak_month)}  
- **Detected anomalies:** {anomaly_count} month(s)
""")

st.info("📤 Upload your own CSV file in the sidebar to visualize and analyze your sales performance.")
