import os
import time
import requests
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="itcs355_project | Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
)

# --- Endpoints ---
DEFAULT_PREDICT_URL = os.getenv(
    "ENDPOINT_NAME", "https://itcs355-project.onrender.com/predict"
)
DEFAULT_HEALTH_URL = "https://itcs355-project.onrender.com/health"

# --- Feature Bounds & Defaults ---
FEATURES = {
    "fixed acidity": {"min": 4.0, "max": 16.0, "default": 7.4, "step": 0.1},
    "volatile acidity": {"min": 0.0, "max": 2.6, "default": 0.70, "step": 0.01},
    "citric acid": {"min": 0.0, "max": 1.0, "default": 0.00, "step": 0.01},
    "residual sugar": {"min": 0.5, "max": 15.5, "default": 1.9, "step": 0.1},
    "chlorides": {"min": 0.01, "max": 0.60, "default": 0.076, "step": 0.001},
    "free sulfur dioxide": {
        "min": 1.0,
        "max": 72.0,
        "default": 11.0,
        "step": 1.0,
    },
    "total sulfur dioxide": {
        "min": 6.0,
        "max": 289.0,
        "default": 34.0,
        "step": 1.0,
    },
    "density": {"min": 0.9900, "max": 1.0037, "default": 0.9978, "step": 0.0005},
    "pH": {"min": 2.70, "max": 4.01, "default": 3.51, "step": 0.01},
    "sulphates": {"min": 0.30, "max": 2.00, "default": 0.56, "step": 0.01},
    "alcohol": {"min": 8.0, "max": 14.9, "default": 9.4, "step": 0.1},
}


def ping_health(health_url: str) -> tuple[bool, int, str]:
    """Sends a real HTTP GET request to Render health endpoint."""
    try:
        res = requests.get(health_url, timeout=10)
        return res.status_code == 200, res.status_code, res.text
    except requests.exceptions.RequestException as e:
        return False, 0, str(e)


# --- UI Layout ---
st.title("🍷 Live Wine Quality Predictor")
st.caption("ITCS355 MLOps Deployment Demo — Real API Invocations")

# Sidebar
st.sidebar.header("🌐 Render Service Connection")
predict_url = st.sidebar.text_input("Prediction URL", value=DEFAULT_PREDICT_URL)
health_url = st.sidebar.text_input("Health URL", value=DEFAULT_HEALTH_URL)
api_token = st.sidebar.text_input(
    "API Secret Key / Token (if required)", type="password"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Instance Status")

if st.sidebar.button("🔍 Check Service Health"):
    with st.spinner("Pinging Render container..."):
        is_ok, status, msg = ping_health(health_url)
        if is_ok:
            st.sidebar.success(f"Online (HTTP {status}) 🟢")
        else:
            st.sidebar.error(f"Offline / Booting (Status: {status}) 🟡")

if st.sidebar.button("⚡ Warmup Render Cold Start"):
    progress_bar = st.progress(0)
    status_box = st.empty()
    for i in range(6):
        status_box.info(f"Attempt {i+1}/6: Pinging {health_url}...")
        is_ok, status, _ = ping_health(health_url)
        progress_bar.progress((i + 1) / 6)
        if is_ok:
            status_box.success("Render instance is awake and ready! 🟢")
            break
        time.sleep(10)

# Main Form
st.subheader("1. Configure Input Features")
col1, col2, col3 = st.columns(3)

payload = {}
keys = list(FEATURES.keys())
for idx, key in enumerate(keys):
    cfg = FEATURES[key]
    target_col = [col1, col2, col3][idx % 3]
    payload[key] = target_col.number_input(
        label=f"{key.title()}",
        min_value=cfg["min"],
        max_value=cfg["max"],
        value=cfg["default"],
        step=cfg["step"],
        format=(
            "%.4f"
            if cfg["step"] < 0.01
            else "%.2f" if cfg["step"] < 1 else "%d"
        ),
    )

st.subheader("2. Send Real POST Request to Render")

if st.button("🚀 Invoke Live Prediction Endpoint", type="primary"):
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    st.write(f"**Target URL:** `{predict_url}`")
    with st.expander("Show Request Payload JSON"):
        st.json(payload)

    with st.spinner("Sending HTTP POST request to Render..."):
        start_time = time.time()
        try:
            # 60s timeout accommodates Render cold boots if not pre-warmed
            response = requests.post(
                predict_url, json=payload, headers=headers, timeout=60
            )
            elapsed_ms = (time.time() - start_time) * 1000

            st.markdown("---")
            st.subheader("3. Render Response")

            col_res1, col_res2 = st.columns(2)
            col_res1.metric("HTTP Status Code", response.status_code)
            col_res2.metric("Response Time", f"{elapsed_ms:.1f} ms")

            if response.status_code == 200:
                res_data = response.json()
                st.success("✓ Live Prediction Call Successful!")
                st.json(res_data)
            else:
                st.error(f"❌ Server Error ({response.status_code})")
                st.code(response.text)

        except requests.exceptions.Timeout:
            st.error(
                "⌛ Request Timed Out (60s). Render container is likely sleeping. Use the 'Warmup Render Cold Start' button in the sidebar."
            )
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network Request Failed: {str(e)}")
