# CPG company ESG Intelligence Platform - AI Kit A (Refined & Robust)
# Production-ready version with comprehensive error handling and validation
# Works offline; uses Anthropic/OpenAI if keys are present.
# ---------------------------------------------------------------

import os
import sys
import time
import json
import base64
import random
import hmac
import hashlib
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional: ML (IsolationForest) for anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
DEFAULT_SUPPLIER_API = "https://api.example.com/suppliers"
DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_ATTESTATION_SECRET = "demo-secret"
MIN_COMPLETENESS_THRESHOLD = 100.0
CONTAMINATION_RATE = 0.03

# --------------------------------------------------------------------------------
# Page configuration & styles
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="CPG company ESG Intelligence Platform - AI Kit A",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stButton>button {
        background-color: #003DA5; color: white; font-weight: 600;
        border-radius: 6px; border: none; padding: 0.5rem 1rem; transition: all 0.2s;
    }
    .stButton>button:hover { background-color: #002D7A; transform: translateY(-1px); }
    .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; color: #fff; }
    .pill.ok { background: #16a34a; } .pill.warn { background: #eab308; } .pill.err { background: #dc2626; }
    .header-style {
        background: linear-gradient(90deg, #003DA5 0%, #0052CC 100%);
        padding: 1.2rem 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;
    }
    .header-style h1, .header-style p {
        color: #FFFFFF !important;
        margin: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 12px; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    .box {
        border: 1px solid #eee; border-radius: 10px; padding: 1rem; background: #fafafa;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------------
def initialize_session_state():
    """Initialize all session state variables with default values."""
    defaults = {
        "csrd_df": None,
        "waste_df": None,
        "evidence_rows": None,
        "assurance_df": None,
        "nav": "Executive Dashboard",
        "trace_log": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --------------------------------------------------------------------------------
# Demo data generators (deterministic for repeatable demos)
# --------------------------------------------------------------------------------
def gen_csrd_sample() -> pd.DataFrame:
    """Generate sample CSRD compliance data."""
    try:
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="M")
        sites = ["Plant A - France", "Plant B - Germany", "Plant C - Belgium"]
        
        n_rows = len(dates) * len(sites)
        df = pd.DataFrame({
            "date": np.tile(dates, len(sites)),
            "facility": np.repeat(sites, len(dates)),
            "total_water_withdrawal": np.random.uniform(1000, 5000, n_rows),
            "water_consumption": np.random.uniform(800, 4000, n_rows),
            "water_discharge": np.random.uniform(200, 1200, n_rows),
            "water_recycled_percentage": np.random.uniform(20, 60, n_rows),
            "plastic_packaging_total": np.random.uniform(100, 500, n_rows),
            "recycled_content_percentage": np.random.uniform(10, 40, n_rows),
            "recyclable_packaging_percentage": np.random.uniform(60, 95, n_rows),
            "hazardous_waste": np.random.uniform(10, 100, n_rows),
            "non_hazardous_waste": np.random.uniform(200, 1000, n_rows),
            "waste_to_landfill": np.random.uniform(50, 300, n_rows),
            "waste_recycled": np.random.uniform(100, 600, n_rows),
            "air_emissions_nox": np.random.uniform(1, 20, n_rows),
            "air_emissions_sox": np.random.uniform(0.5, 10, n_rows),
            "air_emissions_pm": np.random.uniform(0.1, 5, n_rows),
            "product_carbon_footprint": np.random.uniform(0.5, 2.5, n_rows),
        })
        return df
    except Exception as e:
        st.error(f"Error generating CSRD sample data: {e}")
        return pd.DataFrame()

def gen_waste_sample() -> pd.DataFrame:
    """Generate sample waste data."""
    try:
        np.random.seed(7)
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
        products = ["Head & Shoulders", "Pantene", "Olay", "SK-II", "Gillette"]
        plants = ["Plant A - France", "Plant B - Germany", "Plant C - Belgium"]
        rows = []
        
        for d in dates:
            for _ in range(np.random.randint(5, 9)):
                rows.append({
                    "date": d,
                    "product_line": random.choice(products),
                    "plant": random.choice(plants),
                    "batch_id": f"BATCH-{np.random.randint(1000, 9999)}",
                    "production_volume": np.random.uniform(1000, 10000),
                    "waste_kg": np.random.uniform(50, 500),
                    "waste_type": random.choice(["Packaging", "Raw Material", "Product Defects", "Cleaning"]),
                    "waste_reason": random.choice(["Quality Issue", "Changeover", "Expiry", "Equipment Failure", "Human Error"]),
                    "disposal_method": random.choice(["Recycled", "Incinerated", "Landfill", "Composted"]),
                    "cost_usd": np.random.uniform(100, 2000),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error generating waste sample data: {e}")
        return pd.DataFrame()

def gen_evidence_manifest() -> pd.DataFrame:
    """Generate sample evidence manifest."""
    try:
        np.random.seed(9)
        return pd.DataFrame({
            "id": [f"EVD-{i:03d}" for i in range(1, 13)],
            "supplier_name": np.random.choice(["EcoMat Ltd", "PlastoChem GmbH", "ReGenPolymers NV"], 12),
            "evidence_type": np.random.choice(["SVHC Declaration", "Recycled Content Proof", "ISO14001 Certificate"], 12),
            "submission_date": pd.date_range("2024-01-05", periods=12, freq="15D"),
            "file_name": [f"evidence_{i}.pdf" for i in range(1, 13)],
            "status": np.random.choice(["Submitted", "Pending Review", "Approved"], 12),
        })
    except Exception as e:
        st.error(f"Error generating evidence manifest: {e}")
        return pd.DataFrame()

def gen_assurance_tracker() -> pd.DataFrame:
    """Generate sample assurance tracker."""
    try:
        np.random.seed(5)
        return pd.DataFrame({
            "requirement": [f"ESRS E{n}-0{n}" for n in range(1, 6)],
            "control_owner": np.random.choice(["QA Lead", "Sustainability Manager", "Ops Head"], 5),
            "completion_status": np.random.choice(["Completed", "In Progress", "Not Started"], 5),
            "last_update": pd.date_range("2024-02-01", periods=5, freq="9D"),
            "audit_ready": np.random.choice(["Yes", "No"], 5),
        })
    except Exception as e:
        st.error(f"Error generating assurance tracker: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------
# API-READY: service stubs with error handling
# --------------------------------------------------------------------------------
@dataclass
class APIConfig:
    """Configuration for API clients."""
    base_url: str
    api_key: Optional[str] = None

class SupplierAPIClient:
    """Client for supplier portal API with robust error handling."""
    
    def __init__(self, cfg: APIConfig):
        self.cfg = cfg
    
    def health(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            time.sleep(0.2)
            ok = self.cfg.base_url.startswith(("http://", "https://"))
            return {"service": "supplier-api", "ok": ok, "base_url": self.cfg.base_url}
        except Exception as e:
            return {"service": "supplier-api", "ok": False, "error": str(e)}
    
    def fetch_suppliers(self) -> List[str]:
        """Fetch list of suppliers."""
        try:
            time.sleep(0.2)
            return ["EcoMat Ltd", "PlastoChem GmbH", "ReGenPolymers NV"]
        except Exception as e:
            st.error(f"Error fetching suppliers: {e}")
            return []
    
    def submit_magic_link(self, supplier_name: str) -> Dict[str, Any]:
        """Generate magic link for supplier portal."""
        try:
            if not supplier_name:
                raise ValueError("Supplier name cannot be empty")
            
            time.sleep(0.3)
            safe_name = supplier_name.replace(' ', '_').lower()
            return {
                "supplier": supplier_name,
                "link": f"{self.cfg.base_url.rstrip('/')}/magic/{safe_name}",
                "expires_in_hours": 72,
                "status": "sent"
            }
        except Exception as e:
            return {"supplier": supplier_name, "error": str(e), "status": "failed"}

class LLMClient:
    """Offline-first LLM wrapper with error handling."""
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
        self.has_openai = bool(os.getenv("OPENAI_API_KEY"))
        self.has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    def health(self) -> Dict[str, Any]:
        """Check LLM service health."""
        ok = self.has_openai or self.has_anthropic
        return {"service": f"llm-{self.provider}", "ok": ok}
    
    def generate(self, prompt: str) -> str:
        """Generate text using LLM with fallback to demo mode."""
        if not prompt:
            return "[Error: Empty prompt]"
        
        if self.has_openai:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You create concise, audit-ready ESG text with numbered actions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                st.warning(f"OpenAI error: {e}. Falling back to demo mode.")
        
        if self.has_anthropic:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                msg = client.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                    max_tokens=800,
                    system="You create concise, audit-ready ESG text with numbered actions.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return msg.content[0].text
            except Exception as e:
                st.warning(f"Anthropic error: {e}. Falling back to demo mode.")
        
        # Fallback to demo mode
        time.sleep(0.4)
        return f"[AI (offline demo)] {prompt[:400]}\n...(truncated)"

# Initialize API clients with error handling
try:
    SUPPLIER_API = SupplierAPIClient(APIConfig(
        base_url=os.getenv("SUPPLIER_API_BASE", DEFAULT_SUPPLIER_API),
        api_key=os.getenv("SUPPLIER_API_KEY")
    ))
    LLM = LLMClient()
except Exception as e:
    st.error(f"Error initializing API clients: {e}")
    st.stop()

# HMAC secret for attestation
ATTESTATION_SECRET = os.getenv("ATTESTATION_SECRET", DEFAULT_ATTESTATION_SECRET)

# --------------------------------------------------------------------------------
# Common utilities with error handling
# --------------------------------------------------------------------------------
def compliance_by_topic() -> pd.DataFrame:
    """Generate compliance metrics by topic."""
    try:
        return pd.DataFrame({
            "Topic": ["Water", "Circular Economy", "Pollution", "Climate", "Biodiversity"],
            "Compliance": [85, 72, 78, 92, 65],
            "Target": [95, 95, 95, 95, 95]
        })
    except Exception as e:
        st.error(f"Error generating compliance data: {e}")
        return pd.DataFrame()

def show_header():
    """Display application header."""
    st.markdown("""
    <div class="header-style">
        <h1 style='margin:0;color:#fff;'>🌍 CPG company ESG Intelligence Platform - AI Kit A</h1>
        <p style='margin:6px 0 0 0;font-size:16px;color:#fff;'>AI-powered CSRD readiness • Waste optimization • Supplier assurance</p>
    </div>
    """, unsafe_allow_html=True)

def safe_selectbox(label: str, options: List, key: str, **kwargs):
    """Wrapper for selectbox with validation."""
    if not options:
        st.warning(f"No options available for {label}")
        return None
    return st.selectbox(label, options, key=key, **kwargs)

def safe_metric(label: str, value: Any, delta: Optional[str] = None):
    """Display metric with error handling."""
    try:
        st.metric(label, value, delta)
    except Exception as e:
        st.error(f"Error displaying metric {label}: {e}")

# --------------------------------------------------------------------------------
# Evidence completeness rules & attestation helpers
# --------------------------------------------------------------------------------
REQUIRED_EVIDENCE = {
    "Circular Economy": ["Recycled Content Proof", "ISO14001 Certificate"],
    "Pollution": ["SVHC Declaration", "ISO14001 Certificate"],
    "Water": ["ISO14001 Certificate"]
}

def score_evidence_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Score evidence completeness by supplier and topic."""
    try:
        if df is None or df.empty:
            return pd.DataFrame(columns=["supplier_name", "topic", "required", "have", "missing", "completeness"])
        
        rows = []
        for topic, reqs in REQUIRED_EVIDENCE.items():
            for supplier, grp in df.groupby("supplier_name"):
                have_types = set(grp["evidence_type"].tolist())
                missing = [x for x in reqs if x not in have_types]
                completeness = 100.0 * (len(reqs) - len(missing)) / len(reqs) if reqs else 0.0
                
                rows.append({
                    "supplier_name": supplier,
                    "topic": topic,
                    "required": ", ".join(reqs),
                    "have": ", ".join(sorted(have_types)),
                    "missing": ", ".join(missing) if missing else "",
                    "completeness": completeness
                })
        
        return pd.DataFrame(rows).sort_values(["completeness", "supplier_name"], ascending=[True, True])
    except Exception as e:
        st.error(f"Error scoring evidence completeness: {e}")
        return pd.DataFrame()

def sign_text_hmac(text: str, secret: str) -> str:
    """Sign text with HMAC-SHA256."""
    try:
        if not text or not secret:
            raise ValueError("Text and secret cannot be empty")
        return hmac.new(secret.encode("utf-8"), text.encode("utf-8"), hashlib.sha256).hexdigest()
    except Exception as e:
        st.error(f"Error signing text: {e}")
        return ""

def verify_text_hmac(text: str, signature: str, secret: str) -> bool:
    """Verify HMAC signature."""
    try:
        if not all([text, signature, secret]):
            return False
        calc = sign_text_hmac(text, secret)
        return hmac.compare_digest(calc, signature)
    except Exception:
        return False

# --------------------------------------------------------------------------------
# Request trace with error handling
# --------------------------------------------------------------------------------
def trace(event: str, **kw):
    """Append a timestamped trace event to session state."""
    try:
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
            **kw
        }
        st.session_state.trace_log.append(entry)
    except Exception as e:
        # Silent fail for tracing to not disrupt main flow
        pass

# --------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------
with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Procter_%26_Gamble_logo.svg/240px-Procter_%26_Gamble_logo.svg.png", width=130)
    except Exception:
        st.markdown("**CPG Logo**")
    
    nav = st.radio(
        "📊 Navigation",
        [
            "Executive Dashboard",
            "CSRD Compliance Analyzer",
            "Waste Optimizer",
            "Supplier Portal",
            "Assurance Readiness",
            "Agent: Evidence Authenticator",
            "Report Generator",
            "API & Integrations (Demo)"
        ],
        key="nav_radio_main"
    )
    st.session_state.nav = nav
    st.markdown("---")
    st.caption(f"Demo build: {datetime.now().strftime('%b %d, %Y')} | v4.1 (Refined)")

# --------------------------------------------------------------------------------
# Executive Dashboard
# --------------------------------------------------------------------------------
def page_dashboard():
    """Display executive dashboard with key metrics."""
    st.markdown("## 📊 Executive Dashboard")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><div>Fine Risk Identified</div><h2 style='margin:4px 0'>EUR 1.8M</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div>CSRD Compliance</div><h2 style='margin:4px 0'>78%</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div>Waste Savings Potential</div><h2 style='margin:4px 0'>EUR 2.3M</h2></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div>Waste Diverted</div><h2 style='margin:4px 0'>92%</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Charts
    try:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("CSRD Compliance by Topic")
            df = compliance_by_topic()
            if not df.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Current", x=df["Topic"], y=df["Compliance"], marker_color="#003DA5"))
                fig.add_trace(go.Bar(name="Target", x=df["Topic"], y=df["Target"], marker_color="#E0E0E0"))
                fig.update_layout(height=330, barmode="overlay", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("Monthly Waste Trend")
            dates = pd.date_range("2023-01-01", "2023-12-31", freq="M")
            waste_trend = pd.DataFrame({
                "Month": dates,
                "Waste": np.random.uniform(8000, 12000, len(dates))
            })
            fig2 = px.line(waste_trend, x="Month", y="Waste", markers=True)
            fig2.update_traces(line_color="#003DA5", line_width=3)
            fig2.update_layout(height=330, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Risk matrix
        st.subheader("🎯 CSRD Risk Matrix")
        risk = pd.DataFrame({
            "Topic": ["Water stress disclosure", "Scope 3 emissions", "Circular design metrics", "SVHC tracking", "Biodiversity impact"],
            "Likelihood": [3, 4, 2, 4, 2],
            "Impact": [4, 5, 3, 5, 3],
        })
        risk["Risk_Score"] = risk["Likelihood"] * risk["Impact"]
        fig3 = px.scatter(
            risk, x="Likelihood", y="Impact", size="Risk_Score", color="Risk_Score",
            text="Topic", color_continuous_scale="RdYlGn_r",
            labels={"Likelihood": "Likelihood (1-5)", "Impact": "Impact (1-5)"}
        )
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying dashboard charts: {e}")

# --------------------------------------------------------------------------------
# CSRD Compliance Analyzer
# --------------------------------------------------------------------------------
ESRS_GAPS = {
    "Water": [
        {
            "requirement": "ESRS E3-1: Water management policies",
            "status": "Partial",
            "gap": "Missing water stress assessment for 3 facilities",
            "risk": "Medium",
            "fine_risk": "EUR 50,000 - EUR 200,000"
        },
        {
            "requirement": "ESRS E3-4: Water consumption intensity",
            "status": "Missing",
            "gap": "No product-level water intensity metrics",
            "risk": "High",
            "fine_risk": "EUR 200,000 - EUR 500,000"
        },
        {
            "requirement": "ESRS E3-5: Water discharge quality",
            "status": "Complete",
            "gap": "None",
            "risk": "Low",
            "fine_risk": "EUR 0"
        },
    ],
    "Circular Economy": [
        {
            "requirement": "ESRS E5-1: Resource inflows",
            "status": "Partial",
            "gap": "Missing supplier recycled-content certificates for 40% of materials",
            "risk": "High",
            "fine_risk": "EUR 300,000 - EUR 700,000"
        },
        {
            "requirement": "ESRS E5-2: Resource outflows",
            "status": "Complete",
            "gap": "None",
            "risk": "Low",
            "fine_risk": "EUR 0"
        },
        {
            "requirement": "ESRS E5-5: Circular design",
            "status": "Missing",
            "gap": "No systematic circularity assessment for new products",
            "risk": "Medium",
            "fine_risk": "EUR 100,000 - EUR 300,000"
        },
    ],
    "Pollution": [
        {
            "requirement": "ESRS E2-1: Pollution policies",
            "status": "Complete",
            "gap": "None",
            "risk": "Low",
            "fine_risk": "EUR 0"
        },
        {
            "requirement": "ESRS E2-4: Pollution of air",
            "status": "Partial",
            "gap": "Missing Scope 3 air emission data from logistics",
            "risk": "Medium",
            "fine_risk": "EUR 150,000 - EUR 400,000"
        },
        {
            "requirement": "ESRS E2-6: Substances of concern",
            "status": "Missing",
            "gap": "Incomplete SVHC tracking in supply chain",
            "risk": "High",
            "fine_risk": "EUR 500,000 - EUR 1,000,000"
        },
    ]
}

def page_csrd():
    """CSRD Compliance Analyzer page."""
    st.markdown("## 🔍 CSRD Compliance Analyzer")
    tab1, tab2, tab3 = st.tabs(["📤 Data Upload", "🎯 Gap Analysis", "📝 Narrative Generator"])

    with tab1:
        st.markdown("### Upload ESG Data or Load Demo")
        c1, c2 = st.columns([2, 1])
        
        with c1:
            f = st.file_uploader("Choose ESG data file (CSV/Excel)", type=["csv", "xlsx"], key="csrd_uploader")
            if st.button("🔬 Load Demo Data", key="csrd_demo_btn"):
                st.session_state.csrd_df = gen_csrd_sample()
                if not st.session_state.csrd_df.empty:
                    st.success("Demo CSRD dataset loaded.")
                    trace("csrd.demo_loaded", rows=len(st.session_state.csrd_df))
            
            if f is not None:
                try:
                    st.session_state.csrd_df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                    st.success(f"Loaded {f.name} ({len(st.session_state.csrd_df)} rows).")
                    trace("csrd.upload_loaded", rows=len(st.session_state.csrd_df), filename=f.name)
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                    trace("csrd.upload_error", error=str(e))

        with c2:
            st.markdown("**Required (examples)**")
            st.markdown("""
            - Water: withdrawal, consumption, discharge, recycling
            - Circular economy: material inflows/outflows, recycled %
            - Pollution: NOx, SOx, PM, substances of concern
            """)

        if st.session_state.csrd_df is not None and not st.session_state.csrd_df.empty:
            st.markdown("---")
            st.dataframe(st.session_state.csrd_df.head(), use_container_width=True)
            
            try:
                c1, c2, c3 = st.columns(3)
                with c1:
                    safe_metric("Rows", len(st.session_state.csrd_df))
                with c2:
                    safe_metric("Columns", len(st.session_state.csrd_df.columns))
                with c3:
                    total_cells = len(st.session_state.csrd_df) * len(st.session_state.csrd_df.columns)
                    filled_cells = st.session_state.csrd_df.notna().sum().sum()
                    comp = 100 * (filled_cells / total_cells) if total_cells > 0 else 0
                    safe_metric("Completeness", f"{comp:.1f}%")
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")

    with tab2:
        st.markdown("### 🎯 ESRS Gap Analysis")
        if st.session_state.csrd_df is None or st.session_state.csrd_df.empty:
            st.info("Load data in the 'Data Upload' tab.")
        else:
            topic = safe_selectbox("Select ESRS Topic", list(ESRS_GAPS.keys()), key="csrd_topic_select")
            if topic and st.button("🔍 Analyze Gaps", key="csrd_analyze_btn"):
                try:
                    gaps = ESRS_GAPS[topic]
                    total_max = 0
                    high_risk = 0
                    
                    for g in gaps:
                        risk = g["risk"]
                        if risk == "High":
                            badge = "🔴"
                            high_risk += 1
                        elif risk == "Medium":
                            badge = "🟡"
                        else:
                            badge = "🟢"
                        
                        with st.expander(f"{badge} {g['requirement']} - {g['status']}"):
                            c1, c2 = st.columns([2, 1])
                            with c1:
                                st.markdown(f"**Gap:** {g['gap']}")
                                st.markdown(f"**Risk:** {g['risk']}")
                            with c2:
                                st.markdown(f"**Fine Risk:** {g['fine_risk']}")
                        
                        # Calculate total risk
                        if "EUR" in g["fine_risk"] and "-" in g["fine_risk"]:
                            try:
                                max_part = g["fine_risk"].split("-")[1].replace("EUR", "").replace(",", "").strip()
                                total_max += int(max_part)
                            except Exception:
                                pass
                    
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"<div class='box'><b>High risk items</b><h3>{high_risk}</h3></div>", unsafe_allow_html=True)
                    with c2:
                        score = max(0, 100 - high_risk * 15)
                        st.markdown(f"<div class='box'><b>Compliance Score</b><h3>{score}%</h3></div>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<div class='box'><b>Total Fine Risk (max)</b><h3>EUR {total_max:,}</h3></div>", unsafe_allow_html=True)

                    st.info("""**Priority Actions**
1) 🔴 Implement supplier portal for SVHC & recycled-content certificates  
2) 🟡 Complete water stress assessments via WRI Aqueduct  
3) 🟢 Add product-level water intensity metrics (LCA updates)
""")
                    trace("csrd.gap_analysis", topic=topic, high_risk=high_risk, total_fine_risk=total_max)
                except Exception as e:
                    st.error(f"Error analyzing gaps: {e}")

    with tab3:
        st.markdown("### 📝 Disclosure Narrative Generator")
        if st.session_state.csrd_df is None or st.session_state.csrd_df.empty:
            st.info("Load data in the 'Data Upload' tab.")
        else:
            topic = safe_selectbox("Narrative topic", ["Water", "Circular Economy", "Pollution"], key="narrative_topic")
            audit_ready = st.checkbox("This data is audit-ready", value=True, key="narrative_audit_ready")
            word_count = st.select_slider("Word Count", options=[300, 500, 1000, 1500], value=500, key="narrative_wc")
            
            if st.button("📝 Generate Narrative", key="narrative_btn"):
                if not audit_ready:
                    st.error("Data is not marked audit-ready. Complete data collection first.")
                else:
                    try:
                        prompt = f"""Create an ESRS disclosure for {topic} (~{word_count} words).
Include governance, material impacts, risk management, metrics & YoY, targets, and supplier evidence needs.
Be concise and CFO/assurance friendly."""
                        
                        with st.spinner("Generating narrative..."):
                            out = LLM.generate(prompt)
                        
                        trace("llm.generate", provider=LLM.provider, ok=bool(out), topic=topic)
                        st.markdown("#### Generated Narrative")
                        st.markdown(out)
                        st.download_button(
                            "📥 Download Narrative (Markdown)",
                            data=out,
                            file_name=f"ESRS_{topic}_Narrative.md",
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"Error generating narrative: {e}")
                        trace("llm.generate_error", error=str(e))

# --------------------------------------------------------------------------------
# Waste Optimizer
# --------------------------------------------------------------------------------
def page_waste():
    """Waste Optimizer page."""
    st.markdown("## ♻️ Waste Optimizer for Beauty Products")
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Input", "📊 Analysis", "💰 Savings Simulator", "📈 Insights"])

    with tab1:
        st.markdown("### Upload Monthly Waste Data or Generate Demo")
        c1, c2 = st.columns([2, 1])
        
        with c1:
            f = st.file_uploader("Upload waste data (CSV/Excel)", type=["csv", "xlsx"], key="waste_uploader")
            if st.button("🔬 Generate Demo Data", key="waste_demo_btn"):
                st.session_state.waste_df = gen_waste_sample()
                if not st.session_state.waste_df.empty:
                    st.success("Demo waste dataset generated.")
                    trace("waste.demo_generated", rows=len(st.session_state.waste_df))
            
            if f is not None:
                try:
                    st.session_state.waste_df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                    st.success(f"Loaded {f.name} ({len(st.session_state.waste_df)} rows).")
                    trace("waste.upload_loaded", rows=len(st.session_state.waste_df), filename=f.name)
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                    trace("waste.upload_error", error=str(e))
        
        with c2:
            st.markdown("**Required Fields**")
            st.caption("date, product_line, plant, batch_id, production_volume, waste_kg, waste_type, waste_reason, disposal_method, cost_usd")

        if st.session_state.waste_df is not None and not st.session_state.waste_df.empty:
            st.markdown("---")
            df = st.session_state.waste_df
            
            try:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    total_waste = df['waste_kg'].sum() / 1000
                    safe_metric("Total Waste", f"{total_waste:.1f} tonnes")
                with c2:
                    total_cost = df['cost_usd'].sum() / 1e6
                    safe_metric("Total Cost", f"EUR {total_cost:.2f}M")
                with c3:
                    avg_waste_rate = (df['waste_kg'].sum() / df['production_volume'].sum()) * 100
                    safe_metric("Avg Waste Rate", f"{avg_waste_rate:.2f}%")
                with c4:
                    recycling_rate = (len(df[df['disposal_method'] == 'Recycled']) / len(df)) * 100
                    safe_metric("Recycling Rate", f"{recycling_rate:.1f}%")
                
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying waste metrics: {e}")

    with tab2:
        st.markdown("### 📊 Top Waste Loss Drivers & Visuals")
        if st.session_state.waste_df is None or st.session_state.waste_df.empty:
            st.info("Load or generate data first.")
        else:
            try:
                df = st.session_state.waste_df.copy()
                by_prod = df.groupby("product_line")["waste_kg"].sum().sort_values(ascending=False)
                by_reason = df.groupby("waste_reason")["waste_kg"].sum().sort_values(ascending=False)
                by_plant = df.groupby("plant")["waste_kg"].sum().sort_values(ascending=False)

                st.markdown("#### 🎯 Priority Optimization Targets")
                top = by_prod.head(5)
                for i, (p, wkg) in enumerate(top.items(), 1):
                    prod_df = df[df["product_line"] == p]
                    avg_rate = (prod_df["waste_kg"] / prod_df["production_volume"]).mean() * 100
                    total_cost = prod_df["cost_usd"].sum()
                    potential = total_cost * 0.30
                    
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                    c1.markdown(f"**{i}. {p}** (Product Line)")
                    c2.metric("Waste", f"{wkg/1000:.1f} t")
                    c3.metric("Annual Cost", f"EUR {total_cost/1000:.1f}K")
                    c4.metric("Savings Potential", f"EUR {potential/1000:.1f}K", "30%")

                st.markdown("---")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Waste by Product Line")
                    fig = px.bar(
                        x=by_prod.values[:5],
                        y=by_prod.index[:5],
                        orientation="h",
                        labels={"x": "Waste (kg)", "y": "Product Line"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.subheader("Waste by Root Cause")
                    fig = px.pie(values=by_reason.values, names=by_reason.index)
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Waste Heatmap: Product x Plant")
                pivot = df.pivot_table(values="waste_kg", index="product_line", columns="plant", aggfunc="sum")
                fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error displaying analysis: {e}")

    with tab3:
        st.markdown("### 💰 Savings Simulator")
        if st.session_state.waste_df is None or st.session_state.waste_df.empty:
            st.info("Load or generate data first.")
        else:
            try:
                df = st.session_state.waste_df
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("**Process Improvements**")
                    process_reduction = st.slider("Process optimization (%)", 0, 50, 15, key="sim_proc")
                    equip_upgrade = st.checkbox("Equipment upgrade (-20% changeover waste)", value=True, key="sim_equip")
                    pred_maint = st.checkbox("Predictive maintenance (-30% equipment failures)", value=True, key="sim_pm")
                    st.markdown("**Material Changes**")
                    mat_sub = st.slider("Material substitution savings (%)", 0, 30, 10, key="sim_mat")
                    pkg_redesign = st.checkbox("Packaging redesign (-25% packaging waste)", value=True, key="sim_pkg")
                
                with c2:
                    st.markdown("**Spec & Ops**")
                    spec_tol = st.slider("Spec tolerance adjustment (%)", 0, 20, 8, key="sim_spec")
                    quality_up = st.checkbox("Quality system upgrade (-40% defects)", value=True, key="sim_quality")
                    training = st.checkbox("Enhanced operator training (-15% human error)", value=True, key="sim_train")
                    inventory = st.checkbox("Inventory optimization (-50% expiry waste)", value=True, key="sim_inv")

                if st.button("Calculate Savings", key="sim_calc"):
                    base_cost = df["cost_usd"].sum()
                    savings = 0
                    
                    # Calculate savings
                    savings += base_cost * (process_reduction / 100)
                    if equip_upgrade:
                        savings += base_cost * 0.10
                    if pred_maint:
                        savings += base_cost * 0.08
                    savings += base_cost * (mat_sub / 100)
                    if pkg_redesign:
                        savings += base_cost * 0.12
                    savings += base_cost * (spec_tol / 100)
                    if quality_up:
                        savings += base_cost * 0.15
                    if training:
                        savings += base_cost * 0.05
                    if inventory:
                        savings += base_cost * 0.03

                    reduction_pct = (savings / base_cost) * 100 if base_cost > 0 else 0
                    trace("savings.calc", reduction_pct=reduction_pct)

                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='box'><b>Total Savings</b><h3>EUR {savings/1e6:.2f}M</h3><div>per year</div></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='box'><b>Waste Reduction</b><h3>{reduction_pct:.1f}%</h3></div>", unsafe_allow_html=True)
                    roi = ((savings - 500000) / 500000) * 100 if savings > 500000 else 0
                    c3.markdown(f"<div class='box'><b>ROI (12 mo)</b><h3>{roi:.0f}%</h3></div>", unsafe_allow_html=True)

                    st.markdown("#### 🗺️ Implementation Roadmap")
                    roadmap = pd.DataFrame({
                        "Quarter": ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"],
                        "Initiatives": [
                            "Quick wins: Training, Spec adjustments",
                            "Process optimization, Quality upgrade",
                            "Equipment upgrades, Predictive maintenance",
                            "Material substitution, Packaging redesign"
                        ],
                        "Expected Savings": [
                            f"EUR {savings*0.15/1e6:.1f}M",
                            f"EUR {savings*0.35/1e6:.1f}M",
                            f"EUR {savings*0.30/1e6:.1f}M",
                            f"EUR {savings*0.20/1e6:.1f}M"
                        ],
                        "Cumulative": [
                            f"EUR {savings*0.15/1e6:.1f}M",
                            f"EUR {savings*0.50/1e6:.1f}M",
                            f"EUR {savings*0.80/1e6:.1f}M",
                            f"EUR {savings*1.00/1e6:.1f}M"
                        ]
                    })
                    st.dataframe(roadmap, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error calculating savings: {e}")

    with tab4:
        st.markdown("### 📈 AI-Generated Insights & Anomaly Detection")
        if st.session_state.waste_df is None or st.session_state.waste_df.empty:
            st.info("Load or generate data first.")
        else:
            try:
                df = st.session_state.waste_df.copy()
                
                st.markdown("#### 🔍 Key Findings (demo)")
                st.write("""
1) Seasonal uptick in Q4 (holiday production surge) -> Pre-build inventory in Q3  
2) Pantene line waste ~2.3x Olay -> SMED for changeovers  
3) 68% of waste in first 2h post shift change -> strengthen digital handover  
4) Supplier B material variance -> incoming moisture testing  
5) Replicate Plant B practices to A/C -> ~EUR 1.2M opportunity
""")
                
                # Forecast chart
                future = pd.date_range("2024-01-01", "2024-12-31", freq="M")
                baseline = np.random.uniform(9000, 11000, len(future))
                optimized = baseline * 0.65
                forecast = pd.DataFrame({
                    "Month": future,
                    "Business as Usual": baseline,
                    "With Optimizations": optimized
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast["Month"],
                    y=forecast["Business as Usual"],
                    name="Business as Usual",
                    line=dict(color="red", dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=forecast["Month"],
                    y=forecast["With Optimizations"],
                    name="With Optimizations",
                    line=dict(color="green", width=3)
                ))
                fig.update_layout(
                    title="12-Month Waste Forecast",
                    yaxis_title="Waste (kg)",
                    hovermode="x unified",
                    height=380
                )
                st.plotly_chart(fig, use_container_width=True)

                # Anomaly detection
                st.markdown("#### 🧪 Anomaly Detection")
                if SKLEARN_AVAILABLE:
                    try:
                        df_d = df.copy()
                        df_d["waste_rate"] = df_d["waste_kg"] / df_d["production_volume"]
                        X = df_d[["waste_kg", "production_volume", "cost_usd", "waste_rate"]].fillna(0).values
                        
                        iso = IsolationForest(random_state=42, contamination=CONTAMINATION_RATE)
                        labels = iso.fit_predict(X)
                        df_d["anomaly"] = (labels == -1)
                        
                        outliers = df_d[df_d["anomaly"]].head(20)
                        st.success(f"IsolationForest flagged {df_d['anomaly'].sum()} potential outliers.")
                        st.dataframe(
                            outliers[["date", "product_line", "plant", "waste_kg", "production_volume", "cost_usd", "waste_reason"]],
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning(f"Anomaly detection error: {e}. Using fallback method.")
                        df_sus = df.sort_values("waste_kg", ascending=False).head(20)
                        st.dataframe(
                            df_sus[["date", "product_line", "plant", "waste_kg", "production_volume", "cost_usd", "waste_reason"]],
                            use_container_width=True
                        )
                else:
                    st.warning("scikit-learn not available - using heuristic fallback.")
                    df_sus = df.sort_values("waste_kg", ascending=False).head(20)
                    st.dataframe(
                        df_sus[["date", "product_line", "plant", "waste_kg", "production_volume", "cost_usd", "waste_reason"]],
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Error generating insights: {e}")

# --------------------------------------------------------------------------------
# Supplier Portal (Evidence + Magic Links)
# --------------------------------------------------------------------------------
def _make_tiny_pdf_bytes(title: str) -> bytes:
    """Generate a tiny demo PDF."""
    try:
        safe_title = title.replace("(", "[").replace(")", "]")
        content = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 88 >> stream
BT /F1 24 Tf 72 700 Td (Evidence Preview:) Tj T* ({safe_title}) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref 0 6
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000110 00000 n 
0000000344 00000 n 
0000000495 00000 n 
trailer << /Root 1 0 R /Size 6 >>
startxref
594
%%EOF"""
        return content.encode("latin-1")
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return b""

def page_supplier_portal():
    """Supplier Portal page."""
    st.markdown("## 🤝 Supplier Portal - Evidence & Requests")
    st.caption("Collect SVHC declarations & recycled-content certificates; create magic-link requests.")

    c1, c2 = st.columns([2, 1])
    
    with c1:
        f = st.file_uploader("Upload supplier evidence CSV (or load demo)", type=["csv", "xlsx"], key="sup_upload")
        if st.button("Load Demo Evidence Manifest", key="sup_demo_btn"):
            st.session_state.evidence_rows = gen_evidence_manifest()
            if not st.session_state.evidence_rows.empty:
                st.success("Demo evidence manifest loaded.")
                trace("evidence.demo_loaded", rows=len(st.session_state.evidence_rows))
        
        if f is not None:
            try:
                st.session_state.evidence_rows = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                st.success(f"Loaded {f.name} ({len(st.session_state.evidence_rows)} rows).")
                trace("evidence.upload_loaded", rows=len(st.session_state.evidence_rows), filename=f.name)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                trace("evidence.upload_error", error=str(e))

    with c2:
        st.markdown("**API-Ready Hooks**")
        st.caption("Simulates a vendor portal integration via magic links & health checks.")
        
        if st.button("Test Supplier API Health", key="sup_health_btn"):
            res = SUPPLIER_API.health()
            ok = res.get("ok", False)
            st.markdown(
                f"Service: supplier-api | Base: {res.get('base_url', 'N/A')} | Status: " +
                (f"<span class='pill ok'>OK</span>" if ok else f"<span class='pill err'>ERROR</span>"),
                unsafe_allow_html=True
            )
            trace("supplier_api.health", ok=ok, base=res.get("base_url"))
        
        suppliers = SUPPLIER_API.fetch_suppliers()
        if suppliers:
            sel_sup_for_link = safe_selectbox("Supplier (send magic link)", suppliers, key="sup_for_magic")
            if sel_sup_for_link and st.button("Send Magic Link", key="sup_send_magic_btn"):
                try:
                    resp = SUPPLIER_API.submit_magic_link(sel_sup_for_link)
                    if "error" not in resp:
                        st.success(f"Sent to {resp['supplier']}. Expires in {resp['expires_in_hours']}h.")
                        st.code(resp["link"])
                        trace("magic_link.sent", supplier=resp["supplier"], expires=resp["expires_in_hours"])
                    else:
                        st.error(f"Failed to send magic link: {resp['error']}")
                except Exception as e:
                    st.error(f"Error sending magic link: {e}")

    st.markdown("---")
    
    if st.session_state.evidence_rows is not None and len(st.session_state.evidence_rows) > 0:
        st.subheader("Evidence Inbox")
        st.dataframe(st.session_state.evidence_rows, use_container_width=True)
        
        st.markdown("### 📄 Inline PDF Preview (Demo)")
        evidence_ids = st.session_state.evidence_rows["id"].tolist()
        sel_eid = safe_selectbox("Choose evidence to preview", evidence_ids, key="evidence_preview_sel")
        
        if sel_eid and st.button("Preview Selected", key="evidence_preview_btn"):
            try:
                pdf_bytes = _make_tiny_pdf_bytes(title=f"Evidence {sel_eid}")
                if pdf_bytes:
                    b64 = base64.b64encode(pdf_bytes).decode()
                    st.download_button(
                        "Download selected PDF",
                        data=pdf_bytes,
                        file_name=f"{sel_eid}.pdf",
                        mime="application/pdf"
                    )
                    st.markdown(
                        f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='500' style='border:1px solid #eee;border-radius:8px'></iframe>",
                        unsafe_allow_html=True
                    )
                    trace("evidence.preview", id=sel_eid)
            except Exception as e:
                st.error(f"Error previewing evidence: {e}")
    else:
        st.info("Load or generate the evidence manifest above to review submissions.")

# --------------------------------------------------------------------------------
# Assurance Readiness
# --------------------------------------------------------------------------------
def page_assurance():
    """Assurance Readiness page."""
    st.markdown("## 🛡️ Assurance Readiness")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        f = st.file_uploader("Upload assurance readiness CSV", type=["csv", "xlsx"], key="assure_upload")
        if st.button("Load Demo", key="assure_demo_btn"):
            st.session_state.assurance_df = gen_assurance_tracker()
            if not st.session_state.assurance_df.empty:
                st.success("Demo assurance tracker loaded.")
                trace("assurance.demo_loaded", rows=len(st.session_state.assurance_df))
        
        if f is not None:
            try:
                st.session_state.assurance_df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                st.success(f"Loaded {f.name} ({len(st.session_state.assurance_df)} rows).")
                trace("assurance.upload_loaded", rows=len(st.session_state.assurance_df), filename=f.name)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                trace("assurance.upload_error", error=str(e))
    
    with c2:
        st.markdown("**Summary**")
        if st.session_state.assurance_df is not None and not st.session_state.assurance_df.empty:
            try:
                a = st.session_state.assurance_df
                safe_metric("Controls", len(a))
                ready = (a["audit_ready"] == "Yes").sum()
                safe_metric("Audit-ready", ready)
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")
        else:
            st.caption("Load a tracker to view KPIs.")

    st.markdown("---")
    
    if st.session_state.assurance_df is not None and not st.session_state.assurance_df.empty:
        st.subheader("Controls Tracker")
        st.dataframe(st.session_state.assurance_df, use_container_width=True)
    else:
        st.info("Load a tracker to review controls.")

# --------------------------------------------------------------------------------
# Agent - Evidence Authenticator & Gap-Closer (AI + HMAC)
# --------------------------------------------------------------------------------
def page_agent():
    """Agent: Evidence Authenticator page."""
    st.markdown("## 🤖 Agent: Evidence Authenticator & Gap-Closer")
    st.caption("Scores supplier evidence completeness, generates AI disclosures, and signs them (tamper-evident).")

    c1, c2 = st.columns([2, 1])
    
    with c1:
        f = st.file_uploader("Upload evidence manifest (CSV/Excel) or load demo", type=["csv", "xlsx"], key="agent_ev_upl")
        if st.button("Load Demo Evidence", key="agent_demo_btn"):
            st.session_state.evidence_rows = gen_evidence_manifest()
            if not st.session_state.evidence_rows.empty:
                st.success("Demo evidence loaded.")
                trace("agent.evidence.demo_loaded", rows=len(st.session_state.evidence_rows))
        
        if f is not None:
            try:
                st.session_state.evidence_rows = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                st.success(f"Loaded {f.name} ({len(st.session_state.evidence_rows)} rows).")
                trace("agent.evidence.upload_loaded", rows=len(st.session_state.evidence_rows), filename=f.name)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                trace("agent.evidence.upload_error", error=str(e))

    with c2:
        st.markdown("**Attestation Settings**")
        topic = safe_selectbox("Disclosure Topic", ["Circular Economy", "Pollution", "Water"], key="agent_topic")
        include_actions = st.checkbox("Include numbered action plan", value=True, key="agent_actions")
        att_secret_set = "set" if ATTESTATION_SECRET and ATTESTATION_SECRET != DEFAULT_ATTESTATION_SECRET else "demo"
        st.markdown(
            f"Secret status: <span class='pill {'ok' if att_secret_set=='set' else 'warn'}'>{att_secret_set}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Evidence Completeness")
    
    comp_df = pd.DataFrame()
    if st.session_state.evidence_rows is None or len(st.session_state.evidence_rows) == 0:
        st.info("Load evidence to score completeness.")
    else:
        try:
            comp_df = score_evidence_completeness(st.session_state.evidence_rows)
            if not comp_df.empty:
                st.dataframe(comp_df, use_container_width=True)
                missing_any = comp_df[comp_df["completeness"] < MIN_COMPLETENESS_THRESHOLD]
                safe_metric("Suppliers with gaps", int(missing_any["supplier_name"].nunique()))
                safe_metric("Avg completeness", f"{comp_df['completeness'].mean():.1f}%")
        except Exception as e:
            st.error(f"Error scoring completeness: {e}")

    st.markdown("### 📮 Generate Supplier Requests")
    if st.button("Create Magic Links for Missing Evidence", key="agent_magic_btn"):
        if comp_df.empty:
            st.warning("No evidence loaded.")
        else:
            try:
                req_rows = []
                for _, r in comp_df[comp_df["completeness"] < MIN_COMPLETENESS_THRESHOLD].iterrows():
                    resp = SUPPLIER_API.submit_magic_link(r["supplier_name"])
                    if "error" not in resp:
                        req_rows.append({
                            "supplier_name": r["supplier_name"],
                            "topic": r["topic"],
                            "missing": r["missing"],
                            "request_link": resp["link"],
                            "expires_in_hours": resp["expires_in_hours"]
                        })
                        trace("magic_link.sent", supplier=resp["supplier"], topic=r["topic"], expires=resp["expires_in_hours"])
                
                if req_rows:
                    out = pd.DataFrame(req_rows)
                    st.success(f"Created {len(out)} request links.")
                    st.dataframe(out, use_container_width=True)
                    st.download_button(
                        "Download Requests CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name=f"evidence_requests_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No gaps detected. All required evidence present.")
            except Exception as e:
                st.error(f"Error creating magic links: {e}")

    st.markdown("---")
    st.subheader("AI Disclosure + Attestation (HMAC)")
    
    if st.button("Generate & Sign Disclosure", key="agent_sign_btn"):
        try:
            missing_summary = ""
            if st.session_state.evidence_rows is not None and not comp_df.empty:
                comp_topic = comp_df[comp_df["topic"] == topic] if topic else pd.DataFrame()
                if not comp_topic.empty:
                    missing_list = comp_topic[comp_topic["missing"] != ""]["missing"].unique().tolist()
                    if missing_list:
                        missing_summary = ", ".join(sorted(set(", ".join(missing_list).split(", "))))
            
            prompt = f"""Create an ESRS disclosure for {topic}.
Include current state, risks, supplier evidence status (missing: {missing_summary or 'None'}), and next steps.
Be concise and CFO/assurance friendly."""
            
            if include_actions:
                prompt += "\nAdd a numbered action plan (<=5 items) with owners and due dates."
            
            with st.spinner("Generating disclosure..."):
                disclosure = LLM.generate(prompt)
            
            trace("llm.generate", provider=LLM.provider, ok=bool(disclosure), topic=topic)

            secret = ATTESTATION_SECRET
            signature = sign_text_hmac(disclosure, secret)
            
            if signature:
                st.markdown("#### Disclosure (signed)")
                st.markdown(disclosure)
                st.markdown("#### Signature (HMAC-SHA256)")
                st.code(signature, language="text")
                st.caption("Store this signature alongside the disclosure; any change will invalidate the signature.")
                trace("disclosure.signed", topic=topic, length=len(disclosure))

                bundle = json.dumps({
                    "topic": topic,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "disclosure": disclosure,
                    "signature": signature,
                    "algo": "HMAC-SHA256"
                }, indent=2)
                
                st.download_button(
                    "📥 Download Attested Disclosure (JSON)",
                    data=bundle,
                    file_name=f"attested_{topic}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            else:
                st.error("Failed to generate signature.")
                
        except Exception as e:
            st.error(f"Error generating disclosure: {e}")
            trace("disclosure.error", error=str(e))

    st.markdown("### ✅ Verify a Disclosure Signature (Local)")
    v_text = st.text_area("Paste disclosure text", key="verify_text", height=160)
    v_sig = st.text_input("Paste signature (hex)", key="verify_sig")
    
    if st.button("Verify", key="verify_btn"):
        try:
            ok = verify_text_hmac(v_text, v_sig, ATTESTATION_SECRET)
            if ok:
                st.success("Signature VALID ✅")
            else:
                st.error("Signature INVALID ❌")
            st.caption("In production, verification would use the enterprise key (or public key if you adopt asymmetric signing).")
            trace("disclosure.verify", ok=ok)
        except Exception as e:
            st.error(f"Verification error: {e}")

# --------------------------------------------------------------------------------
# Report Generator
# --------------------------------------------------------------------------------
def page_reports():
    """Report Generator page."""
    st.markdown("## 📄 Report Generator")
    tab1, tab2, tab3 = st.tabs(["📊 Quarterly Report", "🎯 Board Presentation", "📋 Compliance Filing"])

    with tab1:
        st.markdown("### Q4 2024 ESG Performance Report")
        if st.button("Generate Quarterly Report", key="rep_qtr_btn"):
            try:
                time.sleep(0.6)
                report = """# CPG company Europe ESG Performance Report - Q4 2024

**Highlights**
- CSRD compliance: 78% (target 95% Q2'25)
- Fine risk identified: EUR 1.8M (mitigation underway)
- Waste diversion: 92% (exceeded 90% target)
- Water consumption: -12% YoY

**Critical Gaps & Actions**
- Scope 3 supplier data portal (Q1)
- SVHC tracking upgrade (Q2)
- Product water intensity (LCA) (Q2)

*Generated by demo app - replace with automated pipeline in production.*
"""
                st.code(report, language="markdown")
                st.download_button("📥 Download (Markdown)", data=report, file_name="ESG_Q4_2024_Report.md")
                trace("report.quarterly_generated")
            except Exception as e:
                st.error(f"Error generating report: {e}")

    with tab2:
        st.markdown("### Board Presentation Draft")
        if st.button("Generate Board Deck Outline", key="rep_board_btn"):
            try:
                time.sleep(0.5)
                st.markdown("""
- Executive Summary - KPIs, risks, ROI
- CSRD Readiness - trajectory, gaps, mitigation
- Waste Excellence - savings, roadmap
- Supplier Assurance - evidence completeness, risk, magic-link status
- Investment Ask - EUR 750k for Q1 initiatives; 12-18 month ROI
- Next Steps - pilot scope, data connectors, Azure OpenAI enablement
""")
                trace("report.board_outline_generated")
            except Exception as e:
                st.error(f"Error generating outline: {e}")

    with tab3:
        st.markdown("### Annual CSRD Compliance Filing (Draft)")
        if st.button("Generate Filing Skeleton", key="rep_filing_btn"):
            try:
                time.sleep(0.7)
                filing = """CSRD Annual Sustainability Statement 2024 - DRAFT

Sections:
- ESRS 2: General Disclosures
- ESRS E1: Climate
- ESRS E2: Pollution
- ESRS E3: Water and Marine Resources
- ESRS E5: Circular Economy
- ESRS S1: Own Workforce
- ESRS G1: Business Conduct

Appendices:
- Evidence Register (SVHC, Recycled Content, ISO)
- Assurance Controls Map
- Attested Disclosures (HMAC signatures)

*Placeholder text; integrate with source-of-truth systems in production.*
"""
                st.code(filing, language="markdown")
                st.download_button("📥 Download Draft (txt)", data=filing, file_name="CSRD_Filing_2024_DRAFT.txt")
                trace("report.filing_skeleton_generated")
            except Exception as e:
                st.error(f"Error generating filing: {e}")

# --------------------------------------------------------------------------------
# API & Integrations (Demo) - with Request Trace + Webhook Verifier
# --------------------------------------------------------------------------------
def page_api():
    """API & Integrations page."""
    st.markdown("## 🔗 API & Integrations - Demo")
    st.caption("Shows API-readiness: service health checks, env-driven config, a sample OpenAPI contract, request trace, and webhook verification.")

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Service Health")
        if st.button("Check LLM & Supplier API", key="api_health_btn"):
            try:
                l = LLM.health()
                s = SUPPLIER_API.health()
                
                st.markdown(
                    f"LLM: {('<span class=\"pill ok\">OK</span>' if l['ok'] else '<span class=\"pill err\">ERROR</span>')} - provider={LLM.provider}",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"Supplier API: {('<span class=\"pill ok\">OK</span>' if s['ok'] else '<span class=\"pill err\">ERROR</span>')} - base={s.get('base_url', 'N/A')}",
                    unsafe_allow_html=True
                )
                
                trace("llm.health", ok=l["ok"], provider=LLM.provider)
                trace("supplier_api.health", ok=s["ok"], base=s.get("base_url"))
            except Exception as e:
                st.error(f"Health check error: {e}")

        st.markdown("### Environment")
        try:
            env_config = {
                "SUPPLIER_API_BASE": os.getenv("SUPPLIER_API_BASE", DEFAULT_SUPPLIER_API),
                "SUPPLIER_API_KEY": "set" if os.getenv("SUPPLIER_API_KEY") else "not set",
                "LLM_PROVIDER": os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
                "ANTHROPIC_API_KEY": "set" if os.getenv("ANTHROPIC_API_KEY") else "not set",
                "OPENAI_API_KEY": "set" if os.getenv("OPENAI_API_KEY") else "not set",
                "ATTESTATION_SECRET": "set" if (os.getenv("ATTESTATION_SECRET") and os.getenv("ATTESTATION_SECRET") != DEFAULT_ATTESTATION_SECRET) else "demo"
            }
            st.code(json.dumps(env_config, indent=2))
        except Exception as e:
            st.error(f"Error displaying environment: {e}")

    with c2:
        st.markdown("### OpenAPI (sample)")
        try:
            openapi = {
                "openapi": "3.0.0",
                "info": {"title": "Supplier Evidence API", "version": "0.1.0"},
                "paths": {
                    "/suppliers": {
                        "get": {"summary": "List suppliers", "responses": {"200": {"description": "OK"}}}
                    },
                    "/suppliers/{name}/magic-link": {
                        "post": {"summary": "Create evidence portal link", "responses": {"200": {"description": "OK"}}}
                    },
                    "/evidence": {
                        "post": {"summary": "Upload evidence metadata", "responses": {"201": {"description": "Created"}}}
                    },
                },
                "components": {
                    "securitySchemes": {
                        "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
                    }
                },
                "security": [{"bearerAuth": []}]
            }
            st.code(json.dumps(openapi, indent=2), language="json")
        except Exception as e:
            st.error(f"Error displaying OpenAPI: {e}")

    st.info("**Client-facing proof of API-readiness:** Health checks, env-driven config, and a publishable OpenAPI schema. Swap stubs with real HTTP calls when credentials are provided.")

    st.markdown("---")
    st.markdown("### 🧾 Request Trace (demo)")
    
    if st.session_state.trace_log:
        try:
            df_trace = pd.DataFrame(st.session_state.trace_log)
            st.dataframe(df_trace, use_container_width=True)
            
            if st.button("Clear Trace", key="trace_clear_btn"):
                st.session_state.trace_log = []
                st.success("Trace cleared.")
        except Exception as e:
            st.error(f"Error displaying trace: {e}")
    else:
        st.caption("No requests yet - run health checks, create magic links, generate/sign disclosures, or calculate savings.")

    st.markdown("---")
    st.markdown("### 🔐 Webhook Verifier (demo)")
    st.caption("Paste a JSON payload (as sent by your supplier portal webhook) and the X-Signature (hex). We verify with your ATTESTATION_SECRET.")
    
    payload = st.text_area("Incoming JSON payload", key="webhook_payload", height=140)
    sig = st.text_input("X-Signature (hex)", key="webhook_sig")
    
    if st.button("Verify Webhook", key="webhook_verify_btn"):
        try:
            ok = verify_text_hmac(payload.strip(), sig.strip(), ATTESTATION_SECRET)
            if ok:
                st.success("Valid webhook ✅")
            else:
                st.error("Invalid signature ❌")
            trace("webhook.verify", ok=ok, payload_len=len(payload.strip()))
        except Exception as e:
            st.error(f"Verification error: {e}")

# --------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------
def main():
    """Main application router."""
    try:
        show_header()
        route = st.session_state.nav
        
        if route == "Executive Dashboard":
            page_dashboard()
        elif route == "CSRD Compliance Analyzer":
            page_csrd()
        elif route == "Waste Optimizer":
            page_waste()
        elif route == "Supplier Portal":
            page_supplier_portal()
        elif route == "Assurance Readiness":
            page_assurance()
        elif route == "Agent: Evidence Authenticator":
            page_agent()
        elif route == "Report Generator":
            page_reports()
        elif route == "API & Integrations (Demo)":
            page_api()

        st.markdown("---")
        st.caption("CPG company ESG Intelligence Platform - AI Kit A (Demo) | All data simulated for demonstration | Copyright 2025")
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
