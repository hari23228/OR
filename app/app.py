import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.load_data import load_all_raw_models, MODEL_NAME_MAPPING
from preprocessing.clean_data import clean_and_validate_records
from analysis.statistics import compute_model_statistics
from analysis.visualization import (
    plot_quality_by_model,
    plot_cost_by_model,
    plot_token_usage_by_model,
    plot_quality_vs_cost,
    COLOR_PALETTE
)
from prediction.capability_predictor import train_capability_predictor, MODEL_COST_ORDER
from optimization.integer_programming import solve_integer_programming_routing
from optimization.goal_programming import solve_goal_programming_routing
from optimization.sensitivity_analysis import run_sensitivity_analysis
from evaluation.comparison import run_full_method_comparison, plot_method_comparison_charts
from evaluation.metrics import calculate_cost_savings

# Page Configuration
st.set_page_config(
    page_title="Cost-Aware LLM Model Routing System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Base styling */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Title styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    
    /* Card containers */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .kpi-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #38BDF8;
        margin-top: 0.3rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_dataset():
    csv_path = os.path.join(BASE_DIR, 'data', 'processed', 'llm_routing_dataset.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raw = load_all_raw_models(os.path.join(BASE_DIR, 'Data'))
        cleaned = clean_and_validate_records(raw)
        rows = []
        for records in cleaned.values():
            rows.extend(records)
        df = pd.DataFrame(rows)
        return df

@st.cache_resource
def get_cached_predictor(df):
    clf, metrics, cap_df = train_capability_predictor(df)
    return clf, metrics, cap_df

# Header
st.markdown('<div class="main-header">⚡ Cost-Aware LLM Model Routing System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An Optimization-Based Approach for Cost, Quality, and Latency-Aware LLM Selection (MMLU-Pro Benchmark)</div>', unsafe_allow_html=True)

# Load dataset
df = get_dataset()
clf, predictor_metrics, cap_df = get_cached_predictor(df)
stats_df = compute_model_statistics(df)

# Sidebar
st.sidebar.title("🎛️ System Controls")
st.sidebar.markdown("---")

# Section 25.1: Dataset Upload / Selection
st.sidebar.subheader("1. Dataset Selection")
dataset_option = st.sidebar.radio(
    "Choose Dataset Source:",
    ["Default MMLU-Pro Benchmark (3,000 Queries)", "Upload Custom JSON/CSV"]
)

if dataset_option == "Upload Custom JSON/CSV":
    uploaded_file = st.sidebar.file_uploader("Upload dataset file", type=['csv', 'json'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                import json
                data = json.load(uploaded_file)
                df = pd.DataFrame(data.get('records', []))
            st.sidebar.success(f"Loaded {len(df)} rows from uploaded file!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

st.sidebar.markdown("---")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Statistics & Data Overview",
    "⚙️ Optimization & Routing Engine",
    "📈 Evaluation & Baselines",
    "🔬 Sensitivity Analysis"
])

# ==========================================
# TAB 1: STATISTICS DASHBOARD (Section 25.2)
# ==========================================
with tab1:
    st.header("📊 Dataset & Model Performance Statistics")
    
    num_queries = df['query_id'].nunique()
    num_models = df['model'].nunique()
    overall_avg_quality = stats_df['Average Quality'].mean() * 100.0
    overall_total_cost = stats_df['Total Cost ($)'].sum()
    cheapest_cost = stats_df['Total Cost ($)'].min()
    strongest_cost = stats_df.loc[stats_df['Accuracy (%)'].idxmax()]['Total Cost ($)']
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">Total Queries</div>
                <div class="kpi-value">{num_queries:,}</div>
                <div class="kpi-sub">Benchmark records</div>
            </div>
        ''', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">Evaluated LLMs</div>
                <div class="kpi-value">{num_models}</div>
                <div class="kpi-sub">Flash to GPT-5</div>
            </div>
        ''', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">Avg Model Accuracy</div>
                <div class="kpi-value">{overall_avg_quality:.1f}%</div>
                <div class="kpi-sub">Across all models</div>
            </div>
        ''', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">Cheapest Model Cost</div>
                <div class="kpi-value">${cheapest_cost:.2f}</div>
                <div class="kpi-sub">DeepSeek V3</div>
            </div>
        ''', unsafe_allow_html=True)
    with kpi5:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">Strongest Model Cost</div>
                <div class="kpi-value">${strongest_cost:.2f}</div>
                <div class="kpi-sub">GPT-5</div>
            </div>
        ''', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Grid
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_quality_by_model(stats_df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_cost_by_model(stats_df), use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_token_usage_by_model(stats_df), use_container_width=True)
    with col4:
        st.plotly_chart(plot_quality_vs_cost(stats_df), use_container_width=True)

    with st.expander("📋 Detailed Model Summary Table"):
        st.dataframe(
            stats_df[['Model', 'Accuracy (%)', 'Total Cost ($)', 'Average Cost ($)', 'Avg Prompt Tokens', 'Avg Completion Tokens', 'Avg Total Tokens']],
            use_container_width=True
        )

# ==========================================
# TAB 2: OPTIMIZATION SETTINGS & ROUTING (Section 25.3 & 25.4)
# ==========================================
with tab2:
    st.header("⚙️ Optimization Settings & Optimal LLM Routing")
    
    opt_col1, opt_col2 = st.columns([1, 2])
    
    with opt_col1:
        st.subheader("1. Select Algorithm")
        algo = st.radio(
            "Optimization Engine:",
            ["Goal Programming (Multi-Objective)", "Integer Programming (PuLP ILP)"]
        )
        
        st.markdown("---")
        if algo == "Integer Programming (PuLP ILP)":
            st.subheader("ILP Constraints")
            min_quality_input = st.slider(
                "Minimum Quality Requirement (Q_min):",
                min_value=0.75, max_value=0.98, value=0.85, step=0.01,
                help="Ensures every query receives a model satisfying minimum quality"
            )
            use_predictor = st.checkbox("Use ML Capability Predictor per query", value=True)
        else:
            st.subheader("Goal Programming Weights")
            st.write("Configure importance weights ($w_1 d_{cost} + w_2 d_{quality} + w_3 d_{latency}$):")
            w_quality = st.slider("Quality Priority (w_quality):", 0.0, 1.0, 0.50, 0.05)
            w_cost = st.slider("Cost Priority (w_cost):", 0.0, 1.0, 0.30, 0.05)
            w_latency = st.slider("Latency Priority (w_latency):", 0.0, 1.0, 0.20, 0.05)

    with opt_col2:
        st.subheader("2. Optimization Execution Results")
        
        if algo == "Integer Programming (PuLP ILP)":
            pred_cap_input = cap_df if use_predictor else None
            results_df, summary = solve_integer_programming_routing(
                df, min_quality=min_quality_input, predicted_capabilities=pred_cap_input
            )
        else:
            results_df, summary = solve_goal_programming_routing(
                df, w_cost=w_cost, w_quality=w_quality, w_latency=w_latency
            )

        # Baseline comparison stats
        strongest_cost_val = stats_df.loc[stats_df['Model'] == 'GPT-5']['Total Cost ($)'].values[0]
        cost_savings_pct = calculate_cost_savings(strongest_cost_val, summary['total_cost'])

        res_kpi1, res_kpi2, res_kpi3, res_kpi4 = st.columns(4)
        with res_kpi1:
            st.metric("Total Optimal Cost", f"${summary['total_cost']:.2f}")
        with res_kpi2:
            st.metric("Average Quality / Accuracy", f"{summary['accuracy_pct']:.2f}%")
        with res_kpi3:
            st.metric("Cost Savings vs GPT-5", f"{cost_savings_pct:.2f}%")
        with res_kpi4:
            st.metric("Avg Tokens / Query", f"{summary['avg_tokens']:.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Distribution Chart
        dist_df = pd.DataFrame(list(summary['model_distribution'].items()), columns=['Model', 'Selected Queries'])
        fig_dist = px.pie(
            dist_df, values='Selected Queries', names='Model',
            title='Optimal Routing Distribution Across Models',
            color='Model', color_discrete_map=COLOR_PALETTE,
            hole=0.4
        )
        fig_dist.update_layout(template='plotly_dark')
        st.plotly_chart(fig_dist, use_container_width=True)

    with st.expander("🔍 Query-by-Query Optimal Assignment Table"):
        st.dataframe(results_df.head(100), use_container_width=True)

# ==========================================
# TAB 3: EVALUATION & BASELINES (Section 19, 20, 21, 22)
# ==========================================
with tab3:
    st.header("📈 Baseline Evaluation & Proof of Effectiveness")
    st.markdown("Comparing **Baseline 1 (Cheapest Model)**, **Baseline 2 (Strongest Model)**, **Integer Programming Router**, and **Goal Programming Router**.")
    
    comp_df, raw_summaries = run_full_method_comparison(df)
    fig_cost, fig_tradeoff = plot_method_comparison_charts(comp_df)

    st.dataframe(
        comp_df.style.format({
            'Accuracy (%)': '{:.2f}%',
            'Total Cost ($)': '${:.2f}',
            'Average Cost ($)': '${:.5f}',
            'Avg Tokens': '{:.1f}',
            'Cost Savings vs Strongest (%)': '{:.2f}%',
            'Quality Retention vs Strongest (%)': '{:.2f}%'
        }),
        use_container_width=True
    )

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.plotly_chart(fig_cost, use_container_width=True)
    with col_e2:
        st.plotly_chart(fig_tradeoff, use_container_width=True)

# ==========================================
# TAB 4: SENSITIVITY ANALYSIS (Section 24)
# ==========================================
with tab4:
    st.header("🔬 Sensitivity Analysis")
    st.markdown("Studying how optimal routing decisions change under different quality constraints and goal priorities.")
    
    sens_df = run_sensitivity_analysis(df, cap_df)
    
    st.dataframe(
        sens_df.style.format({
            'Total Cost ($)': '${:.2f}',
            'Accuracy (%)': '{:.2f}%',
            'Avg Tokens': '{:.1f}'
        }),
        use_container_width=True
    )
    
    fig_sens = px.line(
        sens_df,
        x='Configuration',
        y='Total Cost ($)',
        color='Method',
        markers=True,
        title='Sensitivity of Total Routing Cost across Configurations'
    )
    fig_sens.update_layout(template='plotly_dark')
    st.plotly_chart(fig_sens, use_container_width=True)
