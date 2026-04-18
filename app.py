import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import streamlit.components.v1 as components

# Import from custom modules
from eda_charts import (plot_distribution, plot_boxplot, plot_bar_chart,
                        plot_correlation_heatmap, plot_pairplot,
                        plot_missing_heatmap, plot_skewness_kurtosis,
                        plot_count_plot, plot_line_chart, get_insight)
from preprocessing import (handle_missing_values, remove_duplicates,
                           encode_categorical, scale_numerical,
                           detect_column_types, get_dataset_health_score)
from db_connector import (get_mysql_engine, get_postgres_engine,
                          list_tables, load_table, run_custom_query)

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="DataLens — EDA Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── INJECT THIS EXACT CSS (Along with some advanced additions) ──
st.markdown("""<style>
/* Hide Streamlit defaults */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 0rem;}

/* Page background */
.stApp {background-color: #0a0f1e;}
[data-testid="stAppViewContainer"] {background-color: #0a0f1e;}
[data-testid="stHeader"] {background-color: #0a0f1e;}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #00d4ff33;
    padding: 20px;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #00d4ff44;
    border-radius: 16px;
    box-shadow: 0 0 20px #00d4ff15;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 8px;
}
.metric-card:hover {box-shadow: 0 0 30px #00d4ff33; border-color: #7c3aed;}
.metric-value {color: #00d4ff; font-size: 28px; font-weight: 900;}
.metric-label {color: #9ca3af; font-size: 13px; margin-top: 4px;}

/* Gradient title */
.gradient-title {
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 52px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 8px;
}

/* Section headers */
h2 {
    color: white !important;
    border-left: 4px solid #00d4ff;
    padding-left: 12px;
    margin-top: 32px;
}
h3 {color: #e2e8f0 !important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 1px solid #ffffff11;
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    color: #6b7280;
    font-size: 15px;
    font-weight: 500;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: white !important;
    font-weight: 700;
    border-bottom: 2px solid #00d4ff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    transform: scale(1.02) !important;
    box-shadow: 0 0 15px #7c3aed88;
}

/* Dataframes */
.stDataFrame {
    background: #111827 !important;
    border: 1px solid #ffffff11;
    border-radius: 8px;
}

/* Inputs */
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #00d4ff44 !important;
    color: white !important;
}
.stMultiSelect > div > div {
    background: #111827 !important;
    border: 1px solid #00d4ff44 !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #00d4ff55 !important;
    background: #111827 !important;
    border-radius: 12px !important;
}

/* Alerts */
.stSuccess {background: #064e3b !important; border: 1px solid #10b981 !important; color: #34d399 !important;}
.stWarning {background: #78350f !important; border: 1px solid #f59e0b !important; color: #fbbf24 !important;}
.stError {background: #7f1d1d !important; border: 1px solid #ef4444 !important; color: #f87171 !important;}

/* Insight box */
.insight-box {
    background: #0f172a;
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 12px 16px;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 8px;
}

/* Feature cards */
.feature-card {
    background: #111827;
    border: 1px solid #00d4ff33;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
    height: 150px;
    margin-bottom: 16px;
}
.feature-card:hover {
    border-color: #00d4ff;
    box-shadow: 0 0 20px #00d4ff22;
    transform: translateY(-5px);
}
.feature-card h4 {color: white; margin: 8px 0 4px; font-weight: bold;}
.feature-card p {color: #9ca3af; font-size: 13px; margin: 0;}

/* Column type badges */
.badge-num {background:#1d4ed8;color:white;border-radius:20px;
             padding:3px 12px;font-size:12px;font-weight:600;}
.badge-cat {background:#065f46;color:white;border-radius:20px;
             padding:3px 12px;font-size:12px;font-weight:600;}
.badge-dt  {background:#92400e;color:white;border-radius:20px;
             padding:3px 12px;font-size:12px;font-weight:600;}
.badge-txt {background:#7f1d1d;color:white;border-radius:20px;
             padding:3px 12px;font-size:12px;font-weight:600;}

/* Radio buttons */
.stRadio > div {color: #e2e8f0; display: flex; gap: 20px;}
.stCheckbox > label {color: #e2e8f0 !important;}
p, label, .stText {color: #e2e8f0 !important;}

/* Expander */
.streamlit-expanderHeader {
    background: #111827 !important;
    color: white !important;
    border: 1px solid #00d4ff22 !important;
    border-radius: 8px !important;
}

/* Loader animation */
@keyframes glow {
  0% { box-shadow: 0 0 5px #00d4ff, 0 0 10px #00d4ff; }
  50% { box-shadow: 0 0 20px #7c3aed, 0 0 30px #7c3aed; }
  100% { box-shadow: 0 0 5px #00d4ff, 0 0 10px #00d4ff; }
}
.glow-divider {
    height: 3px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7c3aed, transparent);
    margin: 20px 0;
    border-radius: 5px;
    animation: glow 3s infinite;
}
</style>""", unsafe_allow_html=True)

# ── SESSION STATE ──
for key in ['df', 'processed_df', 'filename', 'db_engine', 'data_source', 'refresh_trigger']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'refresh_trigger' else 0

st.markdown('<div class="gradient-title">DataLens</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #9ca3af; margin-bottom: 30px;">Next-Gen EDA & Preprocessing Studio</p>', unsafe_allow_html=True)

# ── DATA SOURCE SELECTION ──
data_col1, data_col2, data_col3 = st.columns([1,2,1])
with data_col2:
    data_source_ui = st.radio("", ["📁 Upload CSV", "🗄️ Connect to SQL Database"], horizontal=True)
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

# ── LOAD DATA LOGIC ──
if data_source_ui == "📁 Upload CSV":
    uploaded_file = st.file_uploader("Drop your CSV dataset here", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.processed_df = df.copy()
            st.session_state.filename = uploaded_file.name
            st.session_state.data_source = 'csv'
            st.success(f"Successfully loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

elif data_source_ui == "🗄️ Connect to SQL Database":
    st.markdown("## 🗄️ SQL Database Connection")
    c1, c2 = st.columns(2)
    with c1:
        db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL"])
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", value=3306 if db_type == "MySQL" else 5432)
        database = st.text_input("Database Name")
    with c2:
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔌 Connect"):
            with st.spinner("Connecting..."):
                if db_type == "MySQL":
                    engine = get_mysql_engine(host, port, user, password, database)
                else:
                    engine = get_postgres_engine(host, port, user, password, database)
                
                if engine is not None:
                    st.success(f"✅ Connected successfully to {database}!")
                    st.session_state.db_engine = engine

    if st.session_state.db_engine is not None:
        st.markdown("### 📋 Load Data")
        tab_browse, tab_custom = st.tabs(["📂 Browse Tables", "⌨️ Custom SQL Query"])
        
        with tab_browse:
            tables = list_tables(st.session_state.db_engine)
            if tables:
                sel_tab1, sel_tab2 = st.columns([3, 1])
                with sel_tab1: selected_table = st.selectbox("Select Table", tables)
                with sel_tab2: row_limit = st.slider("Row Limit", 100, 50000, 5000)
                
                if st.button("📥 Load Table"):
                    with st.spinner("Loading..."):
                        df = load_table(st.session_state.db_engine, selected_table, row_limit)
                        if df is not None:
                            st.session_state.df = df
                            st.session_state.processed_df = df.copy()
                            st.session_state.filename = selected_table
                            st.session_state.data_source = 'sql'
                            st.success(f"Loaded {len(df)} rows from {selected_table}")
                            st.rerun()
            else:
                st.warning("No tables found or error fetching schema.")
                
        with tab_custom:
            query = st.text_area("Enter SQL Query", placeholder="SELECT * FROM table_name LIMIT 1000", height=120)
            if st.button("▶️ Run Query"):
                with st.spinner("Executing..."):
                    df = run_custom_query(st.session_state.db_engine, query)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.processed_df = df.copy()
                        st.session_state.filename = "Custom_SQL_Query"
                        st.session_state.data_source = 'sql'
                        st.success(f"Query returned {len(df)} rows and {len(df.columns)} columns.")
                        st.rerun()

# ── LANDING PAGE (when no data loaded) ──
if st.session_state.df is None:
    st.markdown("""
        <div style="text-align: center; padding: 40px 20px;">
            <h3>Welcome to DataLens!</h3>
            <p style="color: #9ca3af; font-size: 18px;">To begin exploring, please upload a dataset or connect to your SQL database.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Instant EDA</h4>
            <p>Generate beautiful Plotly charts automatically and derive deep insights into your dataset.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🧹 Smart Preprocessing</h4>
            <p>Clean missing values, scale, encode, and detect outliers instantly without any Python code.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🗄️ SQL Integration</h4>
            <p>Direct connectivity to MySQL & PostgreSQL to instantly analyze your production data.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    # ── MAIN DASHBOARD ──
    df = st.session_state.processed_df
    
    # Sidebar Global Actions
    with st.sidebar:
        st.markdown("### 🎛️ Data Controls")
        st.info(f"**Loaded:** {st.session_state.filename}")
        if st.button("↻ Reset to Original Data"):
            st.session_state.processed_df = st.session_state.df.copy()
            st.success("Reset successful!")
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 💾 Export")
        csv_bin = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Processed CSV",
            data=csv_bin,
            file_name=f"processed_{st.session_state.filename}.csv" if st.session_state.filename else "processed_data.csv",
            mime="text/csv"
        )
    
    # Top Metrics
    health_score = get_dataset_health_score(df)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df.isnull().sum().sum():,}</div><div class="metric-label">Missing Cells</div></div>', unsafe_allow_html=True)
    with m4:
        health_color = "#10b981" if health_score >= 80 else "#f59e0b" if health_score >= 50 else "#ef4444"
        st.markdown(f'<div class="metric-card" style="border-color: {health_color}33;"><div class="metric-value" style="color: {health_color};">{health_score}%</div><div class="metric-label">Dataset Health</div></div>', unsafe_allow_html=True)

    # Main Tabs
    tab_overview, tab_cleaning, tab_charts = st.tabs(["📝 Overview & Profiling", "🧹 Preprocessing", "📊 EDA Visualizations"])

    # ──── TAB: OVERVIEW ────
    with tab_overview:
        st.markdown("### Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.markdown("### Column Structure")
        types = detect_column_types(df)
        
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.markdown(f"**Numerical:** <span class='badge-num'>{len(types['numerical'])}</span>", unsafe_allow_html=True)
            for c in types['numerical']: st.markdown(f"- `{c}`")
        with col_c2:
            st.markdown(f"**Categorical:** <span class='badge-cat'>{len(types['categorical'])}</span>", unsafe_allow_html=True)
            for c in types['categorical']: st.markdown(f"- `{c}`")
        with col_c3:
            st.markdown(f"**Datetime:** <span class='badge-dt'>{len(types['datetime'])}</span>", unsafe_allow_html=True)
            for c in types['datetime']: st.markdown(f"- `{c}`")
        with col_c4:
            st.markdown(f"**Text:** <span class='badge-txt'>{len(types['text'])}</span>", unsafe_allow_html=True)
            for c in types['text']: st.markdown(f"- `{c}`")
            
        st.markdown("### Missing Value Summary")
        st.plotly_chart(plot_missing_heatmap(df), use_container_width=True)

    # ──── TAB: PREPROCESSING ────
    with tab_cleaning:
        st.markdown("### 🧹 Data Cleaning Pipeline")
        clean_col1, clean_col2 = st.columns([1, 1])
        
        with clean_col1:
            with st.expander("1. Handle Missing Values", expanded=True):
                missing_cols = df.columns[df.isnull().any()].tolist()
                if missing_cols:
                    sel_missing = st.multiselect("Select columns with missing values", missing_cols, default=missing_cols)
                    method_missing = st.selectbox("Imputation Method", ['drop', 'mean', 'median', 'mode', 'custom'])
                    custom_v = st.text_input("Custom Value") if method_missing == 'custom' else None
                    if st.button("Apply Fill/Drop"):
                        new_df, msg = handle_missing_values(df, sel_missing, method_missing, custom_v)
                        st.session_state.processed_df = new_df
                        st.success(msg)
                        st.rerun()
                else:
                    st.success("No missing values found in the dataset!")
            
            with st.expander("2. Remove Duplicates"):
                st.write(f"Current Duplicates: {df.duplicated().sum()}")
                if st.button("Drop Duplicate Rows"):
                    new_df, msg = remove_duplicates(df)
                    st.session_state.processed_df = new_df
                    st.success(msg)
                    st.rerun()

        with clean_col2:
            with st.expander("3. Encode Categorical Data"):
                cat_cols = types['categorical']
                if cat_cols:
                    sel_enc = st.multiselect("Select Categorical Columns", cat_cols)
                    enc_method = st.radio("Encoding Type", ['label', 'onehot'], horizontal=True)
                    if st.button("Apply Encoding"):
                        new_df, msg = encode_categorical(df, sel_enc, enc_method)
                        st.session_state.processed_df = new_df
                        st.success(msg)
                        st.rerun()
                else:
                    st.info("No pure categorical columns detected.")

            with st.expander("4. Scale Numerical Features"):
                num_cols = types['numerical']
                if num_cols:
                    sel_scale = st.multiselect("Select Numerical Columns", num_cols)
                    scale_method = st.radio("Scaling Algorithm", ['standard', 'minmax'], horizontal=True)
                    if st.button("Apply Scaling"):
                        new_df, msg = scale_numerical(df, sel_scale, scale_method)
                        st.session_state.processed_df = new_df
                        st.success(msg)
                        st.rerun()
                else:
                    st.info("No numerical columns detected.")

    # ──── TAB: EDA CHARTS ────
    with tab_charts:
        st.markdown("### 📈 Interactive Analytics")
        
        num_cols = types['numerical']
        cat_cols = types['categorical']
        
        # Univariate
        st.markdown("#### Univariate Analysis")
        col_uni1, col_uni2 = st.columns(2)
        with col_uni1:
            st.markdown("**Distribution & Boxplot**")
            sel_num1 = st.selectbox("Select Numerical Feature", num_cols if num_cols else ["None"], key='uni_num')
            if sel_num1 != "None":
                st.plotly_chart(plot_distribution(df, sel_num1), use_container_width=True)
                st.markdown(f'<div class="insight-box">{get_insight(df, "distribution", sel_num1)}</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_boxplot(df, sel_num1), use_container_width=True)
                st.markdown(f'<div class="insight-box">{get_insight(df, "boxplot", sel_num1)}</div>', unsafe_allow_html=True)
                
        with col_uni2:
            st.markdown("**Categorical Frequencies**")
            sel_cat1 = st.selectbox("Select Categorical Feature", cat_cols if cat_cols else ["None"], key='uni_cat')
            if sel_cat1 != "None":
                st.plotly_chart(plot_count_plot(df, sel_cat1), use_container_width=True)
                st.markdown(f'<div class="insight-box">{get_insight(df, "count", sel_cat1)}</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_bar_chart(df, sel_cat1), use_container_width=True)
                
        st.markdown("---")
        # Bivariate / Multivariate
        st.markdown("#### Multivariate Analysis")
        multi1, multi2 = st.columns(2)
        with multi1:
            st.markdown("**Correlation Matrix**")
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
            st.markdown(f'<div class="insight-box">{get_insight(df, "correlation")}</div>', unsafe_allow_html=True)
            
        with multi2:
            st.markdown("**Skewness & Kurtosis**")
            st.plotly_chart(plot_skewness_kurtosis(df), use_container_width=True)
            st.markdown(f'<div class="insight-box">{get_insight(df, "skewness")}</div>', unsafe_allow_html=True)

        st.markdown("**Advanced Scatter Plot Matrix**")
        pair_cols = st.multiselect("Select columns for pairplot (2-4 recommended)", df.columns.tolist(), default=num_cols[:3] if len(num_cols) >= 3 else num_cols)
        pair_color = st.selectbox("Color By (optional)", ["None"] + cat_cols)
        if len(pair_cols) > 1:
            st.plotly_chart(plot_pairplot(df, pair_cols, color_col=None if pair_color == "None" else pair_color), use_container_width=True)
            st.markdown(f'<div class="insight-box">{get_insight(df, "pairplot")}</div>', unsafe_allow_html=True)
        else:
            st.info("Select at least 2 columns to view pairplot.")

        if types['datetime']:
            st.markdown("---")
            st.markdown("#### Timeline Analysis")
            date_col = st.selectbox("Date Column", types['datetime'])
            val_col = st.selectbox("Value to track", num_cols if num_cols else ["None"])
            if val_col != "None":
                st.plotly_chart(plot_line_chart(df, date_col, val_col), use_container_width=True)
                st.markdown(f'<div class="insight-box">{get_insight(df, "line", val_col)}</div>', unsafe_allow_html=True)
