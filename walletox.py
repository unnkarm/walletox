import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Walletox",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DYNAMIC BLACK & WHITE CSS
# ============================================
st.markdown("""
<style>
    /* Dark Mode Base */
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }

    /* Dynamic Headers */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -1px;
        line-height: 0.9;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .main-header:hover {
        letter-spacing: 2px;
        color: #888888;
    }

    /* Sub-headers */
    .sub-header {
        font-size: 0.9rem;
        color: #888888;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 3rem;
    }

    /* Metric Cards with Glassmorphism */
    [data-testid="stMetric"] {
        background: #111111;
        border: 1px solid #333333;
        padding: 20px;
        border-radius: 0px;
        transition: all 0.4s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #ffffff;
        transform: translateY(-5px);
        background: #1a1a1a;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] {
        color: #888888 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.8rem !important;
    }

    /* Buttons - High Contrast Dynamic */
    .stButton>button {
        width: 100%;
        border-radius: 0px !important;
        border: 2px solid #ffffff !important;
        background-color: transparent !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 1rem;
        transition: all 0.3s !important;
    }
    .stButton>button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        box-shadow: 0 0 20px rgba(255,255,255,0.4);
    }

    /* Tables & Dataframes */
    .stDataFrame {
        border: 1px solid #333333;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #222222;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #ffffff;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Input Fields */
    input, select, textarea {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #888888 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #111111;
        border: 1px solid #333333;
        color: #888888;
        border-radius: 0px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #000000;
        border-color: #ffffff;
    }
    
    /* Success/Error boxes */
    .stSuccess, .stError, .stWarning, .stInfo {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
        color: #ffffff !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Headers in content */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Download button */
    .stDownloadButton>button {
        border-radius: 0px !important;
        border: 1px solid #333333 !important;
        background-color: #111111 !important;
        color: #ffffff !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stDownloadButton>button:hover {
        border-color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #111111;
        border: 1px solid #333333;
        border-radius: 0px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INDIAN CURRENCY FORMATTER
# ============================================
def format_inr(number):
    if number == 0:
        return "₹0.00"
    s, temp = f"{float(number):.2f}".split('.')
    last_three = s[-3:]
    other = s[:-3]
    if other != '':
        last_three = ',' + last_three
    while len(other) > 2:
        last_three = ',' + other[-2:] + last_three
        other = other[:-2]
    return "₹" + other + last_three + '.' + temp

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'expenses' not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=['date', 'category', 'amount', 'note', 'recurring'])
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'budget_goals' not in st.session_state:
    st.session_state.budget_goals = {}

DEFAULT_CATEGORIES = [
    'Food & Dining', 'Transportation', 'Shopping', 'Entertainment',
    'Bills & Utilities', 'Healthcare', 'Education', 'Savings & Investment',
    'Personal Care', 'Groceries', 'Rent/EMI', 'Other'
]

# ============================================
# CORE CALCULATIONS
# ============================================
def calculate_summary(df, income):
    total = df['amount'].sum() if not df.empty else 0
    recurring = df[df.get('recurring', False) == True]['amount'].sum() if not df.empty else 0
    remaining = income - total
    rate = (remaining / income * 100) if income > 0 else 0
    return {
        'total': total,
        'remaining': remaining,
        'rate': max(0, rate),
        'recurring': recurring,
        'avg_daily': df['amount'].mean() if not df.empty else 0
    }

def prepare_data(df):
    if df.empty:
        return df
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.to_period('M')
    return df

# ============================================
# ML MODELS
# ============================================
class ExpensePredictionModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def train_and_predict(self, expenses_df):
        # Fallback: if data is too thin for regression, use weighted averages
        if expenses_df.empty:
            return None, None
            
        df = expenses_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        monthly = df.groupby(df['date'].dt.to_period('M'))['amount'].sum().reset_index()
        
        # CATEGORY PREDICTIONS (Heuristic-based fallback)
        cat_pred = {}
        for cat in df['category'].unique():
            cat_data = df[df['category'] == cat]
            # Use last 2 months avg if possible, else total avg
            cat_monthly = cat_data.groupby(df['date'].dt.to_period('M'))['amount'].sum()
            cat_pred[cat] = cat_monthly.tail(2).mean() if len(cat_monthly) >= 2 else cat_monthly.mean()

        # TOTAL PREDICTION
        if len(monthly) >= 2:
            monthly['month_num'] = range(len(monthly))
            X = monthly[['month_num']].values
            y = monthly['amount'].values
            self.model.fit(X, y)
            total_pred = self.model.predict([[len(monthly)]])[0]
        else:
            # Simple heuristic if only one month exists
            total_pred = sum(cat_pred.values())
        
        return max(0, total_pred), cat_pred

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
    
    def detect_anomalies(self, df):
        if df.empty or len(df) < 10:
            return []
        
        anomalies = []
        amounts = df['amount'].values.reshape(-1, 1)
        self.model.fit(amounts)
        predictions = self.model.predict(amounts)
        
        for idx in np.where(predictions == -1)[0]:
            row = df.iloc[idx]
            anomalies.append({
                'date': row['date'],
                'category': row['category'],
                'amount': row['amount'],
                'note': row['note']
            })
        
        return anomalies[:5]

# ============================================
# VISUALIZATIONS
# ============================================
def create_gauge(current, budget):
    percent = min((current / budget) * 100 if budget > 0 else 0, 100)
    color = "#00FF00" if percent < 70 else "#FFA500" if percent < 90 else "#FF0000"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': "BUDGET USAGE", 'font': {'size': 18, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#ffffff"},
            'bar': {'color': color},
            'bgcolor': "#111111",
            'borderwidth': 2,
            'bordercolor': "#333333",
            'steps': [
                {'range': [0, 70], 'color': "#1a1a1a"},
                {'range': [70, 90], 'color': "#222222"},
                {'range': [90, 100], 'color': "#2a2a2a"}
            ]
        },
        number={'font': {'color': '#ffffff'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=280
    )
    return fig

def create_waterfall(df, income):
    cats = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    labels = ["INCOME"] + list(cats.index) + ["NET SAVINGS"]
    values = [income] + [-amt for amt in cats.values] + [income - sum(cats.values)]
    measures = ["absolute"] + ["relative"] * len(cats) + ["total"]
    
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector={"line": {"color": "#333333"}},
        decreasing={"marker": {"color": "#FF4444"}},
        increasing={"marker": {"color": "#00FF00"}},
        totals={"marker": {"color": "#00FFCC"}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'showgrid': False},
        yaxis={'showgrid': True, 'gridcolor': '#222222'},
        height=400
    )
    return fig

def create_heatmap(df):
    df = prepare_data(df)
    pivot = df.pivot_table(values='amount', index='weekday', columns='category', aggfunc='sum', fill_value=0)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex([d for d in days if d in pivot.index])
    
    fig = px.imshow(
        pivot,
        labels=dict(x="CATEGORY", y="DAY", color="AMOUNT (₹)"),
        color_continuous_scale=[[0, '#000000'], [0.5, '#00FFCC'], [1, '#ffffff']],
        aspect="auto"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400
    )
    return fig

# ============================================
# DASHBOARD VIEW
# ============================================
def show_dashboard():
    st.markdown('<p class="main-header">OVERVIEW</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Financial Intelligence Dashboard</p>', unsafe_allow_html=True)

    df = st.session_state.expenses
    income = st.session_state.user_profile['monthly_income']
    summary = calculate_summary(df, income)

    # Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("TOTAL SPENT", format_inr(summary['total']))
    with c2:
        st.metric("AVAILABLE", format_inr(summary['remaining']))
    with c3:
        st.metric("SAVINGS RATE", f"{summary['rate']:.1f}%")
    with c4:
        st.metric("RECURRING", format_inr(summary['recurring']))

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visualizations
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### CATEGORY SPLIT")
        if not df.empty:
            fig_pie = px.pie(
                df, values='amount', names='category',
                hole=0.7,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="white",
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True, theme=None)
        else:
            st.write("NO DATA AVAILABLE")

    with col_b:
        st.markdown("### SPENDING TRAJECTORY")
        if not df.empty:
            df_plot = df.copy()
            df_plot['date'] = pd.to_datetime(df_plot['date'])
            daily = df_plot.groupby('date')['amount'].sum().reset_index()
            
            fig_line = px.line(daily, x='date', y='amount', markers=True)
            fig_line.update_traces(
                line_color='#00FFCC',
                line_width=3,
                marker=dict(size=10, color='white', line=dict(width=2, color='#00FFCC'))
            )
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="white",
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=True, gridcolor='#222222', color='white')
            )
            st.plotly_chart(fig_line, use_container_width=True, theme=None)
        else:
            st.write("NO DATA AVAILABLE")
    
    # Additional Charts
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### BUDGET USAGE")
        st.plotly_chart(create_gauge(summary['total'], income), use_container_width=True, theme=None)
    
    with col2:
        st.markdown("### CASH FLOW")
        if not df.empty:
            st.plotly_chart(create_waterfall(df, income), use_container_width=True, theme=None)
        else:
            st.write("NO DATA AVAILABLE")
    
    # Heatmap
    st.markdown("---")
    st.markdown("### SPENDING HEATMAP")
    if not df.empty and len(df) >= 7:
        st.plotly_chart(create_heatmap(df), use_container_width=True, theme=None)
    else:
        st.write("NEED MORE DATA FOR HEATMAP (MIN 7 TRANSACTIONS)")

# ============================================
# TRANSACTIONS VIEW
# ============================================
def show_transactions():
    st.markdown('<p class="main-header">INPUT</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transaction Management System</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["MANUAL ENTRY", "CSV IMPORT"])
    
    with tab1:
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                date = st.date_input("DATE", datetime.now())
                cat = st.selectbox("CATEGORY", st.session_state.user_profile['categories'])
            with c2:
                amt = st.number_input("AMOUNT (INR)", min_value=0.0, step=10.0)
                note = st.text_input("MEMO")
            
            recurring = st.checkbox("RECURRING EXPENSE")
            
            if st.button("COMMIT TRANSACTION"):
                new_data = pd.DataFrame([{
                    'date': date,
                    'category': cat,
                    'amount': amt,
                    'note': note,
                    'recurring': recurring
                }])
                st.session_state.expenses = pd.concat([st.session_state.expenses, new_data], ignore_index=True)
                st.success("TRANSACTION VERIFIED")
                st.rerun()
        
        st.markdown("### RECENT LOGS")
        if not st.session_state.expenses.empty:
            display = st.session_state.expenses.tail(15).copy()
            display['date'] = pd.to_datetime(display['date']).dt.strftime('%Y-%m-%d')
            display['amount'] = display['amount'].apply(format_inr)
            display['recurring'] = display['recurring'].apply(lambda x: '✓' if x else '')
            st.dataframe(display[['date', 'category', 'amount', 'note', 'recurring']], use_container_width=True, hide_index=True)
        else:
            st.write("NO TRANSACTIONS RECORDED")
    
    with tab2:
        st.markdown("### CSV UPLOAD SYSTEM")
        
        st.download_button(
            "DOWNLOAD TEMPLATE",
            "date,category,amount,note,recurring\n2024-01-15,Food & Dining,450,Lunch,False",
            "template.csv",
            "text/csv"
        )
        
        uploaded = st.file_uploader("UPLOAD CSV FILE", type=['csv'])
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                df['date'] = pd.to_datetime(df['date'])
                df['amount'] = pd.to_numeric(df['amount'])
                
                if 'note' not in df.columns:
                    df['note'] = ''
                if 'recurring' not in df.columns:
                    df['recurring'] = False
                
                st.success(f"VALIDATED {len(df)} TRANSACTIONS")
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("IMPORT ALL TRANSACTIONS"):
                    st.session_state.expenses = pd.concat([st.session_state.expenses, df], ignore_index=True)
                    st.success(f"IMPORTED {len(df)} TRANSACTIONS")
                    st.rerun()
            except Exception as e:
                st.error(f"ERROR: {str(e)}")

# ============================================
# BUDGET PLANNER
# ============================================
def show_budget_planner():
    st.markdown('<p class="main-header">BUDGET</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Category Budget Allocation</p>', unsafe_allow_html=True)
    
    income = st.session_state.user_profile['monthly_income']
    categories = st.session_state.user_profile['categories']
    
    cols = st.columns(3)
    total_budget = 0
    
    for i, cat in enumerate(categories):
        with cols[i % 3]:
            default = st.session_state.budget_goals.get(cat, int(income * 0.1))
            limit = st.number_input(cat.upper(), min_value=0, value=default, key=f"bud_{cat}")
            st.session_state.budget_goals[cat] = limit
            total_budget += limit
            
            current = st.session_state.expenses[st.session_state.expenses['category'] == cat]['amount'].sum()
            pct = min(current / limit, 1.0) if limit > 0 else 0
            
            st.progress(pct)
            st.caption(f"{format_inr(current)} / {format_inr(limit)}")
    
    st.markdown("---")
    
    if total_budget > income:
        st.warning(f"⚠️ BUDGET OVERFLOW: {format_inr(total_budget)} > {format_inr(income)}")
    else:
        st.success(f"✓ UNALLOCATED: {format_inr(income - total_budget)}")

# ============================================
# ML INSIGHTS
# ============================================
def show_ml_insights():
    st.markdown('<p class="main-header">INTELLIGENCE</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Analysis</p>', unsafe_allow_html=True)
    
    df = st.session_state.expenses
    
    # Check for minimum data threshold
    if df.empty:
        st.markdown("""
            <div style="border: 1px solid #333; padding: 3rem; text-align: center; background: #0a0a0a;">
                <h2 style="color: #888; letter-spacing: 5px;">SYSTEM IDLE</h2>
                <p style="color: #444;">AWAITING TRANSACTION LOGS TO INITIALIZE ANALYTICS ENGINE.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    tab1, tab2 = st.tabs(["PREDICTIONS", "ANOMALIES"])
    
    with tab1:
        st.markdown("### NEXT MONTH FORECAST")
        
        @st.cache_resource
        def get_predictor():
            return ExpensePredictionModel()

        predictor = get_predictor()
        total, by_cat = predictor.train_and_predict(df)

        if total is not None:
            income = st.session_state.user_profile.get('monthly_income', 0)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("PREDICTED SPEND", format_inr(total))
                
                # Dynamic Status Box
                if total > income:
                    st.markdown(f"""
                        <div style="border: 2px solid #FF0000; padding: 15px; background: #110000;">
                            <p style="color: #FF0000; font-weight: 900; margin:0;">⚠️ OVER BUDGET</p>
                            <p style="color: white; font-size: 0.8rem; margin:0;">DEFICIT: {format_inr(total - income)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="border: 2px solid #00FFCC; padding: 15px; background: #00110a;">
                            <p style="color: #00FFCC; font-weight: 900; margin:0;">✓ SAVINGS TARGET</p>
                            <p style="color: white; font-size: 0.8rem; margin:0;">SURPLUS: {format_inr(income - total)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.caption("Forecast based on historical spending velocity and recurring patterns.")

            with col2:
                # Prepare forecasting dataframe
                pred_df = pd.DataFrame(by_cat.items(), columns=["Category", "Amount"]).sort_values("Amount", ascending=False)

                # Vibrant chart to contrast B&W theme
                fig = px.bar(
                    pred_df, x="Amount", y="Category", orientation="h",
                    color="Amount", color_continuous_scale="Viridis", text_auto='.2s'
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"}, showlegend=False, height=350,
                    xaxis={"showgrid": True, "gridcolor": "#222222", "title": ""},
                    yaxis={"showgrid": False, "title": ""}, coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

    with tab2:
        st.markdown("### ANOMALY DETECTION")
        detector = AnomalyDetector()
        anomalies = detector.detect_anomalies(df)
        
        if anomalies:
            for a in anomalies:
                st.markdown(f"""
                    <div style="border-left: 4px solid #FF0000; background: #111; padding: 10px; margin-bottom: 10px;">
                        <span style="color: #888;">{a['date']}</span> | <b>{a['category']}</b> | <b style="color: #FF0000;">{format_inr(a['amount'])}</b>
                        <br><small style="color: #555;">{a['note'] if a['note'] else 'No description'}</small>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✓ NO IRREGULAR SPENDING DETECTED IN CURRENT LOGS")
# ============================================
# AI CHAT
# ============================================
def show_ai_chat():
    st.markdown('<p class="main-header">ASSISTANT</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI Financial Consultant</p>', unsafe_allow_html=True)
    
    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Suggested Queries
    if not st.session_state.chat_history:
        st.write("**SUGGESTED QUERIES:**")
        suggestions = ["WEEKLY BREAKDOWN", "HIGHEST EXPENSE", "SAVINGS ADVICE", "CATEGORY TRENDS"]
        cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            if cols[i].button(sug, key=f"sug_{i}"):
                process_ai_query(sug)
    
    # Chat Input
    if prompt := st.chat_input("Ask about your finances (e.g., 'How much on Food?')"):
        process_ai_query(prompt)
    
    if st.session_state.chat_history and st.button("CLEAR HISTORY"):
        st.session_state.chat_history = []
        st.rerun()

def process_ai_query(query):
    """Enhanced Logic Engine for Higher Accuracy Responses"""
    df = st.session_state.expenses.copy()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    
    profile = st.session_state.user_profile
    query_upper = query.upper()
    response = ""

    if df.empty:
        response = "I have no transaction data to analyze yet. Please add some expenses first."
    
    # 1. SPECIFIC CATEGORY SEARCH (e.g., "How much on Food?")
    elif any(cat.upper() in query_upper for cat in st.session_state.user_profile.get('categories', [])):
        matched_cat = next(cat for cat in st.session_state.user_profile['categories'] if cat.upper() in query_upper)
        cat_total = df[df['category'] == matched_cat]['amount'].sum()
        response = f"SYSTEM ANALYSIS for **{matched_cat}**:\n\nTotal spent in this category: {format_inr(cat_total)}\nNumber of transactions: {len(df[df['category'] == matched_cat])}"

    # 2. TEMPORAL ANALYSIS (e.g., "Last 7 days")
    elif "WEEK" in query_upper or "7 DAYS" in query_upper:
        week_ago = datetime.now() - timedelta(days=7)
        recent_df = df[df['date'] >= week_ago]
        total = recent_df['amount'].sum()
        response = f"WEEKLY REPORT (Last 7 Days):\n\nYou spent {format_inr(total)} across {len(recent_df)} transactions."

    # 3. HIGHEST EXPENSE SEARCH
    elif "HIGHEST" in query_upper or "MAX" in query_upper or "TOP" in query_upper:
        max_row = df.loc[df['amount'].idxmax()]
        response = f"Your HIGHEST EXPENSE was {format_inr(max_row['amount'])} for '{max_row['note']}' in the {max_row['category']} category on {max_row['date'].strftime('%Y-%m-%d')}."

    # 4. GENERAL SUMMARY (Fall-back)
    else:
        summary = calculate_summary(df, profile['monthly_income'])
        top_cat = df.groupby('category')['amount'].sum().sort_values(ascending=False).index[0]
        response = (f"GENERAL INTELLIGENCE REPORT:\n\n"
                    f"• Total Spent: {format_inr(summary['total'])}\n"
                    f"• Current Savings Rate: {summary['rate']:.1f}%\n"
                    f"• Most expensive category: {top_cat}\n\n"
                    "Tip: You can ask me about specific categories or recent time periods!")

    st.session_state.chat_history.extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": response}
    ])
    st.rerun()
# ============================================
# SETTINGS
# ============================================
def show_settings():
    st.markdown('<p class="main-header">SETTINGS</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">System Configuration</p>', unsafe_allow_html=True)
    
    income = st.number_input(
        "MONTHLY INCOME (INR)",
        value=st.session_state.user_profile['monthly_income'],
        step=1000.0
    )
    
    user_type = st.selectbox(
        "USER PROFILE",
        ['Student', 'Salaried', 'Self-employed', 'Business Owner'],
        index=['Student', 'Salaried', 'Self-employed', 'Business Owner'].index(
            st.session_state.user_profile.get('user_type', 'Salaried')
        )
    )
    
    if st.button("UPDATE PROFILE"):
        st.session_state.user_profile['monthly_income'] = income
        st.session_state.user_profile['user_type'] = user_type
        st.success("PROFILE UPDATED")
        st.rerun()
    
    st.markdown("---")
    
    st.download_button(
    label="EXPORT ALL DATA (CSV)",
    data=st.session_state.expenses.to_csv(index=False) if not st.session_state.expenses.empty else "",
    file_name="finance_export.csv",
    mime="text/csv",
    disabled=st.session_state.expenses.empty,
    key="export-data"
)

    
    st.markdown("---")
    if st.button("RESET SYSTEM"):
        st.session_state.user_profile = None
        st.session_state.expenses = pd.DataFrame(columns=['date', 'category', 'amount', 'note', 'recurring'])
        st.session_state.chat_history = []
        st.session_state.budget_goals = {}
        st.warning("SYSTEM WIPED")
        st.rerun()

# ============================================
# ONBOARDING SYSTEM
# ============================================
def show_onboarding():
    # Hero Section
    st.markdown('<p class="main-header" style="font-size: 8rem; margin-bottom: 0;">WALLETOX</p>', unsafe_allow_html=True)
    st.markdown("""
        <p style="font-size: 1.2rem; color: #888; letter-spacing: 5px; text-transform: uppercase; margin-top: -10px; margin-bottom: 3rem;">
            Precision Financial Intelligence & Asset Tracking
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Professional Settings Form
    with st.container():
        st.markdown("### INITIAL CONFIGURATION")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p style="color: #888; font-size: 0.8rem; letter-spacing: 1px;">PRIMARY ARCHETYPE</p>', unsafe_allow_html=True)
            u_type = st.selectbox(
                "Select your financial profile", 
                ['Student', 'Salaried', 'Self-employed', 'Business Owner'],
                label_visibility="collapsed"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="color: #888; font-size: 0.8rem; letter-spacing: 1px;">MONTHLY CAPITAL (INR)</p>', unsafe_allow_html=True)
            income = st.number_input(
                "Enter monthly income", 
                min_value=0.0, 
                value=50000.0, 
                step=1000.0,
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("""
                <div style="border-left: 1px solid #333; padding-left: 2rem; height: 100%;">
                    <p style="color: #ffffff; font-weight: 700; margin-bottom: 0.5rem;">CONFIGURATION SPECS</p>
                    <ul style="color: #888; font-size: 0.9rem; list-style-type: none; padding-left: 0;">
                        <li>• Real-time INR Numbering Logic</li>
                        <li>• Machine Learning Forecast Engine</li>
                        <li>• Anomaly Detection Algorithms</li>
                        <li>• Pure Black minimalist UI</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Center-aligned button
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("BOOT SYSTEM"):
                st.session_state.user_profile = {
                    'monthly_income': income,
                    'user_type': u_type,
                    'categories': DEFAULT_CATEGORIES
                }
                st.success("SYSTEM READY. REDIRECTING...")
                st.rerun()

# ============================================
# MAIN APP ENTRY
# ============================================
def main():
    if not st.session_state.user_profile:
        show_onboarding()
    else:
        # Sidebar Navigation
        with st.sidebar:
            st.markdown(f"## {st.session_state.user_profile['user_type'].upper()}")
            st.markdown(f"**INCOME:** {format_inr(st.session_state.user_profile['monthly_income'])}")
            st.markdown("---")
            
            nav = st.radio(
                "NAVIGATION",
                ["DASHBOARD", "INPUT", "BUDGET", "INTELLIGENCE", "ASSISTANT", "SETTINGS"]
            )
            
            st.markdown("---")
            st.caption("SYSTEM STATUS: OPTIMIZED")
            st.caption(f"LOGS: {len(st.session_state.expenses)} ENTRIES")

        # Route to views
        if nav == "DASHBOARD":
            show_dashboard()
        elif nav == "INPUT":
            show_transactions()
        elif nav == "BUDGET":
            show_budget_planner()
        elif nav == "INTELLIGENCE":
            show_ml_insights()
        elif nav == "ASSISTANT":
            show_ai_chat()
        elif nav == "SETTINGS":
            show_settings()

if __name__ == "__main__":
    main()