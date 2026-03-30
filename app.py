import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Support Intelligence Hub",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DESIGN SYSTEM — PROFESSIONAL DARK EXECUTIVE UI
# =========================================================
COLORS = {
    "bg_1": "#07111F",
    "bg_2": "#0B1728",
    "bg_3": "#0F1E33",
    "surface": "rgba(15, 23, 42, 0.88)",
    "surface_2": "rgba(17, 28, 48, 0.94)",
    "card": "rgba(16, 24, 40, 0.92)",
    "border": "rgba(148, 163, 184, 0.18)",
    "text": "#EAF2FF",
    "muted": "#9FB3C8",
    "primary": "#60A5FA",
    "secondary": "#34D399",
    "accent": "#FBBF24",
    "danger": "#F87171",
    "purple": "#A78BFA",
    "cyan": "#22D3EE",
    "white": "#FFFFFF",
    "grid": "rgba(148, 163, 184, 0.12)",
    "shadow": "rgba(0,0,0,0.38)",
}

TEXT = COLORS["text"]
MUTED = COLORS["muted"]

GOOGLE_DRIVE_CSV_URL = "https://drive.google.com/uc?export=download&id=1OnAYFCYHDm1f09GIleMVdw_WbJHXIeAa"

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {TEXT};
    }}

    .stApp {{
        background:
            radial-gradient(circle at 10% 12%, rgba(96,165,250,0.16), transparent 18%),
            radial-gradient(circle at 88% 14%, rgba(52,211,153,0.12), transparent 16%),
            radial-gradient(circle at 50% 100%, rgba(167,139,250,0.12), transparent 22%),
            linear-gradient(135deg, {COLORS["bg_1"]} 0%, {COLORS["bg_2"]} 45%, {COLORS["bg_3"]} 100%);
        color: {TEXT};
    }}

    .block-container {{
        padding-top: 1.15rem;
        padding-bottom: 2rem;
    }}

    [data-testid="stSidebar"] {{
        background:
            linear-gradient(180deg, rgba(7,17,31,0.98) 0%, rgba(11,23,40,0.98) 100%);
        border-right: 1px solid {COLORS["border"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}

    [data-baseweb="radio"] * {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid {COLORS["border"]};
        border-radius: 30px;
        padding: 1.65rem 1.65rem 1.45rem 1.65rem;
        box-shadow:
            0 16px 40px {COLORS["shadow"]},
            inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(14px);
        margin-bottom: 1rem;
    }}

    .hero::before {{
        content: "";
        position: absolute;
        right: -40px;
        top: -40px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(96,165,250,0.20), transparent 60%);
        border-radius: 50%;
    }}

    .hero::after {{
        content: "";
        position: absolute;
        left: 22%;
        bottom: -65px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(52,211,153,0.10), transparent 60%);
        border-radius: 50%;
    }}

    .hero-badge {{
        display: inline-block;
        background: linear-gradient(135deg, rgba(96,165,250,0.18), rgba(167,139,250,0.10));
        color: {COLORS["white"]};
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.44rem 0.86rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 0.85rem;
        letter-spacing: 0.2px;
    }}

    .hero-title {{
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.08;
        color: {COLORS["white"]};
        margin-bottom: 0.45rem;
    }}

    .hero-subtitle {{
        font-size: 1.02rem;
        line-height: 1.8;
        max-width: 930px;
        color: {MUTED};
    }}

    .section-title {{
        font-size: 1.35rem;
        font-weight: 800;
        color: {COLORS["white"]};
        margin-top: 0.35rem;
        margin-bottom: 0.9rem;
    }}

    .glass-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid {COLORS["border"]};
        border-radius: 24px;
        padding: 1rem 1.08rem;
        box-shadow:
            0 12px 28px {COLORS["shadow"]},
            inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        color: {TEXT};
    }}

    .glass-card-static {{
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid {COLORS["border"]};
        border-radius: 24px;
        padding: 1rem 1.08rem;
        box-shadow:
            0 12px 28px {COLORS["shadow"]},
            inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        color: {TEXT};
        transform: none !important;
        transition: none !important;
        animation: none !important;
    }}

    .metric-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid {COLORS["border"]};
        border-radius: 24px;
        padding: 1.08rem 1.08rem;
        box-shadow:
            0 12px 28px {COLORS["shadow"]},
            inset 0 1px 0 rgba(255,255,255,0.03);
        min-height: 152px;
        transition: 0.22s ease;
        backdrop-filter: blur(12px);
    }}

    .metric-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(96,165,250,0.28);
        box-shadow:
            0 16px 34px {COLORS["shadow"]},
            0 0 0 1px rgba(96,165,250,0.06);
    }}

    .metric-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.65rem;
    }}

    .metric-label {{
        font-size: 0.92rem;
        color: {MUTED};
        font-weight: 700;
    }}

    .metric-icon {{
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(96,165,250,0.16), rgba(255,255,255,0.04));
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 1.1rem;
    }}

    .metric-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {COLORS["white"]};
        margin-bottom: 0.2rem;
    }}

    .metric-note {{
        font-size: 0.88rem;
        color: {MUTED};
        line-height: 1.55;
    }}

    .insight {{
        border-radius: 22px;
        padding: 1rem;
        border: 1px solid {COLORS["border"]};
        box-shadow:
            0 10px 22px {COLORS["shadow"]},
            inset 0 1px 0 rgba(255,255,255,0.03);
        min-height: 170px;
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
        backdrop-filter: blur(10px);
    }}

    .insight.info {{ border-left: 6px solid {COLORS["primary"]}; }}
    .insight.success {{ border-left: 6px solid {COLORS["secondary"]}; }}
    .insight.warn {{ border-left: 6px solid {COLORS["accent"]}; }}
    .insight.danger {{ border-left: 6px solid {COLORS["danger"]}; }}

    .insight-title {{
        font-size: 1rem;
        font-weight: 800;
        color: {COLORS["white"]};
        margin-bottom: 0.45rem;
    }}

    .insight-body {{
        font-size: 0.96rem;
        line-height: 1.68;
        color: {TEXT};
    }}

    .pill {{
        display: inline-block;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(96,165,250,0.16), rgba(251,191,36,0.10));
        color: {COLORS["white"]};
        border: 1px solid {COLORS["border"]};
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }}

    .stButton > button {{
        width: 100%;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        color: {COLORS["white"]};
        background: linear-gradient(135deg, rgba(96,165,250,0.18), rgba(167,139,250,0.14));
        font-weight: 800;
        padding: 0.82rem 1rem;
        box-shadow: 0 10px 20px {COLORS["shadow"]};
        transition: all 0.20s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        border-color: rgba(96,165,250,0.28);
        box-shadow: 0 14px 24px {COLORS["shadow"]};
    }}

    .stTextArea textarea,
    .stTextInput input,
    div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.05) !important;
        color: {COLORS["white"]} !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 16px !important;
    }}

    .stMultiSelect [data-baseweb="tag"] {{
        background: rgba(96,165,250,0.14) !important;
        color: {COLORS["white"]} !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
    }}

    .stSlider p, .stMarkdown, .stCaption, label {{
        color: {COLORS["white"]} !important;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        overflow: hidden;
        background: rgba(255,255,255,0.03);
    }}

    div[data-testid="stDataFrame"] * {{
        color: {COLORS["white"]} !important;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
    }}

    hr {{
        border-color: rgba(255,255,255,0.08);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# PLOTLY STYLE HELPER
# =========================================================
def style_fig(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        title_font=dict(size=20, color=COLORS["white"]),
        legend=dict(
            bgcolor="rgba(7,17,31,0.86)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color=TEXT),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=30, r=20, t=70, b=35),
        xaxis=dict(
            title_font=dict(color=TEXT),
            tickfont=dict(color=TEXT),
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title_font=dict(color=TEXT),
            tickfont=dict(color=TEXT),
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
    )
    return fig

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)

    df["ticket_created_date"] = pd.to_datetime(df["ticket_created_date"], errors="coerce")
    df["ticket_resolved_date"] = pd.to_datetime(df["ticket_resolved_date"], errors="coerce")
    df["browser"] = df["browser"].fillna("Unknown")

    def categorize_satisfaction(score):
        if score >= 4:
            return "Satisfied"
        elif score == 3:
            return "Neutral"
        return "Unsatisfied"

    df["satisfaction_category"] = df["customer_satisfaction_score"].apply(categorize_satisfaction)
    df["escalated_flag"] = df["escalated"].map({"Yes": 1, "No": 0}).fillna(0)
    df["sla_breached_flag"] = df["sla_breached"].map({"Yes": 1, "No": 0}).fillna(0)
    df["ticket_duration_days"] = (df["ticket_resolved_date"] - df["ticket_created_date"]).dt.days.fillna(0)
    df["created_month"] = df["ticket_created_date"].dt.month_name()
    df["created_year"] = df["ticket_created_date"].dt.year

    df["response_time_group"] = pd.cut(
        df["first_response_time_hours"],
        bins=[0, 6, 24, 48, 1000],
        labels=["Very Fast", "Fast", "Moderate", "Slow"],
        include_lowest=True,
    )

    df["resolution_time_group"] = pd.cut(
        df["resolution_time_hours"],
        bins=[0, 24, 72, 168, 5000],
        labels=["Quick", "Normal", "Slow", "Very Slow"],
        include_lowest=True,
    )

    df["age_group"] = pd.cut(
        df["customer_age"],
        bins=[0, 25, 35, 50, 100],
        labels=["18-25", "26-35", "36-50", "50+"],
        include_lowest=True,
    )

    df["tenure_group"] = pd.cut(
        df["customer_tenure_months"],
        bins=[0, 6, 24, 60, 120],
        labels=["New", "Growing", "Loyal", "Very Loyal"],
        include_lowest=True,
    )

    df["is_repeat_customer"] = (df["previous_tickets"] > 5).astype(int)
    df["high_complexity"] = (df["issue_complexity_score"] >= 7).astype(int)
    df["long_resolution"] = (df["resolution_time_hours"] > 100).astype(int)

    return df

@st.cache_resource
def build_recommender(issue_series: pd.Series):
    tfidf = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.9)
    X_issue = tfidf.fit_transform(issue_series.fillna(""))
    return tfidf, X_issue

def recommend_business_aware(df: pd.DataFrame, tfidf_issue, X_issue, issue_text: str, preferred_categories=None, top_n: int = 5):
    query_vec = tfidf_issue.transform([issue_text])
    similarities = cosine_similarity(query_vec, X_issue).flatten()

    temp = df[
        [
            "ticket_id",
            "category",
            "issue_description",
            "resolution_notes",
            "priority",
            "status",
            "product",
            "channel",
            "customer_segment",
        ]
    ].copy()

    temp["similarity_score"] = similarities
    temp = temp[temp["status"].isin(["Resolved", "Closed"])]

    if preferred_categories:
        temp = temp[temp["category"].isin(preferred_categories)]

    temp = temp.drop_duplicates(subset=["issue_description", "resolution_notes"])
    temp = temp.sort_values(by="similarity_score", ascending=False)
    return temp.head(top_n)

def metric_card(icon, label, value, note):
    return f"""
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-label">{label}</div>
            <div class="metric-icon">{icon}</div>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """

def insight_card(title, body, kind="info"):
    return f"""
    <div class="insight {kind}">
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
    </div>
    """

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## ✨ Control Center")
st.sidebar.caption("Executive dashboard with recommendation support.")

file_path = st.sidebar.text_input("Dataset URL / Path", value=GOOGLE_DRIVE_CSV_URL)
load_btn = st.sidebar.button("🚀 Launch Hub")

page = st.sidebar.radio(
    "📌 Navigate",
    ["Executive Dashboard", "Deep-Dive Analytics", "Recommendation Assistant", "Action Plan"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Business Goals")
st.sidebar.markdown(
    """
    - Improve customer satisfaction  
    - Reduce resolution time  
    - Surface risky support patterns  
    - Reuse solved-ticket knowledge  
    - Help owners act faster  
    """
)

st.sidebar.markdown("### 🎨 Visual Identity")
st.sidebar.markdown(
    """
    - Premium dark executive design  
    - Strong contrast and readability  
    - Professional cards and clean charts  
    - Stable recommendation page  
    - Cohesive corporate color palette  
    """
)

if not load_btn and "df_loaded" not in st.session_state:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">Executive • Premium • Professional</div>
            <div class="hero-title">Smart Support Intelligence Hub ✨</div>
            <div class="hero-subtitle">
                A polished dark support analytics app designed to help owners see performance clearly,
                act faster, and guide support teams with reusable solutions from past tickets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Click Launch Hub to load the dataset.")
    st.stop()

if load_btn or "df_loaded" in st.session_state:
    if load_btn:
        with st.spinner("Loading data and preparing the recommendation engine..."):
            df = load_data(file_path)
            tfidf_issue, X_issue = build_recommender(df["issue_description"])
            st.session_state["df_loaded"] = df
            st.session_state["tfidf_issue"] = tfidf_issue
            st.session_state["X_issue"] = X_issue
    else:
        df = st.session_state["df_loaded"]
        tfidf_issue = st.session_state["tfidf_issue"]
        X_issue = st.session_state["X_issue"]

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Creative Premium Dashboard</div>
        <div class="hero-title">Smart Support Intelligence Hub ✨</div>
        <div class="hero-subtitle">
            Turn support data into faster action, clearer executive insight, and smarter team decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FILTERS
# =========================================================
st.markdown("<div class='section-title'>🎛 Smart Filters</div>", unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)
with f1:
    selected_region = st.multiselect("🌍 Region", sorted(df["region"].dropna().unique()))
with f2:
    selected_channel = st.multiselect("💬 Channel", sorted(df["channel"].dropna().unique()))
with f3:
    selected_product = st.multiselect("🧩 Product", sorted(df["product"].dropna().unique()))
with f4:
    selected_segment = st.multiselect("👥 Customer Segment", sorted(df["customer_segment"].dropna().unique()))

filtered_df = df.copy()
if selected_region:
    filtered_df = filtered_df[filtered_df["region"].isin(selected_region)]
if selected_channel:
    filtered_df = filtered_df[filtered_df["channel"].isin(selected_channel)]
if selected_product:
    filtered_df = filtered_df[filtered_df["product"].isin(selected_product)]
if selected_segment:
    filtered_df = filtered_df[filtered_df["customer_segment"].isin(selected_segment)]

if filtered_df.empty:
    st.warning("No records match the selected filters. Please adjust the filters.")
    st.stop()

# =========================================================
# KPI CALCULATIONS
# =========================================================
total_tickets = len(filtered_df)
satisfied_pct = filtered_df["satisfaction_category"].eq("Satisfied").mean() * 100
unsatisfied_pct = filtered_df["satisfaction_category"].eq("Unsatisfied").mean() * 100
avg_response = filtered_df["first_response_time_hours"].mean()
avg_resolution = filtered_df["resolution_time_hours"].mean()
sla_breach_pct = filtered_df["sla_breached_flag"].mean() * 100
repeat_pct = filtered_df["is_repeat_customer"].mean() * 100
long_resolution_pct = filtered_df["long_resolution"].mean() * 100
high_complexity_pct = filtered_df["high_complexity"].mean() * 100

# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================
if page == "Executive Dashboard":
    st.markdown("<div class='section-title'>📊 Executive Snapshot</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(metric_card("🎫", "Total Tickets", f"{total_tickets:,}", "Current support workload after all active filters."), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card("😊", "Satisfied Customers", f"{satisfied_pct:.1f}%", "Customers reporting a positive experience."), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card("⚠️", "Unsatisfied Customers", f"{unsatisfied_pct:.1f}%", "Customers that may affect churn and brand trust."), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card("⏱", "Avg Resolution Time", f"{avg_resolution:.1f}h", "Average time needed to solve a ticket."), unsafe_allow_html=True)

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        st.markdown(metric_card("⚡", "Avg First Response", f"{avg_response:.1f}h", "How fast the team first reacts to customers."), unsafe_allow_html=True)
    with k6:
        st.markdown(metric_card("🚨", "SLA Breach Rate", f"{sla_breach_pct:.1f}%", "Tickets missing the promised service threshold."), unsafe_allow_html=True)
    with k7:
        st.markdown(metric_card("🔁", "Repeat Customers", f"{repeat_pct:.1f}%", "Customers returning with repeated support demand."), unsafe_allow_html=True)
    with k8:
        st.markdown(metric_card("🧠", "High Complexity", f"{high_complexity_pct:.1f}%", "Cases likely needing expert intervention."), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📈 Experience & Speed</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.05, 1])

    with c1:
        sat_counts = filtered_df["satisfaction_category"].value_counts().reset_index()
        sat_counts.columns = ["satisfaction_category", "count"]

        fig_sat = px.pie(
            sat_counts,
            names="satisfaction_category",
            values="count",
            hole=0.60,
            title="Customer Satisfaction Mix",
            color="satisfaction_category",
            color_discrete_map={
                "Satisfied": COLORS["secondary"],
                "Neutral": COLORS["accent"],
                "Unsatisfied": COLORS["danger"],
            },
        )
        fig_sat.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(color=COLORS["white"], size=13),
            marker=dict(line=dict(color="rgba(255,255,255,0.10)", width=2)),
        )
        fig_sat = style_fig(fig_sat)
        st.plotly_chart(fig_sat, use_container_width=True)

    with c2:
        time_df = pd.DataFrame(
            {
                "Metric": ["First Response Time", "Resolution Time"],
                "Hours": [avg_response, avg_resolution],
            }
        )
        fig_time = px.bar(
            time_df,
            x="Metric",
            y="Hours",
            text_auto=".1f",
            title="Support Speed Overview",
            color="Metric",
            color_discrete_map={
                "First Response Time": COLORS["primary"],
                "Resolution Time": COLORS["purple"],
            },
        )
        fig_time = style_fig(fig_time)
        fig_time.update_layout(showlegend=False)
        fig_time.update_traces(marker_line_color="rgba(255,255,255,0.08)", marker_line_width=1.2)
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("<div class='section-title'>🧭 Executive Reading</div>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(insight_card("Customer Risk", f"<b>{unsatisfied_pct:.1f}%</b> of customers are unsatisfied. This is the main area likely to damage retention, reviews, and trust if not improved quickly.", "danger"), unsafe_allow_html=True)
    with i2:
        st.markdown(insight_card("Process Friction", f"<b>{long_resolution_pct:.1f}%</b> of tickets are taking too long. That often signals routing issues, rework, or poor knowledge reuse.", "warn"), unsafe_allow_html=True)
    with i3:
        st.markdown(insight_card("Retention Opportunity", f"<b>{repeat_pct:.1f}%</b> of customers are repeat customers. Improving their journey can directly support loyalty and recurring value.", "success"), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🏆 Top Pressure Areas</div>", unsafe_allow_html=True)
    top_cat = filtered_df.groupby("category")["ticket_id"].count().sort_values(ascending=False).head(8).reset_index(name="count")
    top_prod = filtered_df.groupby("product")["ticket_id"].count().sort_values(ascending=False).head(8).reset_index(name="count")

    p1, p2 = st.columns(2)
    with p1:
        fig_cat = px.bar(
            top_cat,
            x="count",
            y="category",
            orientation="h",
            title="Top Ticket Categories",
            text="count",
            color="count",
            color_continuous_scale=["#132A44", "#245A92", COLORS["primary"]],
        )
        fig_cat = style_fig(fig_cat)
        fig_cat.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_colorbar=dict(title="Count"))
        st.plotly_chart(fig_cat, use_container_width=True)

    with p2:
        fig_prod = px.bar(
            top_prod,
            x="count",
            y="product",
            orientation="h",
            title="Top Products Generating Tickets",
            text="count",
            color="count",
            color_continuous_scale=["#12352D", "#1B7A63", COLORS["secondary"]],
        )
        fig_prod = style_fig(fig_prod)
        fig_prod.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_colorbar=dict(title="Count"))
        st.plotly_chart(fig_prod, use_container_width=True)

    st.markdown("<div class='section-title'>💡 Executive Summary</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glass-card">
            <span class="pill">Customer Satisfaction</span>
            <span class="pill">Resolution Efficiency</span>
            <span class="pill">Service Quality</span>
            <span class="pill">Knowledge Reuse</span>
            <br><br>
            <span style="font-size:1.02rem; color:{TEXT};">
                The biggest opportunity is to <b>reduce long-resolution tickets</b>, <b>standardize how frequent issues are handled</b>,
                and <b>reuse proven solutions</b>. That creates faster service, stronger consistency, and a better customer experience.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# DEEP-DIVE ANALYTICS
# =========================================================
elif page == "Deep-Dive Analytics":
    st.markdown("<div class='section-title'>🔎 Deep-Dive Analytics</div>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        region_sat = pd.crosstab(filtered_df["region"], filtered_df["satisfaction_category"], normalize="index") * 100
        region_sat = region_sat.reset_index().melt(id_vars="region", var_name="satisfaction", value_name="percent")
        fig_region = px.bar(
            region_sat,
            x="region",
            y="percent",
            color="satisfaction",
            barmode="group",
            title="Satisfaction by Region",
            color_discrete_map={"Satisfied": COLORS["secondary"], "Neutral": COLORS["accent"], "Unsatisfied": COLORS["danger"]},
        )
        fig_region = style_fig(fig_region)
        st.plotly_chart(fig_region, use_container_width=True)

    with d2:
        channel_sat = pd.crosstab(filtered_df["channel"], filtered_df["satisfaction_category"], normalize="index") * 100
        channel_sat = channel_sat.reset_index().melt(id_vars="channel", var_name="satisfaction", value_name="percent")
        fig_channel = px.bar(
            channel_sat,
            x="channel",
            y="percent",
            color="satisfaction",
            barmode="group",
            title="Satisfaction by Channel",
            color_discrete_map={"Satisfied": COLORS["secondary"], "Neutral": COLORS["accent"], "Unsatisfied": COLORS["danger"]},
        )
        fig_channel = style_fig(fig_channel)
        st.plotly_chart(fig_channel, use_container_width=True)

    d3, d4 = st.columns(2)
    with d3:
        res_group = filtered_df["resolution_time_group"].value_counts(dropna=False).reset_index()
        res_group.columns = ["resolution_time_group", "count"]
        fig_res_group = px.funnel(
            res_group,
            x="count",
            y="resolution_time_group",
            title="Resolution Speed Funnel",
            color="resolution_time_group",
            color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["danger"]],
        )
        fig_res_group = style_fig(fig_res_group)
        st.plotly_chart(fig_res_group, use_container_width=True)

    with d4:
        monthly = filtered_df.groupby(filtered_df["ticket_created_date"].dt.to_period("M")).size().reset_index(name="tickets")
        monthly["ticket_created_date"] = monthly["ticket_created_date"].astype(str)
        fig_monthly = px.line(
            monthly,
            x="ticket_created_date",
            y="tickets",
            markers=True,
            title="Ticket Volume Trend Over Time",
        )
        fig_monthly.update_traces(line=dict(color=COLORS["primary"], width=4), marker=dict(size=8, color=COLORS["accent"]))
        fig_monthly = style_fig(fig_monthly)
        st.plotly_chart(fig_monthly, use_container_width=True)

    d5, d6 = st.columns(2)
    with d5:
        seg_time = filtered_df.groupby("customer_segment")[["first_response_time_hours", "resolution_time_hours"]].mean().reset_index()
        fig_seg = go.Figure()
        fig_seg.add_trace(go.Bar(name="First Response", x=seg_time["customer_segment"], y=seg_time["first_response_time_hours"], marker_color=COLORS["primary"]))
        fig_seg.add_trace(go.Bar(name="Resolution", x=seg_time["customer_segment"], y=seg_time["resolution_time_hours"], marker_color=COLORS["purple"]))
        fig_seg.update_layout(title="Average Time by Customer Segment", barmode="group")
        fig_seg = style_fig(fig_seg)
        st.plotly_chart(fig_seg, use_container_width=True)

    with d6:
        priority_sla = pd.crosstab(filtered_df["priority"], filtered_df["sla_breached"], normalize="index") * 100
        priority_sla = priority_sla.reset_index().melt(id_vars="priority", var_name="sla_breached", value_name="percent")
        fig_sla = px.bar(
            priority_sla,
            x="priority",
            y="percent",
            color="sla_breached",
            barmode="group",
            title="SLA Breach by Priority",
            color_discrete_map={"No": COLORS["secondary"], "Yes": COLORS["danger"]},
        )
        fig_sla = style_fig(fig_sla)
        st.plotly_chart(fig_sla, use_container_width=True)

    st.markdown(insight_card("What this page tells the owner", "This view reveals where service pressure, dissatisfaction, and slow handling are concentrated so management can target the right region, channel, or segment first.", "info"), unsafe_allow_html=True)

# =========================================================
# RECOMMENDATION ASSISTANT
# =========================================================
elif page == "Recommendation Assistant":
    st.markdown("<div class='section-title'>🤖 Recommendation Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card-static">
            Type a new customer issue and let the system find the most similar solved tickets and reusable solution patterns.
            <br><br>
            <span class="pill">Faster Search</span>
            <span class="pill">Smarter Reuse</span>
            <span class="pill">More Consistent Support</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.45, 1])

    with col_a:
        user_issue = st.text_area("📝 Describe the customer issue", value="payment deducted but service not activated", height=150)

    with col_b:
        category_options = sorted(df["category"].dropna().unique())
        default_categories = [c for c in ["Payment Problem", "Refund Request", "Subscription Cancellation"] if c in category_options]
        selected_categories = st.multiselect("🎯 Preferred Categories", category_options, default=default_categories)
        top_n = st.slider("🔢 Number of Similar Tickets", 3, 10, 5)

    b1, b2, b3 = st.columns(3)
    search_clicked = b1.button("🔍 Find Similar Solutions")
    clear_clicked = b2.button("🧹 Clear Results")
    demo_clicked = b3.button("✨ Load Demo Query")

    if demo_clicked:
        user_issue = "payment deducted but service not activated"

    if clear_clicked:
        st.session_state.pop("rec_results", None)

    if search_clicked and user_issue.strip():
        with st.spinner("Searching the solved-ticket knowledge base..."):
            rec_results = recommend_business_aware(
                df=df,
                tfidf_issue=tfidf_issue,
                X_issue=X_issue,
                issue_text=user_issue,
                preferred_categories=selected_categories if selected_categories else None,
                top_n=top_n,
            )
            st.session_state["rec_results"] = rec_results

    if "rec_results" in st.session_state:
        rec_results = st.session_state["rec_results"]

        if not rec_results.empty:
            best = rec_results.iloc[0]

            st.markdown("<div class='section-title'>🏅 Best Recommended Solution</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="insight success" style="animation:none; transition:none; transform:none;">
                    <div class="insight-title">Top Match</div>
                    <div class="insight-body">
                        <b>Suggested Category:</b> {best['category']}<br>
                        <b>Priority Reference:</b> {best['priority']}<br>
                        <b>Status of Reference Ticket:</b> {best['status']}<br>
                        <b>Similarity Score:</b> {best['similarity_score']:.3f}<br><br>
                        <b>Recommended Resolution Notes:</b><br>
                        {best['resolution_notes']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div class='section-title'>📚 Similar Solved Tickets</div>", unsafe_allow_html=True)
            display_cols = [
                "ticket_id",
                "category",
                "priority",
                "status",
                "product",
                "channel",
                "customer_segment",
                "similarity_score",
                "issue_description",
                "resolution_notes",
            ]
            st.dataframe(rec_results[display_cols], use_container_width=True, height=430, hide_index=True)

            st.markdown(
                insight_card(
                    "How this helps the business owner",
                    "This feature reduces search time, improves handling consistency across agents, and lets the team reuse proven ticket resolutions instead of solving repeated issues from scratch.",
                    "success",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.warning("No relevant solved tickets found. Try editing the issue text or removing category filters.")

# =========================================================
# ACTION PLAN
# =========================================================
elif page == "Action Plan":
    st.markdown("<div class='section-title'>🚀 Business Action Plan</div>", unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(insight_card("Immediate Actions (0–30 days)", "1) Route long-resolution tickets to senior agents.<br>2) Prioritize repeat customers for faster handling.<br>3) Use recommendation results as a live knowledge assistant during support.", "danger"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(insight_card("Mid-Term Actions (1–3 months)", "1) Build a formal knowledge base for repeated issues.<br>2) Create response playbooks for top categories.<br>3) Track weekly KPIs: satisfaction, resolution time, SLA breach, and repeat-customer risk.", "info"), unsafe_allow_html=True)

    with a2:
        st.markdown(insight_card("Long-Term Actions (3+ months)", "1) Add agent-level quality scoring.<br>2) Capture richer feedback signals like empathy and first-contact resolution.<br>3) Introduce AI-assisted draft replies for high-volume cases.", "success"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(insight_card("What the owner gains", "Higher satisfaction, lower churn risk, stronger consistency across agents, less wasted effort, and clearer visibility into service weaknesses.", "warn"), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📌 Recommended KPI Targets</div>", unsafe_allow_html=True)
    target_df = pd.DataFrame(
        {
            "KPI": [
                "Unsatisfied Customer Rate",
                "Average Resolution Time",
                "SLA Breach Rate",
                "Long-Resolution Ticket Rate",
                "Repeat Customer Complaint Rate",
            ],
            "Current": [
                f"{unsatisfied_pct:.1f}%",
                f"{avg_resolution:.1f}h",
                f"{sla_breach_pct:.1f}%",
                f"{long_resolution_pct:.1f}%",
                f"{repeat_pct:.1f}%",
            ],
            "Suggested Target": ["< 30%", "< 96h", "< 35%", "< 40%", "< 55%"],
        }
    )
    st.dataframe(target_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class="glass-card">
            <b>Final Message for the Business Owner:</b><br>
            <span style="font-size:1.02rem; color:{TEXT};">
                The biggest opportunity is not only to respond faster, but to solve issues more consistently and more intelligently.
                This app shows where pressure exists, which customers are at risk, and how your team can reuse proven solutions to improve customer satisfaction.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )