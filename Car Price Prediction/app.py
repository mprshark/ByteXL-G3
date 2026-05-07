import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# NEOBRUTALIST CSS + CAR ANIMATION
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&display=swap');

* { font-family: 'Montserrat', sans-serif !important; }

html, body, .main, .block-container {
    background: #f5f5f0 !important;
    scroll-behavior: smooth;
}

.block-container { padding-top: 0rem !important; padding-bottom: 2rem !important; }

/* ── Marquee ── */
.marquee-wrap {
    background: #111;
    color: #00FF90;
    font-weight: 900;
    font-size: 1rem;
    padding: 10px 0;
    overflow: hidden;
    white-space: nowrap;
    border-bottom: 4px solid #000;
}
.marquee-inner {
    display: inline-block;
    animation: marquee 22s linear infinite;
}
@keyframes marquee { from{transform:translateX(0)} to{transform:translateX(-50%)} }

/* ── Hero ── */
.hero {
    background: #FFE100;
    border: 5px solid #000;
    box-shadow: 10px 10px 0 #000;
    padding: 40px 50px 20px;
    margin: 24px 0 32px;
    position: relative;
    overflow: hidden;
}
.hero h1 {
    font-size: 4.5rem;
    font-weight: 900;
    line-height: 0.95;
    color: #000;
    margin: 0;
}
.hero p {
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 12px;
    color: #111;
}

/* ── Car Animation ── */
.road {
    background: #222;
    height: 70px;
    border: 4px solid #000;
    border-top: none;
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
}
.road-line {
    position: absolute;
    top: 50%;
    width: 60px;
    height: 5px;
    background: #FFE100;
    animation: road-scroll 1s linear infinite;
}
@keyframes road-scroll {
    from { left: 110%; }
    to   { left: -10%; }
}
.car-svg {
    position: absolute;
    bottom: 10px;
    left: 40%;
    animation: car-bounce 0.3s ease-in-out infinite alternate;
}
@keyframes car-bounce { from{bottom:10px} to{bottom:14px} }

/* ── Neo Cards ── */
.neo-card {
    background: #fff;
    border: 4px solid #000;
    box-shadow: 8px 8px 0 #000;
    padding: 24px;
    margin-bottom: 28px;
    transition: transform .15s, box-shadow .15s;
}
.neo-card:hover {
    transform: translate(-4px,-4px);
    box-shadow: 12px 12px 0 #000;
}
.neo-card.cyan  { background: #00F0FF; }
.neo-card.green { background: #00FF90; }
.neo-card.pink  { background: #FF2D87; }
.neo-card.yellow{ background: #FFE100; }

/* ── Section headers ── */
.section-tag {
    display: inline-block;
    background: #000;
    color: #FFE100;
    font-weight: 900;
    font-size: 1.2rem;
    padding: 6px 18px;
    margin-bottom: 18px;
    letter-spacing: 2px;
}

/* ── Metrics ── */
.metric-box {
    background: #fff;
    border: 4px solid #000;
    box-shadow: 6px 6px 0 #000;
    padding: 16px 20px;
    text-align: center;
}
.metric-box .val {
    font-size: 2.2rem;
    font-weight: 900;
}
.metric-box .lbl {
    font-size: 0.8rem;
    font-weight: 700;
    color: #555;
    letter-spacing: 1px;
}

/* ── Streamlit buttons ── */
.stButton > button {
    background: #FF2D87 !important;
    color: #fff !important;
    border: 4px solid #000 !important;
    box-shadow: 6px 6px 0 #000 !important;
    border-radius: 0 !important;
    font-size: 1.3rem !important;
    font-weight: 900 !important;
    width: 100%;
    padding: 18px !important;
    margin-top: 10px;
    transition: all .1s;
}
.stButton > button:hover { background: #e0006e !important; }
.stButton > button:active {
    transform: translate(4px,4px);
    box-shadow: 2px 2px 0 #000 !important;
}

/* Inputs / selects */
.stNumberInput input, .stTextInput input {
    border: 3px solid #000 !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
    background: #fff !important;
}
.stSelectbox > div > div {
    border: 3px solid #000 !important;
    border-radius: 0 !important;
    font-weight: 700 !important;
}

/* Footer */
.neo-footer {
    background: #111;
    color: #fff;
    text-align: center;
    font-weight: 900;
    padding: 36px;
    border-top: 5px solid #000;
    letter-spacing: 2px;
    margin-top: 40px;
}

/* Step badges */
.step {
    display: inline-block;
    background: #FF2D87;
    color: #fff;
    font-weight: 900;
    border: 3px solid #000;
    padding: 4px 12px;
    margin-bottom: 10px;
    font-size: 0.9rem;
}

/* Price display */
.price-display {
    font-size: 3.5rem;
    font-weight: 900;
    color: #000;
    line-height: 1;
    margin: 10px 0;
}

/* RAG box */
.rag-box {
    background: #fff;
    border: 3px solid #000;
    padding: 14px 18px;
    margin-top: 14px;
    box-shadow: 4px 4px 0 #000;
}
.rag-box p { font-weight: 700; margin: 0; color: #111; }

/* Hide hamburger & streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MARQUEE
# ─────────────────────────────────────────────
st.markdown("""
<div class="marquee-wrap">
  <span class="marquee-inner">
    🚗 ADVANCED ML PIPELINE &nbsp;•&nbsp; DATA CLEANING &nbsp;•&nbsp; EDA &nbsp;•&nbsp;
    OUTLIER REMOVAL &nbsp;•&nbsp; FEATURE ENGINEERING &nbsp;•&nbsp; RAG INSIGHTS &nbsp;•&nbsp;
    XGBOOST MODEL &nbsp;•&nbsp; 🚗 ADVANCED ML PIPELINE &nbsp;•&nbsp; DATA CLEANING &nbsp;•&nbsp;
    EDA &nbsp;•&nbsp; OUTLIER REMOVAL &nbsp;•&nbsp; FEATURE ENGINEERING &nbsp;•&nbsp; RAG INSIGHTS &nbsp;•&nbsp; XGBOOST MODEL &nbsp;•&nbsp;
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>CAR PRICE<br>PREDICTOR</h1>
  <p>Powered by XGBoost &nbsp;|&nbsp; Full ML Pipeline &nbsp;|&nbsp; RAG Insights</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CAR ANIMATION (CSS only, no JS needed)
# ─────────────────────────────────────────────
st.markdown("""
<div class="road">
  <div class="road-line" style="animation-delay:0s"></div>
  <div class="road-line" style="animation-delay:-0.4s"></div>
  <div class="road-line" style="animation-delay:-0.8s"></div>
  <svg class="car-svg" width="100" height="45" viewBox="0 0 100 45">
    <rect x="5" y="20" width="90" height="20" rx="4" fill="#222"/>
    <rect x="20" y="8" width="55" height="20" rx="6" fill="#444"/>
    <circle cx="22" cy="40" r="8" fill="#111" stroke="#FFE100" stroke-width="3"/>
    <circle cx="78" cy="40" r="8" fill="#111" stroke="#FFE100" stroke-width="3"/>
    <rect x="25" y="11" width="20" height="14" rx="3" fill="#00F0FF" opacity=".7"/>
    <rect x="50" y="11" width="20" height="14" rx="3" fill="#00F0FF" opacity=".7"/>
    <polygon points="0,40 10,28 14,40" fill="#FF2D87"/>
  </svg>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STEP 1 — DATA LOADING & CLEANING
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">STEP 1 — DATA CLEANING & FORMING</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_and_process():
    df_raw = pd.read_csv("car data.csv")
    original_size = len(df_raw)

    # --- Cleaning ---
    df = df_raw.copy()
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()
    df['Year'] = df['Year'].astype(int)
    df['Kms_Driven'] = df['Kms_Driven'].astype(int)

    # --- Outlier Removal (IQR) ---
    before_outlier = len(df)
    for col in ['Selling_Price', 'Kms_Driven', 'Present_Price']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    outliers_removed = before_outlier - len(df)

    # --- Feature Engineering ---
    df['Age'] = 2024 - df['Year']
    df['Depreciation'] = (df['Present_Price'] - df['Selling_Price']) / (df['Age'] + 1)
    df['Brand'] = df['Car_Name'].apply(lambda x: str(x).split()[0].capitalize())

    return df, original_size, outliers_removed

df, orig_size, outliers_removed = load_and_process()

with st.container():
    st.markdown('<div class="neo-card cyan">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col_ui, label, value, color in [
        (c1, "TOTAL ROWS", orig_size, "#000"),
        (c2, "AFTER CLEANING", len(df), "#000"),
        (c3, "OUTLIERS REMOVED", outliers_removed, "#FF2D87"),
        (c4, "FEATURES CREATED", 3, "#000"),
    ]:
        col_ui.markdown(f"""
        <div class="metric-box">
          <div class="val" style="color:{color}">{value}</div>
          <div class="lbl">{label}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STEP 2 — EDA PLOTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">STEP 2 — EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="neo-card">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Price Distribution",
        "🔥 Correlation Heatmap",
        "⛽ Fuel vs Price",
        "📈 Kms vs Price",
    ])

    sns.set_theme(style="whitegrid")
    PLOT_COLOR = "#FF2D87"

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['Selling_Price'], kde=True, ax=ax, color=PLOT_COLOR)
        ax.set_title("Selling Price Distribution", fontweight='bold')
        ax.set_xlabel("Price (Lakhs)")
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 5))
        num_cols = df[['Selling_Price', 'Present_Price', 'Kms_Driven', 'Age', 'Depreciation', 'Owner']].corr()
        sns.heatmap(num_cols, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=2, linecolor='#000')
        ax.set_title("Correlation Heatmap", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df, ax=ax, palette=['#FFE100', '#FF2D87', '#00F0FF'])
        ax.set_title("Fuel Type vs Selling Price", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x='Kms_Driven', y='Selling_Price', hue='Fuel_Type', data=df, ax=ax, palette='Set1', alpha=0.7)
        ax.set_title("Kms Driven vs Selling Price", fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING SUMMARY
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">STEP 3 — FEATURE ENGINEERING</div>', unsafe_allow_html=True)
st.markdown('<div class="neo-card yellow">', unsafe_allow_html=True)
fe_col1, fe_col2, fe_col3 = st.columns(3)
fe_col1.markdown("**Age** (2024 − Year)\nCaptures how old the car is.", unsafe_allow_html=True)
fe_col2.markdown("**Depreciation** ((PresentPrice − SellingPrice) / Age)\nAnnual value loss rate.", unsafe_allow_html=True)
fe_col3.markdown("**Brand** (First word of Car_Name)\nGroups vehicles by manufacturer.", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STEP 4 — RAG KNOWLEDGE BASE (lightweight)
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">STEP 4 — RAG KNOWLEDGE BASE</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def build_rag(df):
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        insights = [
            f"Cars with {f} fuel type sell for an average of ₹{df[df['Fuel_Type']==f]['Selling_Price'].mean():.2f}L."
            for f in df['Fuel_Type'].unique()
        ] + [
            f"{t} transmission cars average ₹{df[df['Transmission']==t]['Selling_Price'].mean():.2f}L."
            for t in df['Transmission'].unique()
        ] + [
            "Low mileage cars (< 20k kms) retain significantly higher value.",
            "First-owner cars command a 15–25% premium over second-owner ones.",
            "Dealer-sold cars usually have better documentation and service records.",
            "Diesel cars tend to depreciate less in high-use markets.",
            "Automatic transmission is gaining demand in urban India.",
            "Cars older than 10 years depreciate sharply regardless of mileage.",
        ]

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(insights).astype('float32')
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return insights, index, model
    except Exception:
        return None, None, None

with st.container():
    st.markdown('<div class="neo-card">', unsafe_allow_html=True)
    with st.spinner("Building RAG knowledge base..."):
        rag_insights, rag_index, rag_embed_model = build_rag(df)
    if rag_insights:
        st.success(f"✅ RAG system ready — {len(rag_insights)} insights indexed.")
    else:
        st.warning("⚠️ RAG fallback mode (install faiss-cpu & sentence-transformers for full RAG).")
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STEP 5 — TRAIN / TEST SPLIT & MODEL TRAINING
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">STEP 5 — MODEL TRAINING & ACCURACY</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def train_model(df):
    FEATURES = ['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Age']
    TARGET   = 'Selling_Price'

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(),        ['Present_Price', 'Kms_Driven', 'Owner', 'Age']),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),
                                         ['Fuel_Type', 'Seller_Type', 'Transmission']),
    ])

    pipe = Pipeline([
        ('pre', preprocessor),
        ('model', XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            subsample=0.8, colsample_bytree=0.9, random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        'r2':   r2_score(y_test, y_pred),
        'mae':  mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'train_size': len(X_train),
        'test_size':  len(X_test),
    }
    return pipe, metrics

with st.spinner("Training XGBoost model..."):
    model, metrics = train_model(df)

with st.container():
    st.markdown('<div class="neo-card green">', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    for col_ui, label, val, fmt in [
        (m1, "R² SCORE",   metrics['r2'],   f"{metrics['r2']:.2%}"),
        (m2, "MAE",        metrics['mae'],  f"₹{metrics['mae']:.3f}L"),
        (m3, "RMSE",       metrics['rmse'], f"₹{metrics['rmse']:.3f}L"),
        (m4, "TRAIN SIZE", metrics['train_size'], str(metrics['train_size'])),
        (m5, "TEST SIZE",  metrics['test_size'],  str(metrics['test_size'])),
    ]:
        col_ui.markdown(f"""
        <div class="metric-box">
          <div class="val">{fmt}</div>
          <div class="lbl">{label}</div>
        </div>""", unsafe_allow_html=True)

    # Actual vs Predicted scatter
    X_all = df[['Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner','Age']]
    y_all = df['Selling_Price']
    _, X_test_df, _, y_test_df = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    y_pred_all = model.predict(X_test_df)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(y_test_df, y_pred_all, alpha=0.6, color='#FF2D87', edgecolors='#000', linewidths=0.5)
    ax.plot([y_test_df.min(), y_test_df.max()],
            [y_test_df.min(), y_test_df.max()], 'k--', linewidth=2)
    ax.set_xlabel("Actual Price (L)", fontweight='bold')
    ax.set_ylabel("Predicted Price (L)", fontweight='bold')
    ax.set_title("Actual vs Predicted", fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STEP 6 — PREDICTION FORM
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">STEP 6 — PREDICT YOUR CAR PRICE</div>', unsafe_allow_html=True)

with st.container():
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="neo-card cyan">', unsafe_allow_html=True)
        with st.form("pred_form"):
            year          = st.number_input("📅 Year of Purchase",      1990, 2024, 2018, step=1)
            present_price = st.number_input("💰 Showroom Price (Lakhs)", 0.5, 100.0, 10.0, step=0.5)
            kms           = st.number_input("🛣️ Kms Driven",             0, 500000, 25000, step=1000)
            fuel          = st.selectbox("⛽ Fuel Type",         df['Fuel_Type'].unique().tolist())
            seller        = st.selectbox("🏪 Seller Type",        df['Seller_Type'].unique().tolist())
            trans         = st.selectbox("⚙️ Transmission",        df['Transmission'].unique().tolist())
            owner         = st.selectbox("👤 Previous Owners",    [0, 1, 2, 3])
            submitted = st.form_submit_button("🚗 PREDICT PRICE")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if submitted:
            age_val = 2024 - year
            input_df = pd.DataFrame([{
                'Present_Price': present_price,
                'Kms_Driven':    kms,
                'Owner':         owner,
                'Age':           age_val,
                'Fuel_Type':     fuel,
                'Seller_Type':   seller,
                'Transmission':  trans,
            }])
            prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, prediction)

            # RAG Insight
            insight_text = "💡 Tip: Maintain service records and keep mileage low to maximize resale value."
            if rag_insights and rag_index and rag_embed_model:
                query_vec = rag_embed_model.encode([f"{fuel} {trans} car"]).astype('float32')
                _, idx = rag_index.search(query_vec, 1)
                insight_text = rag_insights[idx[0][0]]

            st.markdown(f"""
            <div class="neo-card green">
              <div class="step">ESTIMATED PRICE</div>
              <div class="price-display">₹ {prediction:.2f} L</div>
              <p style="font-weight:700;color:#333">Age: {age_val} years &nbsp;|&nbsp; Fuel: {fuel} &nbsp;|&nbsp; {trans}</p>
              <div class="rag-box">
                <p style="font-size:.8rem;color:#666;margin-bottom:4px;">🤖 AI INSIGHT (RAG)</p>
                <p>{insight_text}</p>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="neo-card" style="text-align:center;padding:60px 20px">
              <div style="font-size:4rem">🚗</div>
              <p style="font-weight:900;font-size:1.2rem;margin-top:16px">Fill the form and hit PREDICT</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="neo-footer">
  ADVANCED ML PROJECT &nbsp;•&nbsp; GROUP 5 &nbsp;•&nbsp; XGBOOST + RAG + NEOBRUTALISM
</div>
""", unsafe_allow_html=True)
