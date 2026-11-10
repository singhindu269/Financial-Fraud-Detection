
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Financial Fraud Detection | Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. PROFESSIONAL POWER BI STYLE
# =====================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
        color: #172B4D;
    }

    [data-testid="stHeader"] {
        background: rgba(245,247,251,0.95);
    }

    section[data-testid="stSidebar"] {
        background-color: #102A43;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    h1, h2, h3, h4 {
        color: #12345A !important;
        font-weight: 700 !important;
    }

    p, label, span {
        color: #344563;
    }

    .main-title {
        color: #12345A;
        font-size: 31px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #627D98;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .section-title {
        color: #12345A;
        font-size: 20px;
        font-weight: 700;
        padding-bottom: 8px;
        border-bottom: 2px solid #D9E5F2;
        margin-top: 14px;
        margin-bottom: 16px;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E1E8F0;
        border-left: 4px solid #2878C8;
        border-radius: 9px;
        padding: 15px 16px;
        min-height: 105px;
        box-shadow: 0 2px 7px rgba(16,42,67,0.05);
    }

    .kpi-label {
        color: #627D98;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #12345A;
        font-size: 25px;
        font-weight: 800;
    }

    .kpi-note {
        color: #829AB1;
        font-size: 11px;
        margin-top: 4px;
    }

    .insight-card {
        background: #FFFFFF;
        border: 1px solid #D9E5F2;
        border-radius: 9px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .insight-title {
        color: #12345A;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .insight-text {
        color: #486581;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E1E8F0;
        border-radius: 9px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(16,42,67,0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #627D98;
    }

    div[data-testid="stMetricValue"] {
        color: #12345A;
    }

    div[data-testid="stTabs"] button {
        color: #486581;
        font-weight: 600;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #1769AA;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #D9E5F2;
        border-radius: 8px;
    }

    .footer {
        text-align: center;
        color: #829AB1;
        font-size: 12px;
        padding: 18px;
        border-top: 1px solid #D9E5F2;
        margin-top: 25px;
    }

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. COLOUR PALETTE AND CHART THEME
# =====================================================

NAVY = "#12345A"
BLUE = "#2878C8"
TEAL = "#16A6A1"
RED = "#E05260"
ORANGE = "#F0A33A"
LIGHT_BLUE = "#8CBCE8"

CHART_CONFIG = {
    "displayModeBar": False,
    "responsive": True
}


def style_chart(fig, height=350):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            color=NAVY,
            size=12
        ),
        title_font=dict(
            size=17,
            color=NAVY
        ),
        margin=dict(l=25, r=25, t=60, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(showgrid=False, linecolor="#D9E5F2")
    fig.update_yaxes(gridcolor="#EAF0F6", zeroline=False)
    return fig


# =====================================================
# 4. LOAD MODEL ARTIFACTS
# =====================================================

@st.cache_resource
def load_model_files():
    model = joblib.load("Tuned_Random_Forest.pkl")
    scaler = joblib.load("Robust_Scaler.pkl")
    features = joblib.load("Feature_Columns.pkl")

    return model, scaler, list(features)


try:
    model, scaler, features = load_model_files()

except Exception as e:
    st.error("Model files could not be loaded.")
    st.write("Make sure these files are in the same folder as app.py:")
    st.code("""
Tuned_Random_Forest.pkl
Robust_Scaler.pkl
Feature_Columns.pkl
""")
    st.exception(e)
    st.stop()


# =====================================================
# 5. SIDEBAR
# =====================================================

st.sidebar.markdown("# 🛡️ Fraud Analytics")

st.sidebar.markdown("---")

st.sidebar.markdown("### REPORT NAVIGATION")
st.sidebar.write("📊 Executive Overview")
st.sidebar.write("📈 Fraud Analytics")
st.sidebar.write("⚠️ Risk Monitoring")
st.sidebar.write("🎯 Model Evaluation")
st.sidebar.write("🔍 Transaction Explorer")

st.sidebar.markdown("---")

st.sidebar.markdown("### MODEL DETAILS")
st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write("**Model:** Tuned Random Forest")
st.sidebar.write("**Preprocessing:** Robust Scaler")

st.sidebar.markdown("---")

st.sidebar.markdown("### PROJECT")
st.sidebar.write("Financial Fraud Detection")
st.sidebar.caption("Machine Learning | Python | Streamlit")

st.sidebar.markdown("---")
st.sidebar.caption("Executive Analytics Dashboard")


# =====================================================
# 6. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">Financial Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Executive Intelligence Dashboard | Credit Card Transaction Monitoring'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# 7. FILE UPLOAD
# =====================================================

with st.expander("📂 Upload Transaction Dataset", expanded=True):

    uploaded_file = st.file_uploader(
        "Upload your credit card transaction CSV",
        type=["csv"],
        help="Upload a CSV containing the model training predictor columns."
    )

if uploaded_file is None:
    st.info("Upload your CSV dataset to generate the executive fraud report.")

    st.markdown("### Expected model input features")
    st.code(", ".join(features))
    st.stop()


# =====================================================
# 8. READ AND VALIDATE DATA
# =====================================================

try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

except Exception as e:
    st.error("Unable to read the uploaded CSV.")
    st.exception(e)
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

missing_features = [
    col for col in features if col not in df.columns
]

if missing_features:
    st.error("Required model columns are missing.")
    st.write(missing_features)
    st.stop()

has_actual_labels = "Class" in df.columns
actual_labels = df["Class"].copy() if has_actual_labels else None

X = df[features].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

if X.isna().any().any():
    st.error("Missing or non-numeric values exist in model input columns.")
    st.write(X.columns[X.isna().any()].tolist())
    st.stop()

if not np.isfinite(X.to_numpy(dtype=float)).all():
    st.error("Infinite values found in the model input.")
    st.stop()


# =====================================================
# 9. DATA PREPROCESSING
# =====================================================

try:
    X_processed = X.copy()

    scaler_columns = list(
        getattr(scaler, "feature_names_in_", [])
    )

    if scaler_columns:

        missing_scaler_columns = [
            col for col in scaler_columns
            if col not in X_processed.columns
        ]

        if missing_scaler_columns:
            st.error(
                "Scaler input columns are missing: "
                + ", ".join(missing_scaler_columns)
            )
            st.stop()

        X_processed.loc[:, scaler_columns] = scaler.transform(
            X_processed[scaler_columns]
        )

    else:

        scale_columns = [
            col for col in ["Time", "Amount"]
            if col in X_processed.columns
        ]

        if scale_columns:
            X_processed.loc[:, scale_columns] = scaler.transform(
                X_processed[scale_columns]
            )

    X_processed = X_processed[features]

except Exception as e:
    st.error("Preprocessing failed.")
    st.exception(e)
    st.stop()


# =====================================================
# 10. PREDICTIONS
# =====================================================

try:
    predictions = model.predict(X_processed)

    if not hasattr(model, "predict_proba"):
        st.error("The loaded model does not support probability predictions.")
        st.stop()

    probabilities = model.predict_proba(X_processed)

    classes = list(model.classes_)

    if 1 not in classes:
        st.error("Fraud class 1 was not found in the model classes.")
        st.stop()

    fraud_probability = probabilities[:, classes.index(1)]

except Exception as e:
    st.error("Prediction failed.")
    st.exception(e)
    st.stop()


# =====================================================
# 11. BUILD RESULTS TABLE
# =====================================================

results = df.copy()

results["Fraud_Prediction"] = predictions.astype(int)
results["Fraud_Probability"] = fraud_probability

results["Prediction_Status"] = np.where(
    results["Fraud_Prediction"] == 1,
    "Fraud",
    "Genuine"
)

results["Risk_Category"] = pd.cut(
    results["Fraud_Probability"],
    bins=[-0.000001, 0.30, 0.70, 1.000001],
    labels=["Low Risk", "Medium Risk", "High Risk"]
).astype(str)


# =====================================================
# 12. EXECUTIVE FILTERS
# =====================================================

st.markdown(
    '<div class="section-title">Report Filters</div>',
    unsafe_allow_html=True
)

filter1, filter2 = st.columns(2)

with filter1:
    selected_status = st.multiselect(
        "Transaction Status",
        ["Fraud", "Genuine"],
        default=["Fraud", "Genuine"],
        key="status_filter"
    )

with filter2:
    selected_risk = st.multiselect(
        "Risk Level",
        ["Low Risk", "Medium Risk", "High Risk"],
        default=["Low Risk", "Medium Risk", "High Risk"],
        key="risk_filter"
    )

filtered = results[
    results["Prediction_Status"].isin(selected_status)
    & results["Risk_Category"].isin(selected_risk)
].copy()


# =====================================================
# 13. KPI CALCULATIONS
# =====================================================

total_transactions = len(filtered)

fraud_count = int(
    (filtered["Prediction_Status"] == "Fraud").sum()
)

genuine_count = int(
    (filtered["Prediction_Status"] == "Genuine").sum()
)

fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions else 0
)

avg_probability = (
    filtered["Fraud_Probability"].mean() * 100
    if total_transactions else 0
)

if "Amount" in filtered.columns:
    amount_values = pd.to_numeric(
        filtered["Amount"], errors="coerce"
    )
    total_amount = amount_values.sum()
else:
    total_amount = 0


# =====================================================
# 14. EXECUTIVE KPI CARDS
# =====================================================

st.markdown(
    '<div class="section-title">Executive Overview</div>',
    unsafe_allow_html=True
)


def kpi_card(label, value, note, accent):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    kpi_card(
        "TOTAL TRANSACTIONS",
        f"{total_transactions:,}",
        "Filtered transactions",
        BLUE
    )

with k2:
    kpi_card(
        "FRAUD DETECTED",
        f"{fraud_count:,}",
        "Predicted fraud cases",
        RED
    )

with k3:
    kpi_card(
        "GENUINE TRANSACTIONS",
        f"{genuine_count:,}",
        "Predicted genuine cases",
        TEAL
    )

with k4:
    kpi_card(
        "FRAUD RATE",
        f"{fraud_rate:.2f}%",
        "Predicted fraud percentage",
        ORANGE
    )

with k5:
    kpi_card(
        "AVG FRAUD PROBABILITY",
        f"{avg_probability:.2f}%",
        "Average predicted probability",
        LIGHT_BLUE
    )


st.markdown("")


# =====================================================
# 15. REPORT TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "Fraud & Risk Analytics",
    "Model Evaluation",
    "Transaction Explorer"
])


# =====================================================
# TAB 1: EXECUTIVE OVERVIEW
# =====================================================

with tab1:

    st.markdown(
        '<div class="section-title">Transaction Performance</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        status_data = filtered["Prediction_Status"].value_counts()

        fig = px.pie(
            names=status_data.index,
            values=status_data.values,
            hole=0.62,
            title="Transaction Classification",
            color=status_data.index,
            color_discrete_map={
                "Fraud": RED,
                "Genuine": TEAL
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            style_chart(fig, 340),
            use_container_width=True,
            config=CHART_CONFIG
        )

    with col2:

        risk_data = (
            filtered["Risk_Category"]
            .value_counts()
            .reindex(
                ["Low Risk", "Medium Risk", "High Risk"],
                fill_value=0
            )
            .reset_index()
        )

        risk_data.columns = ["Risk Category", "Transactions"]

        fig = px.bar(
            risk_data,
            x="Risk Category",
            y="Transactions",
            color="Risk Category",
            title="Transaction Risk Distribution",
            color_discrete_map={
                "Low Risk": TEAL,
                "Medium Risk": ORANGE,
                "High Risk": RED
            },
            text="Transactions"
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            style_chart(fig, 340),
            use_container_width=True,
            config=CHART_CONFIG
        )

    st.markdown(
        '<div class="section-title">Fraud Probability Analysis</div>',
        unsafe_allow_html=True
    )

    fig = px.histogram(
        filtered,
        x="Fraud_Probability",
        color="Prediction_Status",
        nbins=35,
        barmode="overlay",
        opacity=0.75,
        title="Distribution of Fraud Probability",
        color_discrete_map={
            "Fraud": RED,
            "Genuine": BLUE
        }
    )

    st.plotly_chart(
        style_chart(fig, 350),
        use_container_width=True,
        config=CHART_CONFIG
    )

    st.markdown("### Executive Insights")

    insight1, insight2 = st.columns(2)

    with insight1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Fraud Monitoring</div>
                <div class="insight-text">
                    The model identified {fraud_count:,} transactions
                    as potentially fraudulent within the selected filters.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Risk Exposure</div>
                <div class="insight-text">
                    The average predicted fraud probability for the
                    selected transactions is {avg_probability:.2f}%.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# TAB 2: FRAUD AND RISK ANALYTICS
# =====================================================

with tab2:

    st.markdown(
        '<div class="section-title">Fraud & Risk Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        risk_summary = (
            filtered.groupby("Risk_Category", observed=False)
            .agg(
                Transactions=("Fraud_Probability", "count"),
                Average_Probability=("Fraud_Probability", "mean")
            )
            .reindex(["Low Risk", "Medium Risk", "High Risk"])
            .reset_index()
        )

        risk_summary["Average_Probability"] *= 100

        fig = px.bar(
            risk_summary,
            x="Risk_Category",
            y="Transactions",
            color="Risk_Category",
            title="Risk Category Breakdown",
            color_discrete_map={
                "Low Risk": TEAL,
                "Medium Risk": ORANGE,
                "High Risk": RED
            },
            text="Transactions"
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
            config=CHART_CONFIG
        )

    with col2:

        fig = px.bar(
            risk_summary,
            x="Risk_Category",
            y="Average_Probability",
            color="Risk_Category",
            title="Average Probability by Risk Level",
            color_discrete_map={
                "Low Risk": TEAL,
                "Medium Risk": ORANGE,
                "High Risk": RED
            }
        )

        fig.update_layout(showlegend=False)

        fig.update_yaxes(title="Average Probability (%)")

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
            config=CHART_CONFIG
        )

    if "Amount" in filtered.columns:

        st.markdown(
            '<div class="section-title">Transaction Amount Analysis</div>',
            unsafe_allow_html=True
        )

        amount_data = filtered.copy()

        amount_data["Amount"] = pd.to_numeric(
            amount_data["Amount"], errors="coerce"
        )

        fig = px.box(
            amount_data,
            x="Prediction_Status",
            y="Amount",
            color="Prediction_Status",
            title="Transaction Amount by Prediction",
            color_discrete_map={
                "Fraud": RED,
                "Genuine": BLUE
            }
        )

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
            config=CHART_CONFIG
        )

    st.markdown(
        '<div class="section-title">High-Risk Transactions</div>',
        unsafe_allow_html=True
    )

    high_risk = filtered[
        filtered["Risk_Category"] == "High Risk"
    ].sort_values(
        "Fraud_Probability", ascending=False
    )

    st.metric("High-Risk Transactions", f"{len(high_risk):,}")

    columns_to_show = [
        col for col in [
            "Time",
            "Amount",
            "Fraud_Probability",
            "Prediction_Status",
            "Risk_Category"
        ]
        if col in high_risk.columns
    ]

    st.dataframe(
        high_risk[columns_to_show].head(100),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# TAB 3: MODEL EVALUATION
# =====================================================

with tab3:

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    if has_actual_labels:

        y_true_all = pd.to_numeric(
            actual_labels, errors="coerce"
        )

        valid = y_true_all.isin([0, 1])

        y_true = y_true_all[valid].astype(int)

        y_pred = pd.Series(
            predictions, index=results.index
        )[valid]

        y_prob = pd.Series(
            fraud_probability, index=results.index
        )[valid]

        if len(y_true) > 0:

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred, zero_division=0
            )
            recall = recall_score(
                y_true, y_pred, zero_division=0
            )
            f1 = f1_score(
                y_true, y_pred, zero_division=0
            )

            a, b, c, d = st.columns(4)

            a.metric("Accuracy", f"{accuracy * 100:.2f}%")
            b.metric("Precision", f"{precision * 100:.2f}%")
            c.metric("Recall", f"{recall * 100:.2f}%")
            d.metric("F1 Score", f"{f1 * 100:.2f}%")

            if y_true.nunique() == 2:

                avg_precision = average_precision_score(
                    y_true, y_prob
                )

                st.metric(
                    "Average Precision",
                    f"{avg_precision:.4f}"
                )

            st.markdown("### Confusion Matrix")

            cm = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            )

            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Predicted Genuine", "Predicted Fraud"],
                    y=["Actual Genuine", "Actual Fraud"],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale=[
                        [0, "#EAF2F8"],
                        [1, "#2878C8"]
                    ],
                    showscale=False
                )
            )

            fig.update_layout(
                title="Actual vs Predicted Transactions",
                template="plotly_white",
                height=400,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color=NAVY)
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=CHART_CONFIG
            )

            st.caption(
                "Metrics compare predictions against the supplied Class "
                "labels. For an unbiased performance estimate, evaluate "
                "on an independent test dataset."
            )

        else:
            st.warning("No valid actual Class labels were found.")

    else:

        st.info(
            "Your uploaded CSV does not contain an actual Class column. "
            "Upload labelled test data to calculate model performance."
        )

    st.markdown(
        '<div class="section-title">Feature Importance</div>',
        unsafe_allow_html=True
    )

    try:

        estimator = model

        if hasattr(model, "named_steps"):

            for step in reversed(
                list(model.named_steps.values())
            ):

                if hasattr(step, "feature_importances_"):
                    estimator = step
                    break

        if hasattr(estimator, "feature_importances_"):

            importance = pd.DataFrame({
                "Feature": features,
                "Importance": estimator.feature_importances_
            }).sort_values(
                "Importance", ascending=False
            ).head(15)

            fig = px.bar(
                importance.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Important Features",
                color="Importance",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(
                style_chart(fig, 450),
                use_container_width=True,
                config=CHART_CONFIG
            )

        else:
            st.info(
                "Feature importance is not available for this model."
            )

    except Exception as e:
        st.warning("Unable to display feature importance.")
        st.caption(str(e))


# =====================================================
# TAB 4: TRANSACTION EXPLORER
# =====================================================

with tab4:

    st.markdown(
        '<div class="section-title">Transaction Explorer</div>',
        unsafe_allow_html=True
    )

    search_col, probability_col = st.columns(2)

    with search_col:

        search_status = st.selectbox(
            "Select transaction type",
            ["All", "Fraud", "Genuine"]
        )

    with probability_col:

        min_probability = st.slider(
            "Minimum fraud probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

    explorer = results[
        results["Fraud_Probability"] >= min_probability
    ].copy()

    if search_status != "All":

        explorer = explorer[
            explorer["Prediction_Status"] == search_status
        ]

    st.write(
        f"Transactions matching filters: {len(explorer):,}"
    )

    st.dataframe(
        explorer,
        use_container_width=True,
        height=420
    )

    st.markdown("### Export Results")

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Complete Fraud Prediction Report",
        data=csv_data,
        file_name="Financial_Fraud_Detection_Report.csv",
        mime="text/csv",
        use_container_width=True
    )


# =====================================================
# 16. FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        Financial Fraud Detection | Executive Analytics Dashboard<br>
        Machine Learning • Python • Random Forest • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
