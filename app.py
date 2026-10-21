"""
CardioScan AI - Heart Disease Prediction & Paper Replication Dashboard
Replication and Interactive Analysis of Ramesh et al. (2022)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data import (
    FEATURES, NUMERIC, BINARY, CATEGORICAL, PAPER_7,
    load_cleveland, split_xy
)
from src.models import build_pipeline, MAIN_MODELS
from src.evaluation import METRICS, PROTOCOLS

# Configure Streamlit Page
st.set_page_config(
    page_title="CardioScan AI | Heart Disease ML Lab",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #090d16 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.5;
        max-width: 900px;
    }
    .stat-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-right: 8px;
        margin-top: 10px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .alert-card-success {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        color: #34d399;
    }
    .alert-card-danger {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 12px;
        padding: 20px;
        color: #fb7185;
    }
    .verdict-refuted {
        background: #ef4444;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-replicated {
        background: #10b981;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .verdict-partial {
        background: #f59e0b;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache dataset and trained models
@st.cache_data
def get_dataset():
    df = load_cleveland()
    return df

@st.cache_resource
def get_trained_models():
    df = get_dataset()
    X, y = split_xy(df)
    models = {}
    for name in MAIN_MODELS:
        pipe = build_pipeline(name)
        pipe.fit(X, y)
        models[name] = pipe
    return models

@st.cache_data
def load_result_csv(filename):
    p = ROOT / "results" / filename
    if p.exists():
        return pd.read_csv(p)
    return None

df_raw = get_dataset()
trained_models = get_trained_models()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
    st.title("CardioScan AI")
    st.caption("Replication & Clinical Diagnostic Suite")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation",
        [
            "🩺 Live Patient Diagnosis",
            "📊 Model Benchmarks & Comparison",
            "⚖️ Paper Replication Audit",
            "🔬 Hyperparameter & Feature Lab",
            "📈 Dataset & EDA Gallery"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📋 Quick Project Info")
    st.markdown("""
    - **Dataset**: UCI Cleveland (303 rows)
    - **Features**: 13 clinical biomarkers
    - **Target**: Presence of CAD (`num > 0`)
    - **Paper**: Ramesh et al., *MJCS* (2022)
    - **Evaluation**: Protocols A, B, and C
    """)
    st.info("💡 Tip: All ML pipelines encapsulate imputation, scaling, and one-hot encoding without data leakage.")

# Header Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">CardioScan AI: Heart Disease Prediction & Paper Replication</div>
    <div class="hero-sub">
        An interactive machine learning laboratory replicating and auditing <em>"Predictive Analysis of Heart Diseases with Machine Learning Approaches"</em> (Ramesh et al., 2022) across 6 classification algorithms and 3 clinical validation protocols.
    </div>
    <div>
        <span class="stat-chip">🧬 303 Cleveland Patients</span>
        <span class="stat-chip">⚡ 6 ML Pipelines</span>
        <span class="stat-chip">🛡️ Zero Data Leakage</span>
        <span class="stat-chip">🔍 Rigorous Audit of Published Claims</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 1: LIVE PATIENT DIAGNOSIS
# -----------------------------------------------------------------------------
if app_mode == "🩺 Live Patient Diagnosis":
    st.subheader("🩺 Real-Time Patient Diagnostic Predictor")
    st.markdown("Enter patient clinical measurements or select a preset clinical scenario to run real-time inference across all 6 machine learning models.")
    
    preset = st.selectbox(
        "Preset Patient Profile",
        [
            "Custom Patient (Use sliders below)",
            "Profile 1: Healthy Asymptomatic Individual (Low Risk)",
            "Profile 2: Confirmed Coronary Artery Disease (High Risk)",
            "Profile 3: Borderline Patient (Moderate / Mixed Risk)"
        ]
    )
    
    # Preset values
    if preset == "Profile 1: Healthy Asymptomatic Individual (Low Risk)":
        p_age, p_sex, p_cp, p_trestbps, p_chol, p_fbs = 42, 0, 2, 115, 180, 0
        p_restecg, p_thalach, p_exang, p_oldpeak, p_slope, p_ca, p_thal = 0, 172, 0, 0.0, 1, 0, 3
    elif preset == "Profile 2: Confirmed Coronary Artery Disease (High Risk)":
        p_age, p_sex, p_cp, p_trestbps, p_chol, p_fbs = 63, 1, 4, 152, 290, 1
        p_restecg, p_thalach, p_exang, p_oldpeak, p_slope, p_ca, p_thal = 2, 112, 1, 2.6, 2, 2, 7
    elif preset == "Profile 3: Borderline Patient (Moderate / Mixed Risk)":
        p_age, p_sex, p_cp, p_trestbps, p_chol, p_fbs = 54, 1, 3, 132, 238, 0
        p_restecg, p_thalach, p_exang, p_oldpeak, p_slope, p_ca, p_thal = 1, 142, 0, 1.2, 2, 1, 6
    else:
        p_age, p_sex, p_cp, p_trestbps, p_chol, p_fbs = 55, 1, 4, 130, 240, 0
        p_restecg, p_thalach, p_exang, p_oldpeak, p_slope, p_ca, p_thal = 0, 150, 0, 1.0, 2, 0, 3

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("##### 👤 Demographic & Vitals")
        age = st.slider("Age (years)", 25, 80, p_age, help="Patient age in years")
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", index=0 if p_sex == 1 else 1)
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, p_trestbps, help="Resting BP on admission")
        chol = st.slider("Serum Cholesterol (mg/dl)", 120, 560, p_chol, help="Serum cholesterol level")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True (> 120 mg/dl)" if x == 1 else "False (<= 120 mg/dl)", index=p_fbs)

    with col2:
        st.markdown("##### 🫀 Symptoms & Cardiac Function")
        cp_map = {1: "1: Typical Angina", 2: "2: Atypical Angina", 3: "3: Non-Anginal Pain", 4: "4: Asymptomatic"}
        cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4], format_func=lambda x: cp_map[x], index=p_cp - 1)
        restecg_map = {0: "0: Normal", 1: "1: ST-T Wave Abnormality", 2: "2: Left Ventricular Hypertrophy"}
        restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2], format_func=lambda x: restecg_map[x], index=p_restecg)
        thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, p_thalach, help="Peak exercise heart rate")
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", index=p_exang)

    with col3:
        st.markdown("##### 🔬 ECG Stress & Fluoroscopy")
        oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.2, float(p_oldpeak), step=0.1)
        slope_map = {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3], format_func=lambda x: slope_map[x], index=p_slope - 1)
        ca = st.slider("Number of Major Vessels Colored (0-3)", 0, 3, p_ca, help="Fluoroscopy vessel count")
        thal_map = {3: "3: Normal", 6: "6: Fixed Defect", 7: "7: Reversible Defect"}
        thal = st.selectbox("Thalassemia (thal)", options=[3, 6, 7], format_func=lambda x: thal_map[x], index=[3, 6, 7].index(p_thal) if p_thal in [3, 6, 7] else 0)

    # Build input dataframe
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }])

    st.markdown("---")
    
    # Run predictions across all models
    preds = {}
    probs = {}
    for name, model in trained_models.items():
        p = model.predict(input_data)[0]
        preds[name] = p
        if hasattr(model, "predict_proba"):
            probs[name] = model.predict_proba(input_data)[0, 1]
        elif hasattr(model, "decision_function"):
            # Map decision function to sigmoid for display
            df_val = model.decision_function(input_data)[0]
            probs[name] = 1 / (1 + np.exp(-df_val))
        else:
            probs[name] = float(p)

    positive_votes = sum(preds.values())
    total_models = len(preds)
    mean_prob = np.mean(list(probs.values()))
    
    # Consensus diagnosis display
    res_col1, res_col2 = st.columns([1.2, 1.8])
    with res_col1:
        st.markdown("#### 🏁 Consensus Diagnosis")
        if positive_votes >= 4:
            st.markdown(f"""
            <div class="alert-card-danger">
                <h3 style="margin-top:0; color:#fb7185;">⚠️ Heart Disease Indicated</h3>
                <p><strong>{positive_votes} of {total_models} models</strong> classify this patient as <strong>Disease Present (Class 1)</strong>.</p>
                <p>Average Predicted Probability: <strong>{mean_prob*100:.1f}%</strong></p>
                <small>High clinical risk factors: Chest pain code {cp}, ST depression {oldpeak}, and {ca} colored vessel(s).</small>
            </div>
            """, unsafe_allow_html=True)
        elif positive_votes <= 2:
            st.markdown(f"""
            <div class="alert-card-success">
                <h3 style="margin-top:0; color:#34d399;">✅ Low Risk / No Disease Detected</h3>
                <p><strong>{total_models - positive_votes} of {total_models} models</strong> classify this patient as <strong>Healthy (Class 0)</strong>.</p>
                <p>Average Predicted Probability: <strong>{mean_prob*100:.1f}%</strong></p>
                <small>Vital signs and biomarker thresholds align with normal cardiac function.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(245, 158, 11, 0.12); border:1px solid rgba(245, 158, 11, 0.3); border-radius:12px; padding:20px; color:#fbbf24;">
                <h3 style="margin-top:0; color:#fbbf24;">⚖️ Borderline / Split Decision</h3>
                <p><strong>{positive_votes} of {total_models} models</strong> vote for Disease Present.</p>
                <p>Average Predicted Probability: <strong>{mean_prob*100:.1f}%</strong></p>
                <small>Clinical correlation and additional cardiology workup recommended.</small>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### 📊 Model-by-Model Breakdown")
        model_rows = []
        for name in MAIN_MODELS:
            model_rows.append({
                "Model": name,
                "Prediction": "🔴 Disease" if preds[name] == 1 else "🟢 Healthy",
                "Disease Probability": probs[name]
            })
        df_display = pd.DataFrame(model_rows)
        
        fig = px.bar(
            df_display,
            x="Disease Probability",
            y="Model",
            orientation="h",
            color="Disease Probability",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            range_x=[0, 1],
            text=df_display["Disease Probability"].apply(lambda v: f"{v*100:.1f}%")
        )
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=20, t=10, b=10),
            xaxis_title="Predicted Probability (Positive Class)",
            yaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: MODEL BENCHMARKS & COMPARISON
# -----------------------------------------------------------------------------
elif app_mode == "📊 Model Benchmarks & Comparison":
    st.subheader("📊 Rigorous Model Evaluation Across Protocols")
    st.markdown("Compare the performance of all 6 algorithms under three distinct validation protocols with zero data leakage.")

    protocol_choice = st.radio(
        "Select Validation Protocol",
        [
            "Protocol B: 70/30 Stratified Split (n=91 test)",
            "Protocol A: 60/40 Stratified Split (n=122 test)",
            "Protocol C: 10-Fold Stratified Cross-Validation (Mean ± Std)"
        ],
        horizontal=True
    )
    
    p_code = "B" if "70/30" in protocol_choice else ("A" if "60/40" in protocol_choice else "C")
    
    csv_file = f"16_test_metrics_all13_protocol{p_code}.csv"
    df_metrics = load_result_csv(csv_file)
    
    if df_metrics is not None:
        if "model" not in df_metrics.columns and "Unnamed: 0" in df_metrics.columns:
            df_metrics = df_metrics.rename(columns={"Unnamed: 0": "model"})
            
        st.markdown(f"##### 📋 Test Set Performance Table ({PROTOCOLS[p_code]})")
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        # Interactive chart
        if p_code in ["A", "B"]:
            plot_df = df_metrics.melt(id_vars=["model"], value_vars=["accuracy", "precision", "recall", "f1", "auc"], var_name="Metric", value_name="Score")
            fig = px.bar(
                plot_df,
                x="model",
                y="Score",
                color="Metric",
                barmode="group",
                title=f"All Metrics by Model - Protocol {p_code} ({PROTOCOLS[p_code]})",
                color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f43f5e"]
            )
            fig.update_layout(yaxis_range=[0.6, 1.0], xaxis_title="Classifier", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Parse C mean
            st.info("💡 In Protocol C (10-fold CV), metrics represent the mean and standard deviation across all 10 held-out folds.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("##### 🔲 Confusion Matrices (Protocol B)")
        cm_df = load_result_csv("18_confusion_matrices_protocolB_all13.csv")
        if cm_df is not None:
            st.dataframe(cm_df, use_container_width=True, hide_index=True)
            st.caption("TP = Disease Caught, FN = Disease Missed, FP = False Alarm, TN = Correct Healthy")
    
    with colB:
        st.markdown("##### 📉 ROC Curves Comparison")
        roc_img = ROOT / "results" / "fig_roc_curves_all_models.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="ROC Curves across Protocols A, B, and C", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: PAPER REPLICATION AUDIT
# -----------------------------------------------------------------------------
elif app_mode == "⚖️ Paper Replication Audit":
    st.subheader("⚖️ Scientific Replication Audit: Ramesh et al. (2022) vs Reality")
    st.markdown("""
    This section cross-examines the published findings in the *Malaysian Journal of Computer Science* against our clean replication.
    Every discrepancy is backed by verifiable code and rigorous statistical methodology.
    """)
    
    audit_tabs = st.tabs([
        "🚨 Critical Audit Findings",
        "📑 Paper vs Replication Table",
        "🔢 Confusion Matrix Recomputation",
        "🧮 Mathematical F1 Anomaly"
    ])
    
    with audit_tabs[0]:
        st.markdown("""
        ### Key Discrepancies Uncovered in the Published Paper:
        
        1. **❌ KNN Claim (94.1% Accuracy, 94.8% Recall) DOES NOT REPLICATE:**
           - **Paper Claim**: K-Nearest Neighbors ($k=8$) is the single best classifier with 94.1% accuracy.
           - **Replication**: In an honest, leak-free evaluation, KNN achieves **82.8% (Protocol A), 86.8% (Protocol B), and 83.8% (10-fold CV)**.
           - KNN ranks **5th** in Protocol A and ties for 1st in CV within standard error. Top performers are **Logistic Regression (89.3%)** and **Naive Bayes (87.9%)**.
           
        2. **❌ Class Balance Inversion in Paper:**
           - **Paper States**: 165 positive (disease) / 138 negative (healthy).
           - **Actual Cleveland Data**: Exactly **139 positive / 164 negative**. The paper inverted the labels and was off by one.
           
        3. **❌ Split Contradiction (60/40 vs 70/30):**
           - **Paper Text**: States models were evaluated on a **60/40 split**.
           - **Paper's Confusion Matrices**: Total $212 + 91 = 303$ patients. That is mathematically identical to scikit-learn's **70/30 split** (60/40 would be 181/122).
           
        4. **❌ Table 3 Accuracies Contradict Table 6 Confusion Matrices:**
           - Summing correct predictions from the paper's own confusion matrices in Table 6:
             - KNN: 79/91 correct = **86.8%** (NOT 94.1% as claimed in Table 3)
             - Random Forest: 75/91 correct = **82.4%** (NOT 87.9%)
             - Naive Bayes: 75/91 correct = **82.4%** (NOT 86.8%)
           - Our replicated KNN accuracy on the exact same 70/30 split is **86.8%**, perfectly matching Table 6!
           
        5. **❌ Mathematically Impossible F1 Scores:**
           - By definition, the harmonic mean $F_1 = \\frac{2PR}{P+R}$ must lie strictly between $\\min(P,R)$ and $\\max(P,R)$.
           - In the paper: Decision Tree reported $F_1 = 0.864$ with $P=0.778, R=0.755$. This is mathematically impossible!
        """)
        
    with audit_tabs[1]:
        st.markdown("##### 🔍 Direct Comparison: Published Table 3 vs Our Results (All 13 Features)")
        cmp_df = load_result_csv("23_paper_vs_ours_all13.csv")
        if cmp_df is not None:
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            
        st.markdown("##### 🏷️ Claims & Evidence Summary")
        claims_df = load_result_csv("32_claims_evidence.csv")
        if claims_df is not None:
            st.dataframe(claims_df, use_container_width=True, hide_index=True)

    with audit_tabs[2]:
        st.markdown("##### 🧮 Recomputed Metrics from Paper's Table 6 Confusion Matrices")
        t6_df = load_result_csv("27_paper_table6_recomputed.csv")
        if t6_df is not None:
            st.dataframe(t6_df, use_container_width=True, hide_index=True)
            
        t3_t6_df = load_result_csv("28_table3_vs_table6_accuracy.csv")
        if t3_t6_df is not None:
            st.markdown("##### 📈 Table 3 Reported Accuracy vs Table 6 Matrix-Implied Accuracy")
            st.dataframe(t3_t6_df, use_container_width=True)

    with audit_tabs[3]:
        st.markdown("##### ⚠️ Proof of Impossible F1 Scores in Published Paper")
        f1_df = load_result_csv("31_paper_f1_consistency.csv")
        if f1_df is not None:
            st.dataframe(f1_df, use_container_width=True)
            st.caption("Rows where 'F1 outside [min(P,R), max(P,R)]?' is True violate the mathematical definition of harmonic mean.")

# -----------------------------------------------------------------------------
# TAB 4: HYPERPARAMETER & FEATURE LAB
# -----------------------------------------------------------------------------
elif app_mode == "🔬 Hyperparameter & Feature Lab":
    st.subheader("🔬 Machine Learning Hyperparameter & Feature Exploration")
    
    lab_tabs = st.tabs([
        "📍 KNN k-Sweep & Distance Metrics",
        "🌲 Decision Tree & Purity Criteria",
        "🎯 Feature Importance & Selection",
        "🛡️ Isolation Forest Outlier Analysis"
    ])
    
    with lab_tabs[0]:
        st.markdown("##### 📍 KNN Performance vs k (Euclidean vs Manhattan)")
        knn_df = load_result_csv("11_knn_k_sweep.csv")
        if knn_df is not None:
            fig = px.line(
                knn_df,
                x="k",
                y=["acc_A", "acc_B", "acc_C"],
                color="metric",
                markers=True,
                title="KNN Test Accuracy Across Neighbors k (1 to 20)",
                labels={"value": "Accuracy", "variable": "Protocol"}
            )
            fig.add_vline(x=8, line_dash="dash", line_color="red", annotation_text="Paper's k=8")
            st.plotly_chart(fig, use_container_width=True)
            
        knn_sum = load_result_csv("12_knn_best_k_summary.csv")
        if knn_sum is not None:
            st.dataframe(knn_sum, use_container_width=True, hide_index=True)
            
    with lab_tabs[1]:
        st.markdown("##### 🌲 Decision Tree: Entropy (Information Gain) vs Gini Purity")
        crit_df = load_result_csv("07_dt_criterion_comparison.csv")
        if crit_df is not None:
            st.dataframe(crit_df, use_container_width=True)
            
        tree_img = ROOT / "results" / "fig_dt_tree_structure.png"
        if tree_img.exists():
            st.image(str(tree_img), caption="Decision Tree Structure (Top 3 Levels, Entropy)", use_container_width=True)

    with lab_tabs[2]:
        st.markdown("##### 🎯 Feature Ranking Across Methods")
        feat_df = load_result_csv("05_feature_selection_scores.csv")
        if feat_df is not None:
            st.dataframe(feat_df, use_container_width=True)
            
        col1, col2 = st.columns(2)
        with col1:
            rf_img = ROOT / "results" / "fig_rf_feature_importance.png"
            if rf_img.exists():
                st.image(str(rf_img), caption="Random Forest Impurity-based Feature Importance", use_container_width=True)
        with col2:
            feat_img = ROOT / "results" / "fig_feature_selection_scores.png"
            if feat_img.exists():
                st.image(str(feat_img), caption="Mutual Information & Chi-Square Feature Rankings", use_container_width=True)

    with lab_tabs[3]:
        st.markdown("##### 🛡️ Isolation Forest Anomaly Removal Impact")
        iso_df = load_result_csv("22_isolation_forest_test_accuracy.csv")
        if iso_df is not None:
            st.dataframe(iso_df, use_container_width=True)
            st.info("Notice: Removing outliers from the training set has no statistically significant positive effect on test performance, and whole-dataset filtering distorts the evaluation population.")

# -----------------------------------------------------------------------------
# TAB 5: DATASET & EDA GALLERY
# -----------------------------------------------------------------------------
elif app_mode == "📈 Dataset & EDA Gallery":
    st.subheader("📈 UCI Cleveland Heart Disease Exploratory Data Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Patients</div>
            <div class="metric-value">303</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Disease Present (1)</div>
            <div class="metric-value" style="color:#fb7185;">139 (45.9%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Healthy (0)</div>
            <div class="metric-value" style="color:#34d399;">164 (54.1%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Features</div>
            <div class="metric-value">13</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown("##### 🔍 Cleveland Heart Disease Dataset Viewer")
    st.dataframe(df_raw, use_container_width=True)
    
    st.markdown("##### 🖼️ Published Exploratory Visualizations Gallery")
    eda_images = [
        ("Correlation with Target", "fig02_correlation_with_target.png"),
        ("Class Distribution", "fig03_class_distribution.png"),
        ("Sex vs Target", "fig04_sex_vs_target.png"),
        ("Chest Pain Type vs Target", "fig05_cp_vs_target.png"),
        ("Fasting Blood Sugar vs Target", "fig06_fbs_vs_target.png"),
        ("Continuous Features Distribution", "fig09_continuous_features.png"),
        ("Binary & Nominal Features", "fig10_binary_nominal_features.png"),
        ("Ordinal Features", "fig11_ordinal_features.png")
    ]
    
    gallery_cols = st.columns(2)
    for idx, (title, img_name) in enumerate(eda_images):
        p = ROOT / "results" / img_name
        if p.exists():
            with gallery_cols[idx % 2]:
                st.image(str(p), caption=title, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
    CardioScan AI Replication Lab • Based on UCI Cleveland Heart Disease Dataset • Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)
