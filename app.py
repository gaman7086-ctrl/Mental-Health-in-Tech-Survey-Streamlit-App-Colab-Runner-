"""
Mental Health in Tech Survey — Streamlit App
Interactive EDA dashboard + a simple "will this person likely seek treatment"
predictor, built on top of the OSMI 2014 Mental Health in Tech Survey dataset.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health in Tech Survey",
    page_icon="🧠",
    layout="wide",
)

DATA_PATH = "survey.csv"


# ----------------------------------------------------------------------------
# Data loading & cleaning (same logic as the EDA notebook)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Age: keep only plausible working-age values, fix the rest to the median
    valid_age_mask = df["Age"].between(18, 75)
    median_age = df.loc[valid_age_mask, "Age"].median()
    df.loc[~valid_age_mask, "Age"] = median_age
    df["Age"] = df["Age"].astype(int)

    # Gender: consolidate free-text spellings into Male / Female / Other
    def clean_gender(g):
        g = str(g).strip().lower()
        male_terms = [
            "male", "m", "man", "cis male", "cis man", "mail", "malr", "maile",
            "make", "guy (-ish) ^_^", "male (cis)", "male-ish", "msle", "mal",
            "male leaning androgynous", "ostensibly male, unsure what that really means",
        ]
        female_terms = [
            "female", "f", "woman", "cis female", "femake", "female (cis)",
            "cis-female/femme", "female (trans)", "trans woman", "trans-female",
        ]
        if g in male_terms:
            return "Male"
        elif g in female_terms:
            return "Female"
        return "Other"

    df["Gender"] = df["Gender"].apply(clean_gender)

    # work_interfere: missing likely means "no condition to interfere" -> own category
    df["work_interfere"] = df["work_interfere"].fillna("Not applicable")

    # self_employed: fill missing with mode
    df["self_employed"] = df["self_employed"].fillna(df["self_employed"].mode()[0])

    # Drop columns not used in this app
    df = df.drop(columns=["state", "comments", "Timestamp"], errors="ignore")

    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a simple RandomForest to predict `treatment` from a handful of
    the most informative survey answers. Cached so it only trains once."""
    feature_cols = [
        "Age", "Gender", "family_history", "work_interfere", "no_employees",
        "remote_work", "benefits", "care_options", "wellness_program",
        "seek_help", "anonymity", "leave", "obs_consequence",
    ]
    model_df = df[feature_cols + ["treatment"]].copy()

    encoders = {}
    for col in feature_cols:
        if col != "Age":
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            encoders[col] = le

    target_le = LabelEncoder()
    model_df["treatment"] = target_le.fit_transform(model_df["treatment"].astype(str))

    X = model_df[feature_cols]
    y = model_df["treatment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    return clf, encoders, target_le, feature_cols, test_acc


df = load_data(DATA_PATH)

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🧠 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "EDA Dashboard", "Predict Treatment Likelihood", "About"],
)

# ============================================================================
# HOME
# ============================================================================
if page == "Home":
    st.title("🧠 Mental Health in Tech Survey")
    st.markdown(
        "Exploring attitudes toward mental health and workplace support in the "
        "tech industry, based on the **OSMI 2014 Mental Health in Tech Survey** "
        "(1,259 responses, 27 questions)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Respondents", f"{len(df):,}")
    col2.metric("Countries", df["Country"].nunique())
    treated_pct = (df["treatment"] == "Yes").mean() * 100
    col3.metric("Sought Treatment", f"{treated_pct:.1f}%")
    fam_hist_pct = (df["family_history"] == "Yes").mean() * 100
    col4.metric("Family History", f"{fam_hist_pct:.1f}%")

    st.markdown("### Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.info(
        "Use the sidebar to explore an **interactive EDA dashboard**, or try the "
        "**treatment-likelihood predictor**, built on the patterns found in this survey."
    )

# ============================================================================
# EDA DASHBOARD
# ============================================================================
elif page == "EDA Dashboard":
    st.title("📊 Interactive EDA Dashboard")

    # ---- Filters ----
    st.sidebar.markdown("### Filters")
    top_countries = df["Country"].value_counts().head(15).index.tolist()
    country_filter = st.sidebar.multiselect(
        "Country (top 15 shown)", options=top_countries, default=top_countries[:5]
    )
    gender_filter = st.sidebar.multiselect(
        "Gender", options=sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique())
    )

    filtered = df.copy()
    if country_filter:
        filtered = filtered[filtered["Country"].isin(country_filter)]
    if gender_filter:
        filtered = filtered[filtered["Gender"].isin(gender_filter)]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} respondents based on current filters.")

    if len(filtered) == 0:
        st.warning("No rows match the current filters. Adjust the filters in the sidebar.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Univariate", "Bivariate vs Treatment", "Multivariate"])

    # ---- Univariate ----
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(filtered, x="Age", nbins=20, title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(filtered, names="treatment", title="Sought Treatment?", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(filtered, x="Gender", title="Gender Distribution", color="Gender")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            order = ["Never", "Rarely", "Sometimes", "Often", "Not applicable"]
            fig = px.histogram(
                filtered, x="work_interfere", category_orders={"work_interfere": order},
                title="Work Interference Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Bivariate vs treatment ----
    with tab2:
        bivariate_options = {
            "family_history": "Family History",
            "benefits": "Employer Benefits",
            "care_options": "Care Options Awareness",
            "Gender": "Gender",
            "no_employees": "Company Size",
            "leave": "Ease of Taking Leave",
            "remote_work": "Remote Work",
            "obs_consequence": "Observed Consequences for Coworkers",
        }
        choice = st.selectbox(
            "Compare treatment rate by:", options=list(bivariate_options.keys()),
            format_func=lambda x: bivariate_options[x],
        )
        rate = (
            pd.crosstab(filtered[choice], filtered["treatment"], normalize="index") * 100
        ).reset_index()
        if "Yes" not in rate.columns:
            rate["Yes"] = 0.0
        fig = px.bar(
            rate, x=choice, y="Yes",
            title=f"Treatment Rate (%) by {bivariate_options[choice]}",
            labels={"Yes": "% Sought Treatment"},
            color="Yes", color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)

        counts = filtered.groupby([choice, "treatment"]).size().reset_index(name="Count")
        fig2 = px.bar(
            counts, x=choice, y="Count", color="treatment", barmode="group",
            title=f"Respondent Counts by {bivariate_options[choice]} and Treatment",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Multivariate ----
    with tab3:
        st.markdown("#### Correlation Heatmap (label-encoded key variables)")
        corr_cols = [
            "Age", "family_history", "treatment", "benefits", "care_options",
            "wellness_program", "seek_help", "anonymity", "obs_consequence",
            "remote_work",
        ]
        corr_df = filtered[corr_cols].copy()
        for c in corr_cols:
            if c != "Age":
                corr_df[c] = LabelEncoder().fit_transform(corr_df[c].astype(str))
        fig = px.imshow(
            corr_df.corr().round(2), text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Company Size vs Work Interference")
        order_size = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
        ct = pd.crosstab(filtered["no_employees"], filtered["work_interfere"])
        ct = ct.reindex(index=[s for s in order_size if s in ct.index])
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PREDICTION
# ============================================================================
elif page == "Predict Treatment Likelihood":
    st.title("🔮 Predict Treatment-Seeking Likelihood")
    st.markdown(
        "This trains a small **Random Forest** model on the survey to estimate how likely "
        "someone with a given profile is to have sought mental-health treatment, based on "
        "the patterns in this dataset."
    )
    st.warning(
        "⚠️ This is an educational demo built on a 2014 survey sample, **not** a diagnostic "
        "or clinical tool. It should not be used to make decisions about any real person."
    )

    clf, encoders, target_le, feature_cols, test_acc = train_model(df)
    st.caption(f"Model hold-out accuracy: **{test_acc * 100:.1f}%** (RandomForest, 80/20 split)")

    st.markdown("#### Enter a profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 70, 30)
        gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
        family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
        work_interfere = st.selectbox(
            "If applicable, how often does it interfere with work?",
            ["Not applicable", "Never", "Rarely", "Sometimes", "Often"],
        )
    with c2:
        no_employees = st.selectbox(
            "Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
        )
        remote_work = st.selectbox("Works remotely ≥50% of the time?", ["Yes", "No"])
        benefits = st.selectbox("Employer provides mental health benefits?", ["Yes", "No", "Don't know"])
        care_options = st.selectbox("Aware of employer's care options?", ["Yes", "No", "Not sure"])
    with c3:
        wellness_program = st.selectbox("Wellness program discusses mental health?", ["Yes", "No", "Don't know"])
        seek_help = st.selectbox("Employer provides resources to seek help?", ["Yes", "No", "Don't know"])
        anonymity = st.selectbox("Anonymity protected if seeking help?", ["Yes", "No", "Don't know"])
        leave = st.selectbox(
            "Ease of taking medical leave",
            ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"],
        )
    obs_consequence = st.selectbox(
        "Observed negative consequences for a coworker who disclosed a mental health issue?",
        ["Yes", "No"],
    )

    if st.button("Predict", type="primary"):
        input_dict = {
            "Age": age, "Gender": gender, "family_history": family_history,
            "work_interfere": work_interfere, "no_employees": no_employees,
            "remote_work": remote_work, "benefits": benefits, "care_options": care_options,
            "wellness_program": wellness_program, "seek_help": seek_help,
            "anonymity": anonymity, "leave": leave, "obs_consequence": obs_consequence,
        }
        row = pd.DataFrame([input_dict])[feature_cols]
        for col in feature_cols:
            if col != "Age":
                le = encoders[col]
                val = row[col].iloc[0]
                if val not in le.classes_:
                    # Unseen category fallback -> use the most frequent training class
                    val = le.classes_[0]
                row[col] = le.transform([val])

        proba = clf.predict_proba(row)[0]
        classes = target_le.inverse_transform(clf.classes_)
        prob_yes = proba[list(classes).index("Yes")]

        st.markdown("### Result")
        st.progress(min(max(prob_yes, 0.0), 1.0))
        st.metric("Estimated likelihood of having sought treatment", f"{prob_yes * 100:.1f}%")

        if prob_yes >= 0.6:
            st.success(
                "This profile matches respondents who **frequently sought treatment** in the survey — "
                "most often driven by family history and clear awareness of care options."
            )
        elif prob_yes <= 0.4:
            st.info(
                "This profile matches respondents who **less often sought treatment** in the survey — "
                "often associated with unclear or absent employer benefits/care-option communication."
            )
        else:
            st.info("This profile falls in a mixed zone — the survey shows a roughly even split for similar respondents.")

# ============================================================================
# ABOUT
# ============================================================================
else:
    st.title("ℹ️ About this app")
    st.markdown(
        """
This app accompanies an Exploratory Data Analysis capstone project on the
**OSMI Mental Health in Tech Survey (2014)**.

**Pages**
- **Home** — dataset overview and quick preview
- **EDA Dashboard** — interactive charts (univariate, bivariate vs. treatment, and multivariate),
  filterable by country and gender
- **Predict Treatment Likelihood** — a small RandomForest classifier trained live on the survey
  data, to illustrate how the identified factors (family history, benefits awareness, care-options
  awareness, work interference, etc.) combine to relate to treatment-seeking behavior
- **About** — this page

**Data source:** Open Sourcing Mental Illness (OSMI), 2014 Mental Health in Tech Survey.

**Disclaimer:** This is an educational/analytical demo built on a single, self-selected,
2014 survey sample (skewed toward male, U.S./U.K. respondents). It is **not** a diagnostic
tool and should not be used to make decisions about, or draw conclusions about, any real
individual.
"""
    )
