import polars as pl
import pandas as pd
from scipy.stats import kruskal, chi2_contingency
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
import os
import warnings
warnings.filterwarnings('ignore')

# OriginalDataSetLink:- https://www.kaggle.com/datasets/ak0212/eye-cancer-patient-records/data

# Load the dataset
df = pl.read_csv(
    r"C:\Users\Vatsal\Document\VSCodeFiles\PythonCode\Eye Cancer Patient Records Data Analysis\eye_cancer_patients.csv"
) 


df = df.with_columns(pl.col("Genetic_Markers").fill_null("HAND"))

df_pandas = df.to_pandas()

# --- Step 1: Univariate Analysis ---
print("\n=== Univariate Analysis ===")

# Age: Summary statistics and histogram
age_stats = df["Age"].describe()
print("\nAge Statistics:")
print(age_stats)

fig_age = px.histogram(df_pandas, x="Age", nbins=20, title="Distribution of Age")
fig_age.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_age.update_layout(showlegend=False, xaxis_title="Age", yaxis_title="Count")
fig_age.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_age.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_age.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_age.update_layout(title_x=0.5)
fig_age.write_html("age_distribution.html")
webbrowser.open('file://' + os.path.realpath("age_distribution.html"))

# Gender: Value counts and bar plot
gender_counts = df["Gender"].value_counts()
print("\nGender Distribution:")
print(gender_counts)

fig_gender = px.histogram(df_pandas, x="Gender", title="Gender Distribution")
fig_gender.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_gender.update_layout(showlegend=False, xaxis_title="Gender", yaxis_title="Count")
fig_gender.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_gender.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_gender.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_gender.update_layout(title_x=0.5)
fig_gender.write_html("gender_distribution.html")
webbrowser.open('file://' + os.path.realpath("gender_distribution.html"))

# Cancer Type: Value counts and bar plot
cancer_counts = df["Cancer_Type"].value_counts()
print("\nCancer Type Distribution:")
print(cancer_counts)

fig_cancer = px.histogram(df_pandas, x="Cancer_Type", title="Cancer Type Distribution")
fig_cancer.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_cancer.update_layout(showlegend=False, xaxis_title="Cancer Type", yaxis_title="Count")
fig_cancer.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_cancer.update_xaxes(showgrid=True, gridcolor='lightgray', tickangle=45)
fig_cancer.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_cancer.update_layout(title_x=0.5)
fig_cancer.write_html("cancer_type_distribution.html")
webbrowser.open('file://' + os.path.realpath("cancer_type_distribution.html"))

# Laterality: Value counts and bar plot
laterality_counts = df["Laterality"].value_counts()
print("\nLaterality Distribution:")
print(laterality_counts)

fig_laterality = px.histogram(df_pandas, x="Laterality", title="Laterality Distribution")
fig_laterality.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_laterality.update_layout(showlegend=False, xaxis_title="Laterality", yaxis_title="Count")
fig_laterality.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_laterality.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_laterality.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_laterality.update_layout(title_x=0.5)
fig_laterality.write_html("laterality_distribution.html")
webbrowser.open('file://' + os.path.realpath("laterality_distribution.html"))

# Stage at Diagnosis: Value counts and bar plot
stage_counts = df["Stage_at_Diagnosis"].value_counts()
print("\nStage at Diagnosis Distribution:")
print(stage_counts)

fig_stage = px.histogram(df_pandas, x="Stage_at_Diagnosis", title="Stage at Diagnosis Distribution",
                         category_orders={"Stage_at_Diagnosis": ["I", "II", "III", "IV"]})
fig_stage.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_stage.update_layout(showlegend=False, xaxis_title="Stage at Diagnosis", yaxis_title="Count")
fig_stage.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_stage.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_stage.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_stage.update_layout(title_x=0.5)
fig_stage.write_html("stage_distribution.html")
webbrowser.open('file://' + os.path.realpath("stage_distribution.html"))

# Outcome Status: Value counts and bar plot
outcome_counts = df["Outcome_Status"].value_counts()
print("\nOutcome Status Distribution:")
print(outcome_counts)

fig_outcome = px.histogram(df_pandas, x="Outcome_Status", title="Outcome Status Distribution")
fig_outcome.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_outcome.update_layout(showlegend=False, xaxis_title="Outcome Status", yaxis_title="Count")
fig_outcome.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_outcome.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_outcome.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_outcome.update_layout(title_x=0.5)
fig_outcome.write_html("outcome_distribution.html")
webbrowser.open('file://' + os.path.realpath("outcome_distribution.html"))

# Survival Time: Summary statistics and histogram
survival_stats = df["Survival_Time_Months"].describe()
print("\nSurvival Time (Months) Statistics:")
print(survival_stats)

fig_survival = px.histogram(df_pandas, x="Survival_Time_Months", nbins=20, title="Distribution of Survival Time (Months)")
fig_survival.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_survival.update_layout(showlegend=False, xaxis_title="Survival Time (Months)", yaxis_title="Count")
fig_survival.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_survival.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_survival.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_survival.update_layout(title_x=0.5)
fig_survival.write_html("survival_time_distribution.html")
webbrowser.open('file://' + os.path.realpath("survival_time_distribution.html"))

# Country: Value counts (top 10) and bar plot
country_counts = df["Country"].value_counts().head(10)
print("\nTop 10 Countries by Patient Count:")
print(country_counts)

fig_country = px.bar(country_counts.to_pandas(), x="count", y="Country", title="Top 10 Countries by Patient Count")
fig_country.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_country.update_layout(xaxis_title="Count", yaxis_title="Country")
fig_country.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_country.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_country.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_country.update_layout(title_x=0.5)
fig_country.write_html("country_distribution.html")
webbrowser.open('file://' + os.path.realpath("country_distribution.html"))

# Genetic Markers: Value counts and bar plot
genetic_counts = df["Genetic_Markers"].value_counts()
print("\nGenetic Markers Distribution:")
print(genetic_counts)

fig_genetic = px.histogram(df_pandas, x="Genetic_Markers", title="Genetic Markers Distribution")
fig_genetic.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_genetic.update_layout(showlegend=False, xaxis_title="Genetic Markers", yaxis_title="Count")
fig_genetic.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_genetic.update_xaxes(showgrid=True, gridcolor='lightgray', tickangle=45)
fig_genetic.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_genetic.update_layout(title_x=0.5)
fig_genetic.write_html("genetic_markers_distribution.html")
webbrowser.open('file://' + os.path.realpath("genetic_markers_distribution.html"))

# Family History: Value counts and bar plot
family_history_counts = df["Family_History"].value_counts()
print("\nFamily History Distribution:")
print(family_history_counts)

fig_family = px.histogram(df_pandas, x="Family_History", title="Family History Distribution")
fig_family.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_family.update_layout(showlegend=False, xaxis_title="Family History", yaxis_title="Count")
fig_family.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_family.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_family.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_family.update_layout(title_x=0.5)
fig_family.write_html("family_history_distribution.html")
webbrowser.open('file://' + os.path.realpath("family_history_distribution.html"))

# --- Step 2: Bivariate Analysis ---
print("\n=== Bivariate Analysis ===")

# 2.1 Age by Cancer Type and Kruskal-Wallis test
age_by_cancer = df.group_by("Cancer_Type").agg(
    mean_age=pl.col("Age").mean(),
    median_age=pl.col("Age").median(),
    std_age=pl.col("Age").std(),
)
print("\nAge Statistics by Cancer Type:")
print(age_by_cancer)

fig_age_cancer = px.box(df_pandas, x="Cancer_Type", y="Age", title="Age Distribution by Cancer Type")
fig_age_cancer.update_layout(xaxis_title="Cancer Type", yaxis_title="Age")
fig_age_cancer.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_age_cancer.update_xaxes(showgrid=True, gridcolor='lightgray', tickangle=45)
fig_age_cancer.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_age_cancer.update_layout(title_x=0.5)
fig_age_cancer.write_html("age_by_cancer.html")
webbrowser.open('file://' + os.path.realpath("age_by_cancer.html"))

# Kruskal-Wallis test for Age by Cancer Type
age_groups = [df_pandas[df_pandas["Cancer_Type"] == ct]["Age"] for ct in df_pandas["Cancer_Type"].unique()]
stat, p = kruskal(*age_groups)
print(f"\nKruskal-Wallis Test for Age by Cancer Type: stat={stat:.2f}, p={p:.4f}")

# 2.2 Treatment Type vs. Outcome Status and Chi-Square test
treatment_outcome_crosstab = pd.crosstab(df_pandas["Treatment_Type"], df_pandas["Outcome_Status"])
print("\nTreatment Type vs. Outcome Status Crosstab:")
print(treatment_outcome_crosstab)

fig_treatment_outcome = go.Figure(data=go.Heatmap(
    z=treatment_outcome_crosstab.values,
    x=treatment_outcome_crosstab.columns,
    y=treatment_outcome_crosstab.index,
    text=treatment_outcome_crosstab.values,
    texttemplate="%{text}",
    colorscale="Blues"
))
fig_treatment_outcome.update_layout(title="Treatment Type vs. Outcome Status",
                                   xaxis_title="Outcome Status",
                                   yaxis_title="Treatment Type",
                                   plot_bgcolor='white', paper_bgcolor='white',
                                   title_x=0.5)
fig_treatment_outcome.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_treatment_outcome.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_treatment_outcome.write_html("treatment_outcome_heatmap.html")
webbrowser.open('file://' + os.path.realpath("treatment_outcome_heatmap.html"))

# Chi-Square test
chi2, p, _, _ = chi2_contingency(treatment_outcome_crosstab)
print(f"\nChi-Square Test for Treatment Type vs. Outcome Status: chi2={chi2:.2f}, p={p:.4f}")

# 2.3 Genetic Markers vs. Cancer Type and Chi-Square test
genetic_cancer_crosstab = pd.crosstab(df_pandas["Genetic_Markers"], df_pandas["Cancer_Type"])
print("\nGenetic Markers vs. Cancer Type Crosstab:")
print(genetic_cancer_crosstab)

fig_genetic_cancer = go.Figure(data=go.Heatmap(
    z=genetic_cancer_crosstab.values,
    x=genetic_cancer_crosstab.columns,
    y=genetic_cancer_crosstab.index,
    text=genetic_cancer_crosstab.values,
    texttemplate="%{text}",
    colorscale="Blues"
))
fig_genetic_cancer.update_layout(title="Genetic Markers vs. Cancer Type",
                                xaxis_title="Cancer Type",
                                yaxis_title="Genetic Markers",
                                plot_bgcolor='white', paper_bgcolor='white',
                                title_x=0.5)
fig_genetic_cancer.update_xaxes(showgrid=True, gridcolor='lightgray', tickangle=45)
fig_genetic_cancer.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_genetic_cancer.write_html("genetic_cancer_heatmap.html")
webbrowser.open('file://' + os.path.realpath("genetic_cancer_heatmap.html"))

# Chi-Square test
chi2, p, _, _ = chi2_contingency(genetic_cancer_crosstab)
print(f"\nChi-Square Test for Genetic Markers vs. Cancer Type: chi2={chi2:.2f}, p={p:.4f}")

# 2.4 Family History vs. Outcome Status and Chi-Square test
family_outcome_crosstab = pd.crosstab(df_pandas["Family_History"], df_pandas["Outcome_Status"])
print("\nFamily History vs. Outcome Status Crosstab:")
print(family_outcome_crosstab)

fig_family_outcome = go.Figure(data=go.Heatmap(
    z=family_outcome_crosstab.values,
    x=family_outcome_crosstab.columns,
    y=family_outcome_crosstab.index,
    text=family_outcome_crosstab.values,
    texttemplate="%{text}",
    colorscale="Blues"
))
fig_family_outcome.update_layout(title="Family History vs. Outcome Status",
                                xaxis_title="Outcome Status",
                                yaxis_title="Family History",
                                plot_bgcolor='white', paper_bgcolor='white',
                                title_x=0.5)
fig_family_outcome.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_family_outcome.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_family_outcome.write_html("family_outcome_heatmap.html")
webbrowser.open('file://' + os.path.realpath("family_outcome_heatmap.html"))

# Chi-Square test
chi2, p, _, _ = chi2_contingency(family_outcome_crosstab)
print(f"\nChi-Square Test for Family History vs. Outcome Status: chi2={chi2:.2f}, p={p:.4f}")

# 2.5 Stage at Diagnosis vs. Outcome Status and Chi-Square test
stage_outcome_crosstab = pd.crosstab(df_pandas["Stage_at_Diagnosis"], df_pandas["Outcome_Status"])
print("\nStage at Diagnosis vs. Outcome Status Crosstab:")
print(stage_outcome_crosstab)

fig_stage_outcome = go.Figure(data=go.Heatmap(
    z=stage_outcome_crosstab.values,
    x=stage_outcome_crosstab.columns,
    y=stage_outcome_crosstab.index,
    text=stage_outcome_crosstab.values,
    texttemplate="%{text}",
    colorscale="Blues"
))
fig_stage_outcome.update_layout(title="Stage at Diagnosis vs. Outcome Status",
                               xaxis_title="Outcome Status",
                               yaxis_title="Stage at Diagnosis",
                               plot_bgcolor='white', paper_bgcolor='white',
                               title_x=0.5)
fig_stage_outcome.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_stage_outcome.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_stage_outcome.write_html("stage_outcome_heatmap.html")
webbrowser.open('file://' + os.path.realpath("stage_outcome_heatmap.html"))

# Chi-Square test
chi2, p, _, _ = chi2_contingency(stage_outcome_crosstab)
print(f"\nChi-Square Test for Stage at Diagnosis vs. Outcome Status: chi2={chi2:.2f}, p={p:.4f}")

# --- Step 4: Survival Analysis ---
print("\n=== Survival Analysis ===")

# Kaplan-Meier Survival Estimates by Cancer Type
fig_km = go.Figure()
kmf = KaplanMeierFitter()
for cancer_type in df_pandas["Cancer_Type"].unique():
    mask = df_pandas["Cancer_Type"] == cancer_type
    kmf.fit(
        df_pandas[mask]["Survival_Time_Months"],
        df_pandas[mask]["Outcome_Status"] == "Deceased",
        label=cancer_type,
    )
    survival_df = kmf.survival_function_.reset_index()
    fig_km.add_trace(go.Scatter(
        x=survival_df["timeline"],
        y=survival_df[cancer_type],
        mode='lines',
        name=cancer_type,
        line=dict(width=2)
    ))
    print(f"\nSurvival Analysis for {cancer_type}:")
    print(f"Median Survival Time: {kmf.median_survival_time_:.2f} months")
    survival_prob = kmf.survival_function_at_times([12, 24, 60, 120]).to_dict()
    print(f"Survival Probabilities at 12, 24, 60, 120 months: {survival_prob}")

fig_km.update_layout(title="Kaplan-Meier Survival Curves by Cancer Type",
                     xaxis_title="Time (Months)",
                     yaxis_title="Survival Probability",
                     plot_bgcolor='white', paper_bgcolor='white',
                     title_x=0.5)
fig_km.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_km.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_km.write_html("km_survival_curves.html")
webbrowser.open('file://' + os.path.realpath("km_survival_curves.html"))

# --- Step 5: Geo-Demographic Trends ---
print("\n=== Geo-Demographic Trends ===")

# Average Survival Time and Outcome Proportions by Country (Top 10)
country_stats = (
    df.group_by("Country")
    .agg(
        patient_count=pl.col("Patient_ID").count(),
        avg_survival=pl.col("Survival_Time_Months").mean(),
        remission_rate=pl.col("Outcome_Status").eq("In Remission").mean(),
        deceased_rate=pl.col("Outcome_Status").eq("Deceased").mean(),
    )
    .sort("patient_count", descending=True)
    .head(10)
)
print("\nCountry Statistics (Top 10 by Patient Count):")
print(country_stats)

fig_country_stats = px.bar(country_stats.to_pandas(), x="avg_survival", y="Country",
                           title="Average Survival Time by Country (Top 10)")
fig_country_stats.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_country_stats.update_layout(xaxis_title="Average Survival Time (Months)", yaxis_title="Country")
fig_country_stats.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_country_stats.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_country_stats.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_country_stats.update_layout(title_x=0.5)
fig_country_stats.write_html("survival_by_country.html")
webbrowser.open('file://' + os.path.realpath("survival_by_country.html"))

# --- Step 1: Interaction Effects ---
print("\n=== Interaction Effects ===")

# 1.1 Age and Stage at Diagnosis vs. Outcome Status
df_pandas["Age_Group"] = pd.cut(
    df_pandas["Age"],
    bins=[0, 18, 40, 60, 90],
    labels=["Child", "Young Adult", "Adult", "Senior"],
)
age_stage_outcome = pd.crosstab(
    index=[df_pandas["Age_Group"], df_pandas["Stage_at_Diagnosis"]],
    columns=df_pandas["Outcome_Status"],
)
print("\nAge Group and Stage vs. Outcome Status Crosstab:")
print(age_stage_outcome)

fig_age_stage_outcome = go.Figure(data=go.Heatmap(
    z=age_stage_outcome.values,
    x=age_stage_outcome.columns,
    y=[f"{idx[0]}_{idx[1]}" for idx in age_stage_outcome.index],
    text=age_stage_outcome.values,
    texttemplate="%{text}",
    colorscale="Blues"
))
fig_age_stage_outcome.update_layout(title="Age Group and Stage vs. Outcome Status",
                                   xaxis_title="Outcome Status",
                                   yaxis_title="Age Group_Stage",
                                   plot_bgcolor='white', paper_bgcolor='white',
                                   title_x=0.5)
fig_age_stage_outcome.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_age_stage_outcome.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_age_stage_outcome.write_html("age_stage_outcome_heatmap.html")
webbrowser.open('file://' + os.path.realpath("age_stage_outcome_heatmap.html"))

# 1.2 Treatment Type and Genetic Markers vs. Outcome
treatment_genetic_outcome = pd.crosstab(
    index=[df_pandas["Treatment_Type"], df_pandas["Genetic_Markers"]],
    columns=df_pandas["Outcome_Status"],
)
print("\nTreatment Type and Genetic Markers vs. Outcome Status Crosstab:")
print(treatment_genetic_outcome)

fig_treatment_genetic = go.Figure(data=go.Heatmap(
    z=treatment_genetic_outcome.values,
    x=treatment_genetic_outcome.columns,
    y=[f"{idx[0]}_{idx[1]}" for idx in treatment_genetic_outcome.index],
    text=treatment_genetic_outcome.values,
    texttemplate="%{text}",
    colorscale="Blues"
))
fig_treatment_genetic.update_layout(title="Treatment Type and Genetic Markers vs. Outcome Status",
                                   xaxis_title="Outcome Status",
                                   yaxis_title="Treatment Type_Genetic Markers",
                                   plot_bgcolor='white', paper_bgcolor='white',
                                   title_x=0.5)
fig_treatment_genetic.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_treatment_genetic.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_treatment_genetic.write_html("treatment_genetic_outcome_heatmap.html")
webbrowser.open('file://' + os.path.realpath("treatment_genetic_outcome_heatmap.html"))

# --- Step 2: Time-Based Analysis ---
print("\n=== Time-Based Analysis ===")

# Extract diagnosis year
df = df.with_columns(
    pl.col("Date_of_Diagnosis").str.to_date().dt.year().alias("Diagnosis_Year")
)
df_pandas = df.to_pandas()

# Survival Time and Outcome by Diagnosis Year
year_stats = (
    df.group_by("Diagnosis_Year")
    .agg(
        patient_count=pl.col("Patient_ID").count(),
        avg_survival=pl.col("Survival_Time_Months").mean(),
        remission_rate=pl.col("Outcome_Status").eq("In Remission").mean(),
        deceased_rate=pl.col("Outcome_Status").eq("Deceased").mean(),
    )
    .sort("Diagnosis_Year")
)
print("\nSurvival and Outcome Statistics by Diagnosis Year:")
print(year_stats)

fig_year_stats = px.line(year_stats.to_pandas(), x="Diagnosis_Year", y="avg_survival",
                         title="Average Survival Time by Diagnosis Year", markers=True)
fig_year_stats.update_traces(line=dict(width=2))
fig_year_stats.update_layout(xaxis_title="Diagnosis Year", yaxis_title="Average Survival Time (Months)")
fig_year_stats.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_year_stats.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_year_stats.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_year_stats.update_layout(title_x=0.5)
fig_year_stats.write_html("survival_by_year.html")
webbrowser.open('file://' + os.path.realpath("survival_by_year.html"))

# --- Step 3: Clustering Analysis ---
print("\n=== Clustering Analysis ===")

# Prepare data for clustering
cluster_df = df_pandas[
    ["Age", "Survival_Time_Months", "Radiation_Therapy", "Chemotherapy"]
]
for col in ["Cancer_Type", "Stage_at_Diagnosis", "Genetic_Markers"]:
    cluster_df[col] = LabelEncoder().fit_transform(df_pandas[col])

# K-means clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_df["Cluster"] = kmeans.fit_predict(cluster_df)
df_pandas["Cluster"] = cluster_df["Cluster"]

# Summarize clusters
cluster_summary = df_pandas.groupby("Cluster").agg(
    {
        "Age": ["mean", "std"],
        "Survival_Time_Months": ["mean", "std"],
        "Radiation_Therapy": ["mean"],
        "Chemotherapy": ["mean"],
        "Cancer_Type": lambda x: x.mode()[0],
        "Outcome_Status": lambda x: x.mode()[0],
    }
)
print("\nCluster Summary:")
print(cluster_summary)

# Visualize clusters (Age vs. Survival Time)
fig_cluster = px.scatter(df_pandas, x="Age", y="Survival_Time_Months", color="Cluster",
                         title="Clusters: Age vs. Survival Time")
fig_cluster.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_cluster.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_cluster.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_cluster.update_layout(title_x=0.5)
fig_cluster.write_html("cluster_scatter.html")
webbrowser.open('file://' + os.path.realpath("cluster_scatter.html"))

# --- Step 4: Feature Importance for Outcome Prediction ---
print("\n=== Feature Importance for Outcome Prediction ===")

# Prepare data for Random Forest
X = df_pandas[
    [
        "Age",
        "Cancer_Type",
        "Stage_at_Diagnosis",
        "Treatment_Type",
        "Radiation_Therapy",
        "Chemotherapy",
        "Genetic_Markers",
        "Family_History",
    ]
]
for col in ["Cancer_Type", "Stage_at_Diagnosis", "Treatment_Type", "Genetic_Markers"]:
    X[col] = LabelEncoder().fit_transform(X[col])
y = df_pandas["Outcome_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf.feature_importances_}
).sort_values("Importance", ascending=False)
print("\nRandom Forest Feature Importance:")
print(feature_importance)

fig_feature = px.bar(feature_importance, x="Importance", y="Feature", title="Random Forest Feature Importance")
fig_feature.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_feature.update_layout(xaxis_title="Importance", yaxis_title="Feature")
fig_feature.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_feature.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_feature.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_feature.update_layout(title_x=0.5)
fig_feature.write_html("feature_importance.html")
webbrowser.open('file://' + os.path.realpath("feature_importance.html"))

# Classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf.predict(X_test)))

# --- Step 5: Treatment Intensity Patterns ---
print("\n=== Treatment Intensity Patterns ===")

# Calculate total treatment intensity
df_pandas["Total_Treatment_Intensity"] = (
    df_pandas["Radiation_Therapy"] + df_pandas["Chemotherapy"]
)

# Treatment Intensity by Cancer Type and Stage
intensity_by_cancer_stage = df_pandas.groupby(["Cancer_Type", "Stage_at_Diagnosis"])[
    "Total_Treatment_Intensity"
].agg(["mean", "std", "count"]).reset_index()
print("\nTotal Treatment Intensity by Cancer Type and Stage:")
print(intensity_by_cancer_stage)

fig_intensity = px.bar(intensity_by_cancer_stage, x="Stage_at_Diagnosis", y="mean", color="Cancer_Type",
                       title="Treatment Intensity by Cancer Type and Stage", barmode="group")
fig_intensity.update_traces(marker=dict(line=dict(color='black', width=1)))
fig_intensity.update_layout(xaxis_title="Stage at Diagnosis", yaxis_title="Mean Treatment Intensity")
fig_intensity.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_intensity.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_intensity.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_intensity.update_layout(title_x=0.5)
fig_intensity.write_html("treatment_intensity.html")
webbrowser.open('file://' + os.path.realpath("treatment_intensity.html"))

# --- Step 6: Survival Disparities by Laterality ---
print("\n=== Survival Disparities by Laterality ===")

# Survival Time by Laterality
survival_by_laterality = df.group_by("Laterality").agg(
    mean_survival=pl.col("Survival_Time_Months").mean(),
    median_survival=pl.col("Survival_Time_Months").median(),
    remission_rate=pl.col("Outcome_Status").eq("In Remission").mean(),
    deceased_rate=pl.col("Outcome_Status").eq("Deceased").mean(),
)
print("\nSurvival and Outcome Statistics by Laterality:")
print(survival_by_laterality)

fig_laterality = px.box(df_pandas, x="Laterality", y="Survival_Time_Months", title="Survival Time by Laterality")
fig_laterality.update_layout(xaxis_title="Laterality", yaxis_title="Survival Time (Months)")
fig_laterality.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_laterality.update_xaxes(showgrid=True, gridcolor='lightgray')
fig_laterality.update_yaxes(showgrid=True, gridcolor='lightgray')
fig_laterality.update_layout(title_x=0.5)
fig_laterality.write_html("survival_by_laterality.html")
webbrowser.open('file://' + os.path.realpath("survival_by_laterality.html"))

# Cox model for Laterality
cox_df = df_pandas.copy()
cox_df["Laterality"] = LabelEncoder().fit_transform(cox_df["Laterality"])
cox_df["Outcome_Event"] = (cox_df["Outcome_Status"] == "Deceased").astype(int)
cox = CoxPHFitter()
cox.fit(
    cox_df[["Laterality", "Age", "Survival_Time_Months", "Outcome_Event"]],
    duration_col="Survival_Time_Months",
    event_col="Outcome_Event",
)
print("\nCox Model for Laterality Impact:")
cox.print_summary(columns=["coef", "exp(coef)", "p"], decimals=4)