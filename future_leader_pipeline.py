"""
Identifying Future Leaders Through Data Analytics

Run:
    python future_leader_pipeline.py

Outputs:
    data/future_leaders_dataset.csv
    plots/*.png
    models/future_leader_model.joblib
    models/model_metadata.joblib
"""

from pathlib import Path
import os
import warnings

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
MPL_CACHE_DIR = BASE_DIR / ".matplotlib"

for folder in [DATA_DIR, PLOTS_DIR, MODELS_DIR, MPL_CACHE_DIR]:
    folder.mkdir(exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


FEATURE_COLUMNS = [
    "department",
    "job_level",
    "education",
    "performance_score",
    "engagement_score",
    "years_experience",
    "years_at_company",
    "leadership_trait_score",
    "peer_feedback_score",
    "manager_feedback_score",
    "innovation_score",
    "learning_hours",
    "projects_led",
    "promotion_last_3yrs",
    "absenteeism_days",
    "attrition_risk_score",
]

NUMERIC_FEATURES = [
    "performance_score",
    "engagement_score",
    "years_experience",
    "years_at_company",
    "leadership_trait_score",
    "peer_feedback_score",
    "manager_feedback_score",
    "innovation_score",
    "learning_hours",
    "projects_led",
    "promotion_last_3yrs",
    "absenteeism_days",
    "attrition_risk_score",
]

CATEGORICAL_FEATURES = ["department", "job_level", "education"]
TARGET = "future_leader"


def objective_summary() -> None:
    print("\n1. Problem Understanding")
    print(
        "Objective: identify employees with high future leadership potential by combining "
        "performance, engagement, experience, feedback, learning, innovation, and leadership indicators."
    )


def feature_design_summary() -> None:
    print("\n2. Feature Design")
    print("Features used:")
    for feature in FEATURE_COLUMNS:
        print(f" - {feature}")


def generate_synthetic_employee_data(n_records: int = 750, random_state: int = 42) -> pd.DataFrame:
    """Create a realistic employee dataset with a signal-rich future_leader target."""
    rng = np.random.default_rng(random_state)

    departments = rng.choice(
        ["Sales", "Engineering", "HR", "Finance", "Operations", "Marketing", "Product"],
        size=n_records,
        p=[0.16, 0.22, 0.10, 0.12, 0.16, 0.12, 0.12],
    )
    job_levels = rng.choice(["Junior", "Mid", "Senior", "Lead", "Manager"], size=n_records, p=[0.25, 0.34, 0.22, 0.12, 0.07])
    education = rng.choice(["Bachelor", "Master", "PhD", "Diploma"], size=n_records, p=[0.50, 0.32, 0.08, 0.10])

    level_experience_boost = pd.Series(job_levels).map({"Junior": 0, "Mid": 3, "Senior": 6, "Lead": 8, "Manager": 10}).to_numpy()
    years_experience = np.clip(rng.normal(3 + level_experience_boost, 2.2), 0, 25).round(1)
    years_at_company = np.clip(years_experience * rng.uniform(0.20, 0.85, n_records), 0, 18).round(1)

    performance_score = np.clip(rng.normal(72, 11, n_records), 35, 99).round(1)
    engagement_score = np.clip(rng.normal(70, 13, n_records), 25, 99).round(1)
    leadership_trait_score = np.clip(rng.normal(68, 14, n_records), 20, 99).round(1)
    peer_feedback_score = np.clip(rng.normal(3.7, 0.65, n_records), 1, 5).round(2)
    manager_feedback_score = np.clip(rng.normal(3.75, 0.7, n_records), 1, 5).round(2)
    innovation_score = np.clip(rng.normal(65, 16, n_records), 10, 99).round(1)
    learning_hours = np.clip(rng.gamma(shape=4.0, scale=8.0, size=n_records), 0, 120).round(1)
    projects_led = np.clip(rng.poisson(lam=1.2 + (level_experience_boost / 5)), 0, 12)
    promotion_last_3yrs = rng.binomial(1, p=np.clip(0.08 + performance_score / 500 + leadership_trait_score / 700, 0.05, 0.55))
    absenteeism_days = np.clip(rng.poisson(lam=5.5), 0, 25)
    attrition_risk_score = np.clip(rng.normal(45, 18, n_records) - engagement_score * 0.18, 1, 99).round(1)

    df = pd.DataFrame(
        {
            "employee_id": [f"EMP{10000 + i}" for i in range(n_records)],
            "department": departments,
            "job_level": job_levels,
            "education": education,
            "performance_score": performance_score,
            "engagement_score": engagement_score,
            "years_experience": years_experience,
            "years_at_company": years_at_company,
            "leadership_trait_score": leadership_trait_score,
            "peer_feedback_score": peer_feedback_score,
            "manager_feedback_score": manager_feedback_score,
            "innovation_score": innovation_score,
            "learning_hours": learning_hours,
            "projects_led": projects_led,
            "promotion_last_3yrs": promotion_last_3yrs,
            "absenteeism_days": absenteeism_days,
            "attrition_risk_score": attrition_risk_score,
        }
    )

    # Hidden scoring rule creates a realistic but learnable target.
    leadership_score = (
        0.24 * df["performance_score"]
        + 0.17 * df["engagement_score"]
        + 0.22 * df["leadership_trait_score"]
        + 5.0 * df["manager_feedback_score"]
        + 3.5 * df["peer_feedback_score"]
        + 0.08 * df["innovation_score"]
        + 0.10 * df["learning_hours"]
        + 2.7 * df["projects_led"]
        + 6.0 * df["promotion_last_3yrs"]
        - 0.45 * df["absenteeism_days"]
        - 0.12 * df["attrition_risk_score"]
        + rng.normal(0, 7, n_records)
    )
    threshold = np.percentile(leadership_score, 72)
    df[TARGET] = (leadership_score >= threshold).astype(int)

    # Add small missingness to demonstrate preprocessing.
    columns_with_missing = [
        "performance_score",
        "engagement_score",
        "leadership_trait_score",
        "peer_feedback_score",
        "manager_feedback_score",
        "education",
    ]
    for col in columns_with_missing:
        missing_idx = rng.choice(df.index, size=int(0.035 * n_records), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def save_eda_plots(df: pd.DataFrame) -> None:
    """Create univariate and multivariate EDA visualizations."""
    clean_for_plots = df.copy()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=clean_for_plots, x=TARGET)
    plt.title("Future Leader Class Distribution")
    plt.xlabel("Future Leader (0 = No, 1 = Yes)")
    plt.ylabel("Employee Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(data=clean_for_plots, x="performance_score", hue=TARGET, kde=True, bins=25)
    plt.title("Performance Score by Future Leader Status")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "performance_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=clean_for_plots, x=TARGET, y="leadership_trait_score")
    plt.title("Leadership Trait Score vs Future Leader")
    plt.xlabel("Future Leader")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "leadership_trait_boxplot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    department_rate = clean_for_plots.groupby("department")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=department_rate.index, y=department_rate.values)
    plt.title("Future Leader Rate by Department")
    plt.ylabel("Future Leader Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "department_future_leader_rate.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 9))
    corr = clean_for_plots[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def print_eda_patterns(df: pd.DataFrame) -> None:
    print("\n5. EDA Key Patterns")
    leader_means = df.groupby(TARGET)[NUMERIC_FEATURES].mean(numeric_only=True).round(2)
    print("\nAverage numeric values by target:")
    print(leader_means)

    top_correlations = (
        df[NUMERIC_FEATURES + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
        .head(6)
        .round(3)
    )
    print("\nTop positive correlations with future_leader:")
    print(top_correlations)
    print(f"\nEDA plots saved to: {PLOTS_DIR}")


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def train_and_compare_models(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []
    fitted_models = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )
        fitted_models[model_name] = pipeline

        print(f"\n{model_name} Classification Report")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
    best_model_name = results_df.loc[0, "model"]
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, MODELS_DIR / "future_leader_model.joblib")
    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "best_model_name": best_model_name,
            "model_results": results_df,
        },
        MODELS_DIR / "model_metadata.joblib",
    )

    return results_df, best_model_name, best_model


def predict_future_leader(employee_data: dict, model=None) -> dict:
    """Predict whether one employee is a future leader.

    Example:
        result = predict_future_leader({
            "department": "Engineering",
            "job_level": "Senior",
            "education": "Master",
            "performance_score": 88,
            ...
        })
    """
    if model is None:
        model = joblib.load(MODELS_DIR / "future_leader_model.joblib")

    employee_df = pd.DataFrame([employee_data], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(employee_df)[0])

    probability = None
    if hasattr(model.named_steps["model"], "predict_proba"):
        probability = float(model.predict_proba(employee_df)[0][1])

    return {
        "prediction": prediction,
        "label": "Future Leader" if prediction == 1 else "Not Future Leader Yet",
        "future_leader_probability": probability,
    }


def main() -> None:
    objective_summary()
    feature_design_summary()

    print("\n3. Data Creation")
    df = generate_synthetic_employee_data(n_records=750, random_state=42)
    dataset_path = DATA_DIR / "future_leaders_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Synthetic dataset created with {len(df)} records.")
    print(f"Dataset saved to: {dataset_path}")

    print("\n4. Data Preprocessing")
    print("Missing values will be handled inside model pipelines using median/mode imputers.")
    print("Categorical variables will be one-hot encoded.")
    print("Numeric variables will be scaled with StandardScaler.")

    print("\n5. Exploratory Data Analysis")
    save_eda_plots(df)
    print_eda_patterns(df)

    print("\n6. Model Building and Evaluation")
    results_df, best_model_name, best_model = train_and_compare_models(df)

    print("\n7. Model Selection")
    print(results_df.round(4))
    print(f"\nBest model selected by F1-score: {best_model_name}")
    print(f"Best model saved to: {MODELS_DIR / 'future_leader_model.joblib'}")

    print("\n8. Leadership Prediction System")
    sample_employee = {
        "department": "Engineering",
        "job_level": "Senior",
        "education": "Master",
        "performance_score": 90,
        "engagement_score": 86,
        "years_experience": 8,
        "years_at_company": 4,
        "leadership_trait_score": 91,
        "peer_feedback_score": 4.5,
        "manager_feedback_score": 4.6,
        "innovation_score": 88,
        "learning_hours": 55,
        "projects_led": 5,
        "promotion_last_3yrs": 1,
        "absenteeism_days": 2,
        "attrition_risk_score": 18,
    }
    prediction = predict_future_leader(sample_employee, best_model)
    print("Sample employee prediction:")
    print(prediction)

    print("\n9. Dashboard")
    print("Run the Streamlit dashboard with:")
    print("streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
