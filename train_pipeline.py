import os
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class Config:
    data_path: str = "data/loan_approval.csv"
    target_col: str = "loan_approved"
    id_cols: Tuple[str, ...] = ("name",)
    leakage_cols: Tuple[str, ...] = ("points",)
    test_size: float = 0.20
    val_size: float = 0.20
    random_state: int = 42
    model_dir: str = "models"
    report_dir: str = "reports"
    model_path: str = "models/best_loan_model.joblib"
    metrics_path: str = "reports/model_metrics.json"
    prediction_path: str = "reports/test_predictions.csv"
    feature_importance_path: str = "reports/feature_importance.csv"


config = Config()


# ============================================================
# 2. LOGGING
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ============================================================
# 3. UTILITIES
# ============================================================

def ensure_directories(cfg: Config) -> None:
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.report_dir, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    logging.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logging.info("Dataset loaded with shape: %s", df.shape)
    return df


def validate_dataset(df: pd.DataFrame, cfg: Config) -> None:
    logging.info("Validating dataset...")

    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in dataset.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    duplicate_count = df.duplicated().sum()
    logging.info("Duplicate rows: %d", duplicate_count)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logging.info("Missing values found:\n%s", missing.to_string())
    else:
        logging.info("No missing values found.")

    logging.info("Data types:\n%s", df.dtypes.to_string())
    logging.info("Validation completed.")


def basic_eda_report(df: pd.DataFrame, cfg: Config) -> None:
    logging.info("========== BASIC EDA ==========")
    logging.info("Shape: %s", df.shape)
    logging.info("Columns: %s", list(df.columns))
    logging.info("Target distribution:\n%s", df[cfg.target_col].value_counts(dropna=False).to_string())
    logging.info("Target ratio:\n%s", df[cfg.target_col].value_counts(normalize=True, dropna=False).to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        logging.info("Numeric summary:\n%s", df[numeric_cols].describe().to_string())

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        logging.info("Top categories in '%s':\n%s", col, df[col].value_counts().head(10).to_string())


def leakage_check(df: pd.DataFrame, cfg: Config) -> None:
    logging.info("Running leakage check...")

    for col in cfg.leakage_cols:
        if col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    threshold_analysis = pd.crosstab(df[col] >= 60, df[cfg.target_col])
                    logging.info("Possible leakage check for '%s':\n%s", col, threshold_analysis.to_string())
                else:
                    logging.info("Leakage candidate '%s' is non-numeric. Review manually.", col)
            except Exception as e:
                logging.warning("Could not analyze leakage column '%s': %s", col, str(e))


def clean_data(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    logging.info("Cleaning dataset...")

    df = df.copy()
    df = df.drop_duplicates()

    drop_cols = [col for col in list(cfg.id_cols) + list(cfg.leakage_cols) if col in df.columns]
    if drop_cols:
        logging.info("Dropping columns: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # Convert target to int if boolean
    if df[cfg.target_col].dtype == bool:
        df[cfg.target_col] = df[cfg.target_col].astype(int)

    logging.info("Data cleaned. New shape: %s", df.shape)
    return df


def split_data(df: pd.DataFrame, cfg: Config):
    logging.info("Splitting data into train, validation, and test sets...")

    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col].astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

    relative_val_size = cfg.val_size / (1 - cfg.test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=relative_val_size,
        random_state=cfg.random_state,
        stratify=y_train_full
    )

    logging.info("Train shape: %s", X_train.shape)
    logging.info("Validation shape: %s", X_val.shape)
    logging.info("Test shape: %s", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    logging.info("Numeric columns: %s", numeric_cols)
    logging.info("Categorical columns: %s", categorical_cols)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def build_model_candidates(preprocessor: ColumnTransformer) -> Dict[str, Tuple[Pipeline, Dict]]:
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ])

    gb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            random_state=42
        ))
    ])

    rf_params = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None]
    }

    gb_params = {
        "model__n_estimators": [100, 150, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 4, 5],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.8, 0.9, 1.0]
    }

    return {
        "random_forest": (rf_pipeline, rf_params),
        "gradient_boosting": (gb_pipeline, gb_params)
    }


def tune_model(
    pipeline: Pipeline,
    param_grid: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str
) -> RandomizedSearchCV:
    logging.info("Tuning model: %s", model_name)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)

    logging.info("Best params for %s: %s", model_name, search.best_params_)
    logging.info("Best CV ROC-AUC for %s: %.4f", model_name, search.best_score_)

    return search


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str
) -> Dict:
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None

    metrics = {
        "dataset": dataset_name,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y, y_pred, zero_division=0), 4),
    }

    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y, y_prob), 4)

    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, output_dict=False, zero_division=0)

    logging.info("========== %s METRICS ==========", dataset_name.upper())
    logging.info("Metrics: %s", metrics)
    logging.info("Confusion Matrix:\n%s", cm)
    logging.info("Classification Report:\n%s", cr)

    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }


def compare_and_select_best(
    tuned_models: Dict[str, RandomizedSearchCV],
    X_val: pd.DataFrame,
    y_val: pd.Series
):
    best_model_name = None
    best_model = None
    best_auc = -1
    comparison_results = {}

    for model_name, search in tuned_models.items():
        result = evaluate_model(search.best_estimator_, X_val, y_val, f"validation_{model_name}")
        auc = result["metrics"].get("roc_auc", 0.0)
        comparison_results[model_name] = result

        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name
            best_model = search.best_estimator_

    logging.info("Best selected model: %s with validation ROC-AUC: %.4f", best_model_name, best_auc)
    return best_model_name, best_model, comparison_results


def save_test_predictions(model, X_test: pd.DataFrame, y_test: pd.Series, path: str) -> None:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    result_df = X_test.copy()
    result_df["actual"] = y_test.values
    result_df["predicted"] = preds
    result_df["predicted_probability"] = probs

    result_df.to_csv(path, index=False)
    logging.info("Saved test predictions to %s", path)


def save_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, path: str) -> None:
    try:
        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        inner_model = model.named_steps["model"]

        if hasattr(inner_model, "feature_importances_"):
            importances = inner_model.feature_importances_
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False)
        else:
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                scoring="roc_auc",
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )
            fi_df = pd.DataFrame({
                "feature": X_test.columns,
                "importance": perm.importances_mean
            }).sort_values(by="importance", ascending=False)

        fi_df.to_csv(path, index=False)
        logging.info("Saved feature importance to %s", path)
    except Exception as e:
        logging.warning("Feature importance could not be saved: %s", str(e))


def save_metrics_report(report: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=4)
    logging.info("Saved metrics report to %s", path)


# ============================================================
# 4. MAIN TRAINING FLOW
# ============================================================

def main():
    setup_logging()
    ensure_directories(config)

    # Step 1: Load
    df = load_data(config.data_path)

    # Step 2: Validate
    validate_dataset(df, config)

    # Step 3: EDA
    basic_eda_report(df, config)

    # Step 4: Leakage check
    leakage_check(df, config)

    # Step 5: Clean
    df = clean_data(df, config)

    # Step 6: Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, config)

    # Step 7: Preprocessor
    preprocessor = build_preprocessor(X_train)

    # Step 8: Candidate models
    candidates = build_model_candidates(preprocessor)

    tuned_models = {}

    # Step 9: Train + Tune
    for model_name, (pipeline, params) in candidates.items():
        search = tune_model(pipeline, params, X_train, y_train, model_name)
        tuned_models[model_name] = search

    # Step 10: Compare on validation set
    best_model_name, best_model, val_results = compare_and_select_best(
        tuned_models, X_val, y_val
    )

    # Step 11: Final evaluation on test set
    test_result = evaluate_model(best_model, X_test, y_test, "test")

    # Step 12: Save model
    joblib.dump(best_model, config.model_path)
    logging.info("Best model saved to %s", config.model_path)

    # Step 13: Save predictions
    save_test_predictions(best_model, X_test, y_test, config.prediction_path)

    # Step 14: Save feature importance
    save_feature_importance(best_model, X_test, y_test, config.feature_importance_path)

    # Step 15: Save full report
    final_report = {
        "best_model_name": best_model_name,
        "validation_results": val_results,
        "test_result": test_result
    }
    save_metrics_report(final_report, config.metrics_path)

    logging.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()