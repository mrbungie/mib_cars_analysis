from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.utils.multiclass as multiclass
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor

_original_type_of_target = multiclass.type_of_target


def patched_type_of_target(y, input_name: str = "", raise_unknown: bool = False):
    if getattr(y, "dtype", None) in [np.float32, np.float64]:
        values = np.asarray(y)
        finite_values = values[np.isfinite(values)]
        unique_values = np.unique(finite_values)
        if (
            finite_values.size
            and unique_values.size <= 20
            and np.allclose(finite_values, np.round(finite_values))
        ):
            return _original_type_of_target(
                np.round(values).astype(int),
                input_name=input_name,
                raise_unknown=raise_unknown,
            )
        return "continuous"
    return _original_type_of_target(
        y, input_name=input_name, raise_unknown=raise_unknown
    )


setattr(multiclass, "type_of_target", patched_type_of_target)


ROOT = Path(__file__).resolve().parents[1]
MODELLING_DIR = ROOT / "notebooks/4_modelling"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


binning_module = load_module("modelling_binning", MODELLING_DIR / "binning.py")
statsmodels_module = load_module(
    "modelling_statsmodels_api", MODELLING_DIR / "statsmodels_api.py"
)

DataFrameScaler = binning_module.DataFrameScaler
NamedBinningProcess = binning_module.NamedBinningProcess
StatsModelsClassifier = statsmodels_module.StatsModelsClassifier
StatsModelsRegressor = statsmodels_module.StatsModelsRegressor


TRAIN_PATH = ROOT / "data/intermediate/df_train_stratified.parquet"
TEST_PATH = ROOT / "data/intermediate/df_test_stratified.parquet"
SLIDE_DATA_DIR = ROOT / "slidedeck/data"
CLASSIFICATION_REPORT = SLIDE_DATA_DIR / "classification_model_report.xlsx"
REGRESSION_REPORT = SLIDE_DATA_DIR / "regression_model_report.xlsx"

CLASSIFICATION_FEATURES = [
    "Supplies Group",
    "Supplies Subgroup",
    "Region",
    "Route To Market",
    "Elapsed Days In Sales Stage",
    "Sales Stage Change Count",
    "Total Days Identified Through Closing",
    "Total Days Identified Through Qualified",
    "Client Size By Revenue (USD)",
    "Client Size By Employee Count",
    "Revenue From Client Past Two Years (USD)",
    "Competitor Type",
    "Ratio Days Identified To Total Days",
    "Ratio Days Validated To Total Days",
    "Ratio Days Qualified To Total Days",
]
CLASSIFICATION_TARGET = "Opportunity Result Bool"

REGRESSION_FEATURES = CLASSIFICATION_FEATURES.copy()
REGRESSION_TARGET = "Opportunity Amount USD"


class RandomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        self.y_train_ = np.asarray(y, dtype=float)
        self.random_state_ = np.random.RandomState(self.random_state)
        return self

    def predict(self, X):
        n = len(X)
        return self.random_state_.choice(self.y_train_, size=n, replace=True)


class IntegerTargetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.asarray(y, dtype=int))
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def classes_(self):
        return self.estimator_.classes_

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


def get_classic_preprocessor(
    categorical_cols: list[str], numerical_cols: list[str], scaler=None
):
    if scaler is None:
        scaler = RobustScaler()
    return ColumnTransformer(
        [
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", scaler),
                    ]
                ),
                numerical_cols,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="MISSING"),
                        ),
                        (
                            "ohe",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    ).set_output(transform="pandas")


def get_transformed_regressor_log1p(model):
    return TransformedTargetRegressor(
        regressor=model, func=np.log1p, inverse_func=np.expm1
    )


def get_power_transformed_regressor(model, method: str):
    from sklearn.preprocessing import PowerTransformer

    return TransformedTargetRegressor(
        regressor=model,
        transformer=PowerTransformer(method=method, standardize=True),
    )


def normalize_feature_name(transformed_name: str, original_features: list[str]) -> str:
    name = str(transformed_name)
    for prefix in [
        "numerical__",
        "categorical__",
        "binning__",
        "remainder__",
        "scaler__",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    feature_order = sorted(original_features, key=len, reverse=True)
    for feature in feature_order:
        feature = str(feature)
        if name == feature:
            return feature
        if (
            name.startswith(feature + "_")
            or name.startswith(feature + "[")
            or name.startswith(feature + "__")
            or name.startswith(feature + ":")
        ):
            return feature
    for feature in feature_order:
        feature = str(feature)
        if feature in name:
            return feature
    return name


def get_classifier_importance(
    final_pipeline, X: pd.DataFrame, original_features: list[str]
) -> pd.DataFrame:
    if "preprocessing" in final_pipeline.named_steps:
        transformed = final_pipeline.named_steps["preprocessing"].transform(
            X.iloc[: min(len(X), 500)].copy()
        )
    elif "binning" in final_pipeline.named_steps:
        transformed = final_pipeline.named_steps["binning"].transform(
            X.iloc[: min(len(X), 500)].copy()
        )
    else:
        transformed = X.iloc[: min(len(X), 500)].copy()

    model = final_pipeline.named_steps["model"]
    transformed_columns = (
        list(transformed.columns)
        if hasattr(transformed, "columns")
        else [f"feature_{i}" for i in range(transformed.shape[1])]
    )

    def extract_classifier_term_importance(current_model) -> pd.DataFrame | None:
        if hasattr(current_model, "coef_"):
            coef_values = np.asarray(current_model.coef_).reshape(-1)
            return pd.DataFrame(
                {
                    "transformed_feature": transformed_columns,
                    "term_importance": np.abs(coef_values),
                }
            )
        if isinstance(current_model, CalibratedClassifierCV):
            calibrated_models = getattr(current_model, "calibrated_classifiers_", [])
            if not calibrated_models:
                return None
            term_frames: list[pd.DataFrame] = []
            for calibrated_model in calibrated_models:
                base_estimator = getattr(calibrated_model, "estimator", None)
                if base_estimator is None or not hasattr(base_estimator, "coef_"):
                    continue
                coef_values = np.asarray(base_estimator.coef_).reshape(-1)
                term_frames.append(
                    pd.DataFrame(
                        {
                            "transformed_feature": transformed_columns,
                            "term_importance": np.abs(coef_values),
                        }
                    )
                )
            if not term_frames:
                return None
            return (
                pd.concat(term_frames, ignore_index=True)
                .groupby("transformed_feature", as_index=False)
                .agg(term_importance=("term_importance", "mean"))
            )
        return None

    coef_df = extract_classifier_term_importance(model)
    if coef_df is not None:
        coef_df["feature"] = coef_df["transformed_feature"].apply(
            lambda x: normalize_feature_name(x, original_features)
        )
        feature_importance_df = (
            coef_df.groupby("feature", as_index=False)
            .agg(
                importance_mean=("term_importance", lambda s: float(s.sum())),
                transformed_terms=("transformed_feature", "count"),
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_).reshape(-1)
        feature_importance_df = pd.DataFrame(
            {
                "feature": [
                    normalize_feature_name(c, original_features)
                    for c in transformed_columns
                ],
                "importance_mean": importances,
                "transformed_terms": 1,
            }
        )
        feature_importance_df = (
            feature_importance_df.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance_mean", "sum"),
                transformed_terms=("transformed_terms", "sum"),
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    else:
        feature_importance_df = pd.DataFrame(
            {
                "feature": original_features,
                "importance_mean": 0.0,
                "transformed_terms": 1,
            }
        )

    feature_importance_df["importance_std"] = 0.0
    return feature_importance_df


def get_regressor_importance(
    final_pipeline, X: pd.DataFrame, original_features: list[str]
) -> pd.DataFrame:
    if "preprocessing" in final_pipeline.named_steps:
        transformed = final_pipeline.named_steps["preprocessing"].transform(
            X.iloc[: min(len(X), 500)].copy()
        )
    elif "binning" in final_pipeline.named_steps:
        transformed = final_pipeline.named_steps["binning"].transform(
            X.iloc[: min(len(X), 500)].copy()
        )
    else:
        transformed = X.iloc[: min(len(X), 500)].copy()

    model = final_pipeline.named_steps["model"]
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_

    transformed_columns = (
        list(transformed.columns)
        if hasattr(transformed, "columns")
        else [f"feature_{i}" for i in range(transformed.shape[1])]
    )

    if hasattr(model, "coef_"):
        coef_values = np.asarray(model.coef_).reshape(-1)
        feature_importance_df = pd.DataFrame(
            {
                "feature": [
                    normalize_feature_name(c, original_features)
                    for c in transformed_columns
                ],
                "importance_mean": np.abs(coef_values),
                "transformed_terms": 1,
            }
        )
        feature_importance_df = (
            feature_importance_df.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance_mean", "sum"),
                transformed_terms=("transformed_terms", "sum"),
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_).reshape(-1)
        feature_importance_df = pd.DataFrame(
            {
                "feature": [
                    normalize_feature_name(c, original_features)
                    for c in transformed_columns
                ],
                "importance_mean": importances,
                "transformed_terms": 1,
            }
        )
        feature_importance_df = (
            feature_importance_df.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance_mean", "sum"),
                transformed_terms=("transformed_terms", "sum"),
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    else:
        feature_importance_df = pd.DataFrame(
            {
                "feature": original_features,
                "importance_mean": 0.0,
                "transformed_terms": 1,
            }
        )

    feature_importance_df["importance_std"] = 0.0
    return feature_importance_df


def build_classification_reports() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X = train_df[CLASSIFICATION_FEATURES].copy()
    y = train_df[CLASSIFICATION_TARGET].astype(int).copy()
    X_test = test_df[CLASSIFICATION_FEATURES].copy()
    y_test = test_df[CLASSIFICATION_TARGET].astype(int).copy()

    categorical_cols = [
        "Supplies Group",
        "Supplies Subgroup",
        "Region",
        "Route To Market",
        "Competitor Type",
        "Client Size By Revenue (USD)",
        "Client Size By Employee Count",
        "Revenue From Client Past Two Years (USD)",
    ]
    numerical_cols = [c for c in CLASSIFICATION_FEATURES if c not in categorical_cols]
    scale_pos_weight = float((y == 0).sum() / max(int((y == 1).sum()), 1))

    experiment_grid = {
        "random_classifier": DummyClassifier(strategy="stratified", random_state=42),
        "always_true_classifier": DummyClassifier(strategy="constant", constant=1),
        "classic_logit_robust_scaler": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                ("model", StatsModelsClassifier(sm.Logit)),
            ]
        ),
        "logit_elasticnet_robust_scaler": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.5,
                        penalty="elasticnet",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logit_elasticnet_robust_scaler_balanced": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.5,
                        penalty="elasticnet",
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest_classifier": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    IntegerTargetClassifier(
                        RandomForestClassifier(
                            n_estimators=400,
                            min_samples_leaf=5,
                            n_jobs=-1,
                            random_state=42,
                        )
                    ),
                ),
            ]
        ),
        "random_forest_classifier_balanced": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    IntegerTargetClassifier(
                        RandomForestClassifier(
                            n_estimators=400,
                            min_samples_leaf=5,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=42,
                        )
                    ),
                ),
            ]
        ),
        "logit_binned_ohe": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=CLASSIFICATION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="bin_ohe",
                    ),
                ),
                ("model", StatsModelsClassifier(sm.Logit)),
            ]
        ),
        "logit_binned_ohe_balanced": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=CLASSIFICATION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="bin_ohe",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logit_binned_ohe_balanced_calibrated": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=CLASSIFICATION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="bin_ohe",
                    ),
                ),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=LogisticRegression(
                            solver="saga",
                            class_weight="balanced",
                            max_iter=10000,
                            random_state=42,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        "logit_elasticnet_binned_ohe": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=CLASSIFICATION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="bin_ohe",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.5,
                        penalty="elasticnet",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logit_elasticnet_binned_ohe_balanced": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=CLASSIFICATION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="bin_ohe",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.5,
                        penalty="elasticnet",
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logit_binned_woe": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=CLASSIFICATION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="woe",
                    ),
                ),
                ("model", StatsModelsClassifier(sm.Logit)),
            ]
        ),
        "logit_elastic_binned_woe": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=CLASSIFICATION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="woe",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.5,
                        penalty="elasticnet",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logit_elastic_binned_woe_balanced": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=CLASSIFICATION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="woe",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.5,
                        penalty="elasticnet",
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=42,
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                    ),
                ),
            ]
        ),
        "xgboost_balanced": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=42,
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: list[dict[str, object]] = []
    for experiment_name, estimator in tqdm(
        experiment_grid.items(), desc="Classification experiments"
    ):
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]
            try:
                model = clone(estimator)
                model.fit(X_train, y_train)
                valid_proba = model.predict_proba(X_valid)[:, 1]
                valid_label = (valid_proba >= 0.5).astype(int)
                results.append(
                    {
                        "experiment": experiment_name,
                        "fold": fold,
                        "roc_auc": roc_auc_score(y_valid, valid_proba),
                        "pr_auc": average_precision_score(y_valid, valid_proba),
                        "accuracy": accuracy_score(y_valid, valid_label),
                        "precision": precision_score(
                            y_valid, valid_label, zero_division=0
                        ),
                        "recall": recall_score(y_valid, valid_label, zero_division=0),
                        "f1": f1_score(y_valid, valid_label),
                        "status": "ok",
                        "error": np.nan,
                    }
                )
            except Exception as exc:  # noqa: PERF203
                results.append(
                    {
                        "experiment": experiment_name,
                        "fold": fold,
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    results_df = pd.DataFrame(results)
    summary_df = (
        results_df.groupby("experiment", dropna=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n_failed=("status", lambda s: int((s == "failed").sum())),
        )
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index()
    )

    selected_experiment = "random_forest_classifier_balanced"
    if selected_experiment not in summary_df["experiment"].values:
        raise ValueError(
            f"Selected classification experiment missing: {selected_experiment}"
        )
    baseline_experiment = "random_classifier"
    selected_cv_metrics = summary_df.loc[
        summary_df["experiment"] == selected_experiment
    ].reset_index(drop=True)
    baseline_cv_metrics = summary_df.loc[
        summary_df["experiment"] == baseline_experiment
    ].reset_index(drop=True)

    final_pipeline = clone(experiment_grid[selected_experiment])
    final_pipeline.fit(X, y)
    baseline_pipeline = clone(experiment_grid[baseline_experiment])
    baseline_pipeline.fit(X, y)
    classification_comparison_experiments = [
        selected_experiment,
        "logit_binned_ohe_balanced",
        "logit_binned_ohe_balanced_calibrated",
        "random_classifier",
        "always_true_classifier",
    ]
    classification_comparison_predictions = []
    for experiment_name in classification_comparison_experiments:
        comparison_pipeline = clone(experiment_grid[experiment_name])
        comparison_pipeline.fit(X, y)
        comparison_proba = comparison_pipeline.predict_proba(X_test)[:, 1]
        comparison_label = comparison_pipeline.predict(X_test)
        classification_comparison_predictions.append(
            pd.DataFrame(
                {
                    "row_id": list(X_test.index),
                    "experiment": experiment_name,
                    "actual_result": y_test.to_numpy(),
                    "predicted_win_probability": comparison_proba,
                    "predicted_label": comparison_label,
                }
            )
        )

    train_oof_proba = cross_val_predict(
        clone(experiment_grid[selected_experiment]),
        X,
        y,
        cv=skf,
        method="predict_proba",
    )[:, 1]
    train_threshold_grid = np.unique(np.concatenate(([0.0], train_oof_proba, [1.0])))
    best_train_threshold = 0.5
    best_train_f1 = -1.0
    for threshold in train_threshold_grid:
        threshold_pred = (train_oof_proba >= threshold).astype(int)
        score = f1_score(y, threshold_pred, zero_division=0)
        if score > best_train_f1:
            best_train_f1 = score
            best_train_threshold = float(threshold)

    train_threshold_pred = (train_oof_proba >= best_train_threshold).astype(int)
    best_train_accuracy = accuracy_score(y, train_threshold_pred)
    best_train_precision = precision_score(y, train_threshold_pred, zero_division=0)
    best_train_recall = recall_score(y, train_threshold_pred, zero_division=0)

    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_label = (test_proba >= best_train_threshold).astype(int)

    baseline_test_proba = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_test_label = (baseline_test_proba >= 0.5).astype(int)

    test_metrics_df = pd.DataFrame(
        [
            {
                "experiment": selected_experiment,
                "dataset": "test",
                "roc_auc": roc_auc_score(y_test, test_proba),
                "pr_auc": average_precision_score(y_test, test_proba),
                "accuracy": accuracy_score(y_test, test_label),
                "f1": f1_score(y_test, test_label),
                "precision": precision_score(y_test, test_label, zero_division=0),
                "recall": recall_score(y_test, test_label, zero_division=0),
                "threshold": best_train_threshold,
                "n_rows": len(y_test),
            },
            {
                "experiment": baseline_experiment,
                "dataset": "test",
                "roc_auc": roc_auc_score(y_test, baseline_test_proba),
                "pr_auc": average_precision_score(y_test, baseline_test_proba),
                "accuracy": accuracy_score(y_test, baseline_test_label),
                "f1": f1_score(y_test, baseline_test_label),
                "precision": precision_score(
                    y_test, baseline_test_label, zero_division=0
                ),
                "recall": recall_score(y_test, baseline_test_label, zero_division=0),
                "threshold": 0.5,
                "n_rows": len(y_test),
            },
        ]
    )

    feature_importance_df = get_classifier_importance(
        final_pipeline, X, CLASSIFICATION_FEATURES
    )

    test_prediction_export = pd.DataFrame(
        {
            "row_id": list(X_test.index),
            "actual_result": y_test.to_numpy(),
            "predicted_win_probability": test_proba,
            "predicted_label": test_label,
        }
    )
    test_prediction_export["score_decile"] = pd.qcut(
        test_prediction_export["predicted_win_probability"].rank(method="first"),
        10,
        labels=list(range(1, 11)),
    ).astype(int)
    prioritization_lift_df = (
        test_prediction_export.groupby("score_decile", as_index=False)
        .agg(
            opportunities=("row_id", "count"),
            wins=("actual_result", "sum"),
            avg_win_probability=("predicted_win_probability", "mean"),
            observed_win_rate=("actual_result", "mean"),
        )
        .sort_values("score_decile")
        .reset_index(drop=True)
    )
    prioritization_lift_df["share_of_total_wins"] = (
        prioritization_lift_df["wins"] / prioritization_lift_df["wins"].sum()
    )
    prioritization_lift_df["cumulative_share_of_total_wins"] = prioritization_lift_df[
        "share_of_total_wins"
    ].cumsum()

    roc_curve_df = pd.DataFrame(
        {"fpr": np.r_[0.0, np.sort(np.unique(np.linspace(0, 1, 200)))], "tpr": np.nan}
    )
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    roc_curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    metadata_df = pd.DataFrame(
        [
            {
                "target": "Opportunity Result",
                "selected_experiment": selected_experiment,
                "baseline_experiment": baseline_experiment,
                "selection_rule": "Selected classification model with threshold optimized on train OOF F1",
                "cv_roc_auc_mean": float(selected_cv_metrics.loc[0, "roc_auc_mean"]),
                "cv_pr_auc_mean": float(selected_cv_metrics.loc[0, "pr_auc_mean"]),
                "cv_accuracy_mean": float(selected_cv_metrics.loc[0, "accuracy_mean"]),
                "cv_f1_mean": float(selected_cv_metrics.loc[0, "f1_mean"]),
                "test_roc_auc": float(
                    test_metrics_df.loc[
                        test_metrics_df["experiment"] == selected_experiment, "roc_auc"
                    ].iloc[0]
                ),
                "test_pr_auc": float(
                    test_metrics_df.loc[
                        test_metrics_df["experiment"] == selected_experiment, "pr_auc"
                    ].iloc[0]
                ),
                "test_accuracy": float(
                    test_metrics_df.loc[
                        test_metrics_df["experiment"] == selected_experiment, "accuracy"
                    ].iloc[0]
                ),
                "test_f1": float(
                    test_metrics_df.loc[
                        test_metrics_df["experiment"] == selected_experiment, "f1"
                    ].iloc[0]
                ),
                "train_selected_threshold": best_train_threshold,
                "train_selected_threshold_accuracy": best_train_accuracy,
                "train_selected_threshold_precision": best_train_precision,
                "train_selected_threshold_recall": best_train_recall,
                "train_selected_threshold_f1": best_train_f1,
                "notes": "Final model trained on full train split and scored on held-out test split.",
            }
        ]
    )

    SLIDE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(CLASSIFICATION_REPORT, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="cv_results", index=False)
        summary_df.to_excel(writer, sheet_name="cv_summary", index=False)
        selected_cv_metrics.to_excel(
            writer, sheet_name="cv_selected_model", index=False
        )
        baseline_cv_metrics.to_excel(
            writer, sheet_name="cv_baseline_model", index=False
        )
        test_metrics_df.to_excel(writer, sheet_name="test_metrics", index=False)
        feature_importance_df.to_excel(
            writer, sheet_name="feature_importance", index=False
        )
        test_prediction_export.to_excel(
            writer, sheet_name="test_predictions", index=False
        )
        pd.concat(classification_comparison_predictions, ignore_index=True).to_excel(
            writer, sheet_name="comparison_test_predictions", index=False
        )
        prioritization_lift_df.to_excel(
            writer, sheet_name="prioritization_lift", index=False
        )
        roc_curve_df.to_excel(writer, sheet_name="roc_curve", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)


def build_regression_reports() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X = train_df[REGRESSION_FEATURES].copy()
    y = train_df[REGRESSION_TARGET].astype(np.float32).copy()
    X_test = test_df[REGRESSION_FEATURES].copy()
    y_test = test_df[REGRESSION_TARGET].astype(np.float32).copy()

    categorical_cols = [
        "Supplies Group",
        "Supplies Subgroup",
        "Region",
        "Route To Market",
        "Competitor Type",
        "Client Size By Revenue (USD)",
        "Client Size By Employee Count",
        "Revenue From Client Past Two Years (USD)",
    ]
    numerical_cols = [c for c in REGRESSION_FEATURES if c not in categorical_cols]

    experiment_grid = {
        "random_regressor": RandomRegressor(random_state=42),
        "mean_regressor": DummyRegressor(strategy="mean"),
        "median_regressor": DummyRegressor(strategy="median"),
        "classic_linear_log1p_robust_scaler": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    get_transformed_regressor_log1p(StatsModelsRegressor(sm.OLS)),
                ),
            ]
        ),
        "classic_linear_yeojohnson_robust_scaler": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    get_power_transformed_regressor(
                        StatsModelsRegressor(sm.OLS), method="yeo-johnson"
                    ),
                ),
            ]
        ),
        "classic_linear_boxcox_robust_scaler": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    get_power_transformed_regressor(
                        StatsModelsRegressor(sm.OLS), method="box-cox"
                    ),
                ),
            ]
        ),
        "classic_linear_robust_scaler": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                ("model", StatsModelsRegressor(sm.OLS)),
            ]
        ),
        "elasticnet_robust_scaler": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                ("model", ElasticNet(l1_ratio=0.5, max_iter=10000, random_state=42)),
            ]
        ),
        "linear_binned_ohe": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=REGRESSION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="bin_ohe",
                    ),
                ),
                ("model", StatsModelsRegressor(sm.OLS)),
            ]
        ),
        "linear_power_scaler_binned_ohe": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=REGRESSION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="bin_ohe",
                    ),
                ),
                (
                    "model",
                    get_power_transformed_regressor(
                        StatsModelsRegressor(sm.OLS), method="yeo-johnson"
                    ),
                ),
            ]
        ),
        "linear_binned_woe_robust_scaler": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=REGRESSION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="mean",
                    ),
                ),
                ("scaler", DataFrameScaler(RobustScaler())),
                ("model", StatsModelsRegressor(sm.OLS)),
            ]
        ),
        "random_forest_regressor": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "elastic_binned_woe": Pipeline(
            [
                (
                    "binning",
                    NamedBinningProcess(
                        variable_names=REGRESSION_FEATURES,
                        categorical_variables=categorical_cols,
                        output_method="mean",
                    ),
                ),
                ("model", ElasticNet(l1_ratio=0.5, max_iter=10000, random_state=42)),
            ]
        ),
        "xgboost": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
                        random_state=42,
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                    ),
                ),
            ]
        ),
        "xgboost_log1p": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    get_transformed_regressor_log1p(
                        XGBRegressor(
                            random_state=42,
                            n_estimators=250,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                        )
                    ),
                ),
            ]
        ),
        "xgboost_yeojohnson": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    get_power_transformed_regressor(
                        XGBRegressor(
                            random_state=42,
                            n_estimators=250,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                        ),
                        method="yeo-johnson",
                    ),
                ),
            ]
        ),
        "xgboost_boxcox": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    get_power_transformed_regressor(
                        XGBRegressor(
                            random_state=42,
                            n_estimators=250,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                        ),
                        method="box-cox",
                    ),
                ),
            ]
        ),
        "xgboost_pseudohubererror": Pipeline(
            [
                (
                    "preprocessing",
                    get_classic_preprocessor(
                        categorical_cols, numerical_cols, scaler=RobustScaler()
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
                        random_state=42,
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:pseudohubererror",
                    ),
                ),
            ]
        ),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results: list[dict[str, object]] = []
    for experiment_name, estimator in tqdm(
        experiment_grid.items(), desc="Regression experiments"
    ):
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]
            try:
                model = clone(estimator)
                model.fit(X_train, y_train)
                valid_pred = np.asarray(model.predict(X_valid), dtype=float)
                valid_abs_error = np.abs(y_valid - valid_pred)
                valid_ape = valid_abs_error / y_valid.replace(0, np.nan)
                results.append(
                    {
                        "experiment": experiment_name,
                        "fold": fold,
                        "r2": r2_score(y_valid, valid_pred),
                        "mae": mean_absolute_error(y_valid, valid_pred),
                        "mse": mean_squared_error(y_valid, valid_pred),
                        "mape": mean_absolute_percentage_error(y_valid, valid_pred),
                        "mdape": float(np.nanmedian(valid_ape)),
                        "status": "ok",
                        "error": np.nan,
                    }
                )
            except Exception as exc:  # noqa: PERF203
                results.append(
                    {
                        "experiment": experiment_name,
                        "fold": fold,
                        "r2": np.nan,
                        "mae": np.nan,
                        "mse": np.nan,
                        "mape": np.nan,
                        "mdape": np.nan,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    results_df = pd.DataFrame(results)
    summary_df = (
        results_df.groupby("experiment", dropna=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            median_ape=("mdape", "mean"),
            median_ape_std=("mdape", "std"),
            n_failed=("status", lambda s: int((s == "failed").sum())),
        )
        .sort_values(["median_ape", "mae_mean"], ascending=[True, True])
        .reset_index()
    )

    baseline_experiments = [
        "random_regressor",
        "mean_regressor",
        "median_regressor",
    ]
    non_baseline_summary = summary_df.loc[
        ~summary_df["experiment"].isin(baseline_experiments)
    ].copy()
    selected_experiment = str(
        non_baseline_summary.sort_values(
            ["median_ape", "mae_mean"], ascending=[True, True]
        ).iloc[0]["experiment"]
    )
    baseline_experiment = "random_regressor"
    selected_cv_metrics = summary_df.loc[
        summary_df["experiment"] == selected_experiment
    ].reset_index(drop=True)
    baseline_cv_metrics = summary_df.loc[
        summary_df["experiment"].isin(baseline_experiments)
    ].reset_index(drop=True)

    cv_pred = cross_val_predict(
        clone(experiment_grid[selected_experiment]), X, y, cv=kf
    )
    cv_abs_error = np.abs(y - cv_pred)
    cv_ape = cv_abs_error / y.replace(0, np.nan)
    selected_cv_metrics["median_ape"] = float(np.nanmedian(cv_ape))

    final_pipeline = clone(experiment_grid[selected_experiment])
    final_pipeline.fit(X, y)
    baseline_predictions: dict[str, np.ndarray] = {}
    for baseline_name in baseline_experiments:
        baseline_pipeline = clone(experiment_grid[baseline_name])
        baseline_pipeline.fit(X, y)
        baseline_predictions[baseline_name] = np.asarray(
            baseline_pipeline.predict(X_test), dtype=float
        )

    test_pred = np.asarray(final_pipeline.predict(X_test), dtype=float)
    regression_comparison_experiments = [
        "xgboost_boxcox",
        "xgboost_log1p",
        "xgboost_yeojohnson",
        "random_forest_regressor",
        "xgboost",
        "classic_linear_boxcox_robust_scaler",
    ]
    regression_comparison_predictions = []
    for experiment_name in regression_comparison_experiments:
        comparison_pipeline = clone(experiment_grid[experiment_name])
        comparison_pipeline.fit(X, y)
        comparison_pred = np.asarray(comparison_pipeline.predict(X_test), dtype=float)
        comparison_df = pd.DataFrame(
            {
                "row_id": list(X_test.index),
                "experiment": experiment_name,
                "actual_amount": y_test.to_numpy(),
                "predicted_amount": comparison_pred,
                "amount_bin": test_df.loc[X_test.index, "Deal Size Category (USD)"]
                .astype(str)
                .to_numpy(),
            }
        )
        comparison_df["absolute_error"] = (
            comparison_df["actual_amount"] - comparison_df["predicted_amount"]
        ).abs()
        comparison_df["ape"] = comparison_df["absolute_error"] / comparison_df[
            "actual_amount"
        ].replace(0, np.nan)
        regression_comparison_predictions.append(comparison_df)

    def reg_metrics(name: str, pred: np.ndarray) -> dict[str, object]:
        abs_error = np.abs(y_test - pred)
        ape = abs_error / y_test.replace(0, np.nan)
        return {
            "experiment": name,
            "dataset": "test",
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
            "mse": mean_squared_error(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "mape": mean_absolute_percentage_error(y_test, pred),
            "median_ape": float(np.nanmedian(ape)),
            "n_rows": len(y_test),
        }

    test_metrics_df = pd.DataFrame(
        [reg_metrics(selected_experiment, test_pred)]
        + [
            reg_metrics(baseline_name, pred)
            for baseline_name, pred in baseline_predictions.items()
        ]
    )

    feature_importance_df = get_regressor_importance(
        final_pipeline, X, REGRESSION_FEATURES
    )

    test_prediction_export = pd.DataFrame(
        {
            "row_id": list(X_test.index),
            "actual_amount": y_test.to_numpy(),
            "predicted_amount": test_pred,
        }
    )
    test_prediction_export["absolute_error"] = (
        test_prediction_export["actual_amount"]
        - test_prediction_export["predicted_amount"]
    ).abs()
    test_prediction_export["ape"] = test_prediction_export[
        "absolute_error"
    ] / test_prediction_export["actual_amount"].replace(0, np.nan)

    forecast_summary_df = pd.DataFrame(
        [
            {
                "metric": "actual_total_amount",
                "value": float(test_prediction_export["actual_amount"].sum()),
            },
            {
                "metric": "predicted_total_amount",
                "value": float(
                    test_prediction_export["predicted_amount"].clip(lower=0).sum()
                ),
            },
            {
                "metric": "mean_absolute_error",
                "value": float(test_prediction_export["absolute_error"].mean()),
            },
            {
                "metric": "median_absolute_percentage_error",
                "value": float(np.nanmedian(test_prediction_export["ape"])),
            },
        ]
    )

    metadata_df = pd.DataFrame(
        [
            {
                "target": "Opportunity Amount USD",
                "selected_experiment": selected_experiment,
                "baseline_experiment": baseline_experiment,
                "baseline_experiments": ", ".join(baseline_experiments),
                "selection_rule": "Best regression model by CV Median APE, tie-broken by MAE",
                "cv_r2_mean": float(selected_cv_metrics.loc[0, "r2_mean"]),
                "cv_mae_mean": float(selected_cv_metrics.loc[0, "mae_mean"]),
                "cv_mape_mean": float(selected_cv_metrics.loc[0, "mape_mean"]),
                "test_r2": float(
                    test_metrics_df.loc[
                        test_metrics_df["experiment"] == selected_experiment, "r2"
                    ].iloc[0]
                ),
                "test_mae": float(
                    test_metrics_df.loc[
                        test_metrics_df["experiment"] == selected_experiment, "mae"
                    ].iloc[0]
                ),
                "test_mape": float(
                    test_metrics_df.loc[
                        test_metrics_df["experiment"] == selected_experiment, "mape"
                    ].iloc[0]
                ),
                "notes": "Final model trained on full train split and scored on held-out test split.",
            }
        ]
    )

    SLIDE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(REGRESSION_REPORT, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="cv_results", index=False)
        summary_df.to_excel(writer, sheet_name="cv_summary", index=False)
        selected_cv_metrics.to_excel(
            writer, sheet_name="cv_selected_model", index=False
        )
        baseline_cv_metrics.to_excel(
            writer, sheet_name="cv_baseline_model", index=False
        )
        test_metrics_df.to_excel(writer, sheet_name="test_metrics", index=False)
        feature_importance_df.to_excel(
            writer, sheet_name="feature_importance", index=False
        )
        test_prediction_export.to_excel(
            writer, sheet_name="test_predictions", index=False
        )
        pd.concat(regression_comparison_predictions, ignore_index=True).to_excel(
            writer, sheet_name="comparison_test_predictions", index=False
        )
        forecast_summary_df.to_excel(writer, sheet_name="forecast_summary", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)


def main() -> None:
    build_classification_reports()
    build_regression_reports()
    print(f"saved: {CLASSIFICATION_REPORT}")
    print(f"saved: {REGRESSION_REPORT}")


if __name__ == "__main__":
    main()
