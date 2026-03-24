import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def infer_statsmodels_family(model_class, model_kwargs=None):
    """
    Infer behavioral family from a statsmodels class plus model_kwargs.

    Returns one of:
    - "binary"
    - "regression"
    - "count"
    - "unknown"
    """
    model_kwargs = model_kwargs or {}

    regression_classes = {
        sm.OLS,
        sm.WLS,
        sm.GLS,
    }

    binary_classes = {
        sm.Logit,
        sm.Probit,
    }

    count_classes = {
        sm.Poisson,
        sm.NegativeBinomial,
    }

    if model_class in binary_classes:
        return "binary"

    if model_class in regression_classes:
        return "regression"

    if model_class in count_classes:
        return "count"

    if model_class is sm.GLM:
        family = model_kwargs.get("family", None)

        if isinstance(family, sm.families.Binomial):
            return "binary"

        if isinstance(family, sm.families.Gaussian):
            return "regression"

        if isinstance(family, sm.families.Poisson):
            return "count"

    return "unknown"


class _StatsModelsBase(BaseEstimator):
    """
    Shared sklearn-compatible base wrapper for statsmodels estimators.
    """

    def __init__(
        self,
        model_class,
        model_kwargs=None,
        fit_kwargs=None,
        add_constant=True,
        return_dataframe=True,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.fit_kwargs = fit_kwargs
        self.add_constant = add_constant
        self.return_dataframe = return_dataframe

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()

        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(X, columns=self.feature_names_in_)

        return pd.DataFrame(X)

    def _prepare_exog(self, X, fitting=False):
        X = self._to_dataframe(X)

        if fitting:
            self.feature_names_in_ = list(X.columns)

        X = X.loc[:, self.feature_names_in_]

        if self.add_constant:
            X = sm.add_constant(X, has_constant="add")

        self.exog_names_ = list(X.columns)

        if self.return_dataframe:
            return X

        return X.to_numpy()

    def _fit_core(self, X, y):
        X_prepared = self._prepare_exog(X, fitting=True)

        if isinstance(X, pd.DataFrame):
            y_series = pd.Series(y, index=X.index)
        else:
            y_series = pd.Series(y)

        model_kwargs = {} if self.model_kwargs is None else dict(self.model_kwargs)
        fit_kwargs = {} if self.fit_kwargs is None else dict(self.fit_kwargs)

        self.model_ = self.model_class(y_series, X_prepared, **model_kwargs)
        self.result_ = self.model_.fit(**fit_kwargs)

        params = self.result_.params
        if isinstance(params, pd.Series):
            self.params_ = params.copy()
        else:
            self.params_ = pd.Series(params, index=self.exog_names_)

        if self.add_constant and "const" in self.params_.index:
            self.intercept_ = np.array([self.params_.loc["const"]])
            self.coef_ = np.array([self.params_.drop("const").to_numpy()])
        else:
            self.intercept_ = np.array([0.0])
            self.coef_ = np.array([self.params_.to_numpy()])

        self.n_features_in_ = len(self.feature_names_in_)
        return X_prepared

    def summary(self):
        return self.result_.summary()

    @property
    def pvalues_(self):
        return getattr(self.result_, "pvalues", None)

    @property
    def bse_(self):
        return getattr(self.result_, "bse", None)

    @property
    def tvalues_(self):
        return getattr(self.result_, "tvalues", None)


class StatsModelsClassifier(_StatsModelsBase, ClassifierMixin):
    """
    sklearn-compatible wrapper for binary statsmodels classifiers.

    Supports:
    - sm.Logit
    - sm.Probit
    - sm.GLM with Binomial family
    """

    def __init__(
        self,
        model_class,
        model_kwargs=None,
        fit_kwargs=None,
        add_constant=True,
        return_dataframe=True,
        threshold=0.5,
    ):
        super().__init__(
            model_class=model_class,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
            add_constant=add_constant,
            return_dataframe=return_dataframe,
        )
        self.threshold = threshold

    def fit(self, X, y):
        family = infer_statsmodels_family(self.model_class, self.model_kwargs)
        if family != "binary":
            raise ValueError(
                f"StatsModelsClassifier expected a binary model, got family={family!r}."
            )

        self._fit_core(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X_prepared = self._prepare_exog(X, fitting=False)
        p1 = np.asarray(self.result_.predict(X_prepared), dtype=float)
        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= self.threshold).astype(int)

    def decision_function(self, X):
        X_prepared = self._prepare_exog(X, fitting=False)

        if isinstance(X_prepared, pd.DataFrame):
            return np.asarray(
                X_prepared @ self.params_.loc[self.exog_names_],
                dtype=float
            )

        return np.asarray(
            X_prepared @ self.params_.loc[self.exog_names_].to_numpy(),
            dtype=float
        )


class StatsModelsRegressor(_StatsModelsBase, RegressorMixin):
    """
    sklearn-compatible wrapper for statsmodels regressors.

    Supports:
    - sm.OLS
    - sm.WLS
    - sm.GLS
    - sm.Poisson
    - sm.NegativeBinomial
    - sm.GLM with Gaussian or Poisson family

    Note:
    count models are treated as regressors from sklearn's point of view.
    """

    def fit(self, X, y):
        family = infer_statsmodels_family(self.model_class, self.model_kwargs)
        if family not in {"regression", "count"}:
            raise ValueError(
                f"StatsModelsRegressor expected a regression/count model, got family={family!r}."
            )

        self._fit_core(X, y)
        return self

    def predict(self, X):
        X_prepared = self._prepare_exog(X, fitting=False)
        return np.asarray(self.result_.predict(X_prepared), dtype=float)