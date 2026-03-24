import numpy as np
import pandas as pd
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from optbinning import BinningProcess


class NamedBinningProcess(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible DataFrame-preserving wrapper around optbinning.BinningProcess.

    Modes
    -----
    output_method:
    - "bin": return the optimized bin labels
    - "bin_ohe": return one-hot-encoded optimized bin labels
    - "woe": return WoE
    - any other valid optbinning metric is also accepted, e.g. "event_rate", "indices", "mean"

    Notes
    -----
    - BinningProcess supports transform metrics including "bins", "indices", and target-dependent
      metrics like "woe", "event_rate", or "mean". :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        variable_names=None,
        binning_fit_params=None,
        categorical_variables=None,
        selection_criteria=None,
        special_codes=None,
        output_method="woe",
        transform_kwargs=None,
        onehot_kwargs=None,
    ):
        self.variable_names = variable_names
        self.binning_fit_params = binning_fit_params
        self.categorical_variables = categorical_variables
        self.selection_criteria = selection_criteria
        self.special_codes = special_codes
        self.output_method = output_method
        self.transform_kwargs = transform_kwargs
        self.onehot_kwargs = onehot_kwargs

    def _bp_kwargs(self):
        kwargs = {}

        if self.variable_names is not None:
            kwargs["variable_names"] = list(self.variable_names)
        if self.binning_fit_params is not None:
            kwargs["binning_fit_params"] = self.binning_fit_params
        if self.categorical_variables is not None:
            kwargs["categorical_variables"] = self.categorical_variables
        if self.selection_criteria is not None:
            kwargs["selection_criteria"] = self.selection_criteria
        if self.special_codes is not None:
            kwargs["special_codes"] = self.special_codes

        return kwargs

    def _resolve_metric(self):
        if self.output_method == "bin":
            return "bins"
        if self.output_method == "bin_ohe":
            return "bins"
        return self.output_method

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.input_cols_ = list(X.columns)

        bp_kwargs = self._bp_kwargs()
        if "variable_names" not in bp_kwargs:
            bp_kwargs["variable_names"] = self.input_cols_

        self.bp_ = BinningProcess(**bp_kwargs)
        self.bp_.fit(X, y)

        # Surviving columns after selection, if any
        try:
            self.base_output_cols_ = list(self.bp_.get_support(names=True))
            if len(self.base_output_cols_) == 0:
                self.base_output_cols_ = list(bp_kwargs["variable_names"])
        except Exception:
            self.base_output_cols_ = list(bp_kwargs["variable_names"])

        if self.output_method == "bin_ohe":
            Xt_bins = self._transform_bins_df(X)
            onehot_kwargs = {} if self.onehot_kwargs is None else dict(self.onehot_kwargs)

            # Dense output for easier DataFrame reconstruction
            if "drop" not in onehot_kwargs:
                onehot_kwargs["drop"] = "first"
            if "sparse_output" not in onehot_kwargs:
                onehot_kwargs["sparse_output"] = False
            if "handle_unknown" not in onehot_kwargs:
                onehot_kwargs["handle_unknown"] = "ignore"

            self.ohe_ = OneHotEncoder(**onehot_kwargs)
            self.ohe_.fit(Xt_bins)

            self.output_cols_ = list(self.ohe_.get_feature_names_out(Xt_bins.columns))
        else:
            self.output_cols_ = list(self.base_output_cols_)

        return self

    def _transform_bins_df(self, X):
        transform_kwargs = {} if self.transform_kwargs is None else dict(self.transform_kwargs)

        Xt = self.bp_.transform(
            X,
            metric="bins",
            **transform_kwargs,
        )

        if isinstance(Xt, pd.DataFrame):
            Xt = Xt.copy()
            Xt.index = X.index
            return Xt.apply(lambda col: col.map(self._normalize_bin_label))

        Xt_df = pd.DataFrame(Xt, columns=self.base_output_cols_, index=X.index)
        return Xt_df.apply(lambda col: col.map(self._normalize_bin_label))

    @staticmethod
    def _normalize_bin_label(value):
        try:
            if pd.isna(value):
                return value
        except Exception:
            pass

        if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
            return str(value.tolist())

        if isinstance(value, (list, tuple, set)):
            return str(list(value))

        if isinstance(value, str):
            value = " ".join(value.split())
            return re.sub(r"\s*Length:\s*\d+\s*,\s*dtype:\s*[^\s]+\s*$", "", value).strip()

        return value

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.input_cols_)

        if self.output_method == "bin_ohe":
            Xt_bins = self._transform_bins_df(X)
            Xt_ohe = self.ohe_.transform(Xt_bins)
            return pd.DataFrame(Xt_ohe, columns=self.output_cols_, index=X.index)

        metric = self._resolve_metric()
        transform_kwargs = {} if self.transform_kwargs is None else dict(self.transform_kwargs)

        Xt = self.bp_.transform(
            X,
            metric=metric,
            **transform_kwargs,
        )

        if isinstance(Xt, pd.DataFrame):
            Xt = Xt.copy()
            Xt.index = X.index
            self.output_cols_ = list(Xt.columns)
            return Xt

        return pd.DataFrame(Xt, columns=self.output_cols_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_cols_, dtype=object)
    
    
class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        self.cols_ = list(X.columns)
        self.scaler_ = self.scaler.fit(X)
        return self

    def transform(self, X):
        Xt = self.scaler_.transform(X)
        return pd.DataFrame(Xt, columns=self.cols_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_)