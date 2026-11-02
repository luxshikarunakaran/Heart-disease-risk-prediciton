"""
================================================================================
HEART DISEASE RISK 2026 — COMPLETE SINGLE-FILE PIPELINE
================================================================================

Everything in one script:
  STEP 0  package installation
  STEP 1  exploratory data analysis
  STEP 2  classical statistical analysis
  STEP 3  train deep tabular models (6 architecture families) + baselines
  STEP 4  metrics, visualisation, significance testing
  STEP 5  LIME explainability (+ SHAP / NAM / TabNet cross-checks)

USAGE
-----
  Colab / Jupyter :  put heart_disease_risk_2026.csv next to this file,
                     rename it data.csv, then run the whole thing.
  Command line    :  python heart_disease_2026_full_pipeline.py
                     python heart_disease_2026_full_pipeline.py --quick
                     python heart_disease_2026_full_pipeline.py --models MLP TabM
                     python heart_disease_2026_full_pipeline.py --steps 1 2

Set INSTALL_PACKAGES = True below on a fresh machine (Colab) to pip-install
everything first.
================================================================================
"""
from __future__ import annotations

# ==============================================================================
# STEP 0 — PACKAGE INSTALLATION
# ==============================================================================
INSTALL_PACKAGES = False      # flip to True on a fresh Colab / new environment


def install_packages():
    """pip-install everything. Optional extras are allowed to fail."""
    import subprocess, sys

    def pip(*args, optional=False):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])
        except Exception as e:
            if not optional:
                print(f"  ! required install failed: {args} -> {e}")
            else:
                print(f"  (optional package unavailable: {args[0]})")

    print("STEP 0 — installing packages")
    pip("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn",
        "joblib", "tqdm", "tabulate")
    pip("statsmodels", "scikit-posthocs")
    pip("torch")
    pip("lime", "shap")
    pip("xgboost", optional=True)
    pip("lightgbm", optional=True)
    pip("catboost", optional=True)
    pip("pytorch-tabnet", optional=True)   # model 7  (official TabNet)
    pip("tabpfn", optional=True)           # models 29/30
    pip("tabicl", optional=True)           # model 31
    pip("ctgan", optional=True)            # models 42/44
    pip("be-great", optional=True)         # model 47
    print("STEP 0 complete.\n")


if INSTALL_PACKAGES:
    install_packages()


# ==============================================================================
# IMPORTS
# ==============================================================================
import argparse
import copy
import importlib
import math
import time
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             fbeta_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import (OrdinalEncoder, QuantileTransformer,
                                   StandardScaler)
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")




# ==============================================================================
# CONFIG
# ==============================================================================
"""Global configuration for the Heart Disease Risk 2026 benchmark."""
from pathlib import Path

ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_CSV = ROOT / "data.csv"
OUT = ROOT / "outputs"
FIGS = ROOT / "figs"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

TARGET = "has_heart_disease"
ID_COL = "patient_id"          # dropped — pure row index, would leak nothing but adds noise

SEED = 42
N_FOLDS = 5                    # stratified outer CV
TEST_SIZE = 0.20               # held-out set carved off before CV
VAL_SIZE = 0.20                # inner split for early stopping

# Deep-model training defaults (overridable per model in registry.py)
MAX_EPOCHS = 200  # lower to 20-30 for a fast pass
PATIENCE = 20
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5

DEVICE = "cuda"                # falls back to cpu automatically in nn_core.py

# Set False to run the full 47-model sweep; True runs a fast subset for smoke tests
QUICK = False
QUICK_MODELS = ["LogisticRegression", "XGBoost", "MLP", "FT-Transformer", "TabM"]



# ==============================================================================
# DATA_PREP
# ==============================================================================
"""
Data loading + preprocessing.

Three preprocessing "views" are produced because different model families need
different inputs:

  view="dense"  -> everything numeric, one-hot for categoricals, quantile-scaled.
                   Used by: MLP-family, CTR models, SSL models, sklearn/GBDT.
  view="mixed"  -> (X_num float tensor, X_cat int tensor) with raw category codes.
                   Used by: TabTransformer, FT-Transformer, SAINT, TabNet, NODE...
  view="raw"    -> the original dataframe with cleaned dtypes.
                   Used by: EDA, statistics, LIME (needs human-readable features),
                   generative models (CTGAN etc. want raw columns + dtypes).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder, StandardScaler



# ----------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    # string-like columns -> bool where possible, else category
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            continue
        u = set(map(str, df[c].dropna().unique()))
        if u <= {"True", "False"}:
            df[c] = df[c].map({"True": True, "False": False}).astype(bool)
        else:
            df[c] = df[c].astype("category")
    return df


def column_types(df: pd.DataFrame):
    """Return (numeric_cols, categorical_cols) excluding the target."""
    feats = [c for c in df.columns if c != TARGET]
    num, cat = [], []
    for c in feats:
        if pd.api.types.is_bool_dtype(df[c]) or not pd.api.types.is_numeric_dtype(df[c]):
            cat.append(c)
        else:
            num.append(c)
    return num, cat


# ----------------------------------------------------------------------
# 2. Feature engineering (clinically motivated, computed BEFORE splitting
#    because every term is a row-wise function — no cross-row statistics,
#    therefore no leakage)
# ----------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["pulse_pressure"] = d["resting_bp_systolic"] - d["resting_bp_diastolic"]
    d["map_bp"] = d["resting_bp_diastolic"] + (d["pulse_pressure"] / 3.0)
    d["chol_hdl_ratio"] = d["cholesterol_total"] / d["hdl"].clip(lower=1)
    d["ldl_hdl_ratio"] = d["ldl"] / d["hdl"].clip(lower=1)
    d["tg_hdl_ratio"] = d["triglycerides"] / d["hdl"].clip(lower=1)
    d["hr_reserve"] = d["max_heart_rate_achieved"] - d["resting_heart_rate"]
    d["pct_max_hr"] = d["max_heart_rate_achieved"] / (220 - d["age"]).clip(lower=1)
    d["non_hdl_chol"] = d["cholesterol_total"] - d["hdl"]
    d["steps_per_exercise_min"] = d["daily_steps"] / d["exercise_minutes_per_week"].clip(lower=1)
    d["metabolic_load"] = (
        (d["bmi"] > 30).astype(int)
        + (d["hba1c"] >= 6.5).astype(int)
        + (d["resting_bp_systolic"] >= 140).astype(int)
        + (d["triglycerides"] >= 150).astype(int)
        + (d["hdl"] < 40).astype(int)
    )
    d["lifestyle_score"] = (
        d["diet_quality_score"] / 100
        + (d["exercise_minutes_per_week"] / 150).clip(upper=2)
        - (d["stress_score"] / 100)
        - (d["alcohol_units_per_week"] / 14).clip(upper=2)
    )
    return d


# ----------------------------------------------------------------------
# 3. Splits
# ----------------------------------------------------------------------
def make_splits(df: pd.DataFrame):
    """Held-out test set + stratified CV folds over the remainder."""
    y = df[TARGET].values
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(tr_idx, y[tr_idx]))
    folds = [(tr_idx[a], tr_idx[b]) for a, b in folds]
    return tr_idx, te_idx, folds


# ----------------------------------------------------------------------
# 4. Preprocessors — fitted on train indices only
# ----------------------------------------------------------------------
class DensePrep:
    """One-hot + quantile-normal transform. Output: float32 matrix."""

    def __init__(self, num, cat, scaler="quantile"):
        self.num, self.cat, self.scaler_kind = num, cat, scaler

    def fit(self, df, tr_idx):
        sub = df.iloc[tr_idx]
        if self.scaler_kind == "quantile":
            n = min(1000, max(10, len(sub) // 30))
            self.sc = QuantileTransformer(
                n_quantiles=n, output_distribution="normal",
                subsample=200_000, random_state=SEED)
        else:
            self.sc = StandardScaler()
        self.sc.fit(sub[self.num].astype(float))
        self.dummy_cols = pd.get_dummies(
            df[self.cat].astype(str), drop_first=False).columns.tolist()
        return self

    def transform(self, df):
        Xn = self.sc.transform(df[self.num].astype(float))
        Xc = pd.get_dummies(df[self.cat].astype(str), drop_first=False)
        Xc = Xc.reindex(columns=self.dummy_cols, fill_value=0).values.astype(float)
        self.feature_names_ = list(self.num) + list(self.dummy_cols)
        return np.hstack([Xn, Xc]).astype(np.float32)

    def fit_transform(self, df, tr_idx):
        return self.fit(df, tr_idx).transform(df)


class MixedPrep:
    """Numeric -> quantile-normal; categorical -> integer codes + cardinalities."""

    def __init__(self, num, cat):
        self.num, self.cat = num, cat

    def fit(self, df, tr_idx):
        sub = df.iloc[tr_idx]
        n = min(1000, max(10, len(sub) // 30))
        self.sc = QuantileTransformer(
            n_quantiles=n, output_distribution="normal",
            subsample=200_000, random_state=SEED).fit(sub[self.num].astype(float))
        self.oe = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit(sub[self.cat].astype(str))
        self.cardinalities = [len(cs) + 1 for cs in self.oe.categories_]
        return self

    def transform(self, df):
        Xn = self.sc.transform(df[self.num].astype(float)).astype(np.float32)
        Xc = self.oe.transform(df[self.cat].astype(str)).astype(np.int64) + 1
        Xc = np.clip(Xc, 0, None)
        return Xn, Xc

    def fit_transform(self, df, tr_idx):
        return self.fit(df, tr_idx).transform(df)


# ----------------------------------------------------------------------
# 5. One-call bundle
# ----------------------------------------------------------------------
class Dataset:
    def __init__(self, feature_engineering=True):
        raw = load_raw()
        self.raw = engineer(raw) if feature_engineering else raw
        self.y = self.raw[TARGET].values.astype(np.int64)
        self.num, self.cat = column_types(self.raw)
        self.tr_idx, self.te_idx, self.folds = make_splits(self.raw)

    def dense(self, fit_idx=None):
        fit_idx = self.tr_idx if fit_idx is None else fit_idx
        p = DensePrep(self.num, self.cat)
        X = p.fit_transform(self.raw, fit_idx)
        return X, p

    def mixed(self, fit_idx=None):
        fit_idx = self.tr_idx if fit_idx is None else fit_idx
        p = MixedPrep(self.num, self.cat)
        Xn, Xc = p.fit_transform(self.raw, fit_idx)
        return Xn, Xc, p

    def summary(self):
        return dict(
            n_rows=len(self.raw), n_num=len(self.num), n_cat=len(self.cat),
            pos_rate=float(self.y.mean()), n_folds=len(self.folds),
            n_test=len(self.te_idx),
        )



# ==============================================================================
# NN_CORE
# ==============================================================================
"""
Shared plumbing for every PyTorch tabular model in this benchmark.

Contract every model obeys:
    forward(x_num: FloatTensor[B, n_num], x_cat: LongTensor[B, n_cat]) -> Tensor[B]
returning raw logits.  Models that ignore one of the inputs simply don't use it.

`TorchTabClassifier` wraps any such module in a scikit-learn-style estimator
(fit / predict_proba / predict) so the whole zoo can be driven by one loop.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================
# Reusable building blocks
# ======================================================================
class CatEmbeddings(nn.Module):
    """Standard entity embeddings for categorical columns."""

    def __init__(self, cardinalities, dim=None):
        super().__init__()
        self.cards = list(cardinalities)
        if dim is None:
            dims = [min(32, max(2, int(round(1.6 * c ** 0.56)))) for c in self.cards]
        else:
            dims = [dim] * len(self.cards)
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(self.cards, dims)])
        self.out_dim = sum(dims)

    def forward(self, x_cat):
        if x_cat is None or x_cat.shape[1] == 0:
            return torch.zeros(x_cat.shape[0], 0, device=x_cat.device)
        return torch.cat([e(x_cat[:, i].clamp(0, self.cards[i] - 1))
                          for i, e in enumerate(self.embs)], dim=1)


class FeatureTokenizer(nn.Module):
    """
    FT-Transformer style tokenizer: every feature (numeric or categorical)
    becomes one d-dimensional token. Numeric features get a learned weight
    vector times the scalar, plus a bias; categoricals get an embedding table.
    """

    def __init__(self, n_num, cardinalities, d_token):
        super().__init__()
        self.n_num, self.d = n_num, d_token
        self.cards = list(cardinalities)
        if n_num:
            self.w_num = nn.Parameter(torch.empty(n_num, d_token))
            self.b_num = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.uniform_(self.w_num, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
            nn.init.uniform_(self.b_num, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        self.cat_emb = nn.ModuleList([nn.Embedding(c, d_token) for c in self.cards])
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.uniform_(self.cls, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))

    @property
    def n_tokens(self):
        return self.n_num + len(self.cards) + 1

    def forward(self, x_num, x_cat, with_cls=True):
        toks = []
        if self.n_num:
            toks.append(x_num.unsqueeze(-1) * self.w_num + self.b_num)   # B,n_num,d
        if len(self.cards):
            toks.append(torch.stack(
                [e(x_cat[:, i].clamp(0, self.cards[i] - 1))
                 for i, e in enumerate(self.cat_emb)], dim=1))           # B,n_cat,d
        t = torch.cat(toks, dim=1)
        if with_cls:
            t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        return t


class PeriodicEmbeddings(nn.Module):
    """
    Numeric embeddings from Gorishniy et al. 2022 — sin/cos of learned
    frequencies followed by a linear layer. Used by RealMLP / FT-T variants.
    """

    def __init__(self, n_features, d_embedding=24, n_frequencies=24, sigma=0.05):
        super().__init__()
        self.freqs = nn.Parameter(torch.normal(0., sigma, (n_features, n_frequencies)))
        self.lin = nn.Parameter(torch.empty(n_features, 2 * n_frequencies, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        for p in (self.lin, self.bias):
            nn.init.uniform_(p, -1 / math.sqrt(d_embedding), 1 / math.sqrt(d_embedding))
        self.out_dim = n_features * d_embedding

    def forward(self, x):                       # B,F -> B,F,d
        z = 2 * math.pi * self.freqs * x.unsqueeze(-1)
        z = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        z = torch.einsum("bfk,fkd->bfd", z, self.lin) + self.bias
        return F.relu(z)


class MLPBlock(nn.Module):
    def __init__(self, din, dout, dropout=0.1, norm="batch", act="relu"):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.norm = (nn.BatchNorm1d(dout) if norm == "batch"
                     else nn.LayerNorm(dout) if norm == "layer" else nn.Identity())
        self.act = {"relu": nn.ReLU(), "gelu": nn.GELU(),
                    "selu": nn.SELU(), "mish": nn.Mish()}[act]
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.lin(x))))


class GEGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


# ======================================================================
# Regularisation utilities (used by Regularization Cocktails, #2)
# ======================================================================
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_features(x, prob=0.1):
    """Feature-wise cutmix / swap-noise: replace a random subset of columns
    with values drawn from other rows in the batch."""
    mask = torch.rand_like(x) < prob
    idx = torch.randperm(x.size(0), device=x.device)
    return torch.where(mask, x[idx], x)


class StochasticDepth(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x, residual):
        if not self.training or self.p == 0:
            return x + residual
        keep = torch.rand(x.size(0), 1, device=x.device) > self.p
        return x + residual * keep / (1 - self.p)


# ======================================================================
# The universal estimator wrapper
# ======================================================================
class TorchTabClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps any (x_num, x_cat) -> logits module.

    Parameters
    ----------
    build_fn : callable(n_num, cardinalities, **kw) -> nn.Module
    kw       : passed to build_fn
    """

    def __init__(self, build_fn, name="torch_model", lr=LR,
                 weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 class_weight=True, label_smoothing=0.0,
                 use_mixup=False, mixup_alpha=0.4, swap_noise=0.0,
                 grad_clip=1.0, scheduler="cosine", warmup_frac=0.05,
                 swa=False, verbose=False, **kw):
        self.build_fn = build_fn
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.swap_noise = swap_noise
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.warmup_frac = warmup_frac
        self.swa = swa
        self.verbose = verbose
        self.kw = kw

    # ------------------------------------------------------------------
    def _loaders(self, Xn, Xc, y, shuffle):
        t = [torch.as_tensor(Xn, dtype=torch.float32),
             torch.as_tensor(Xc, dtype=torch.long),
             torch.as_tensor(y, dtype=torch.float32)]
        return DataLoader(TensorDataset(*t), batch_size=self.batch_size,
                          shuffle=shuffle, drop_last=shuffle and len(y) > self.batch_size)

    def fit(self, X, y, eval_set=None):
        seed_all(SEED)
        Xn, Xc = X
        self.n_num_ = Xn.shape[1]
        self.cards_ = self.kw.pop("cardinalities", None) or self._infer_cards(Xc)
        self.classes_ = np.array([0, 1])

        self.model_ = self.build_fn(self.n_num_, self.cards_, **self.kw).to(DEVICE)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        tr_loader = self._loaders(Xn, Xc, y, True)
        steps = max(1, len(tr_loader)) * self.max_epochs
        if self.scheduler == "cosine":
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, total_steps=steps,
                pct_start=self.warmup_frac, anneal_strategy="cos")
        else:
            sch = None

        pw = None
        if self.class_weight:
            pos = max(1, int(y.sum()))
            pw = torch.tensor([(len(y) - pos) / pos], device=DEVICE)

        if eval_set is None:                       # carve an internal val split
            from sklearn.model_selection import train_test_split
            i = np.arange(len(y))
            a, b = train_test_split(i, test_size=VAL_SIZE, stratify=y,
                                    random_state=SEED)
            eval_set = ((Xn[b], Xc[b]), y[b])
            tr_loader = self._loaders(Xn[a], Xc[a], y[a], True)

        (Vn, Vc), vy = eval_set
        best, best_state, bad = -np.inf, None, 0
        swa_state, n_swa = None, 0

        for ep in range(self.max_epochs):
            self.model_.train()
            for xn, xc, yy in tr_loader:
                xn, xc, yy = xn.to(DEVICE), xc.to(DEVICE), yy.to(DEVICE)
                if self.swap_noise > 0:
                    xn = cutmix_features(xn, self.swap_noise)
                opt.zero_grad(set_to_none=True)
                if self.use_mixup:
                    xm, ya, yb, lam = mixup(xn, yy, self.mixup_alpha)
                    out = self.model_(xm, xc)
                    loss = (lam * F.binary_cross_entropy_with_logits(out, ya, pos_weight=pw)
                            + (1 - lam) * F.binary_cross_entropy_with_logits(out, yb, pos_weight=pw))
                else:
                    tgt = yy * (1 - self.label_smoothing) + .5 * self.label_smoothing
                    loss = F.binary_cross_entropy_with_logits(
                        self.model_(xn, xc), tgt, pos_weight=pw)
                if hasattr(self.model_, "aux_loss"):
                    loss = loss + self.model_.aux_loss()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                opt.step()
                if sch is not None:
                    sch.step()

            score = roc_auc_score(vy, self._raw_proba(Vn, Vc))
            if self.swa and ep >= int(.75 * self.max_epochs):
                sd = {k: v.detach().float().cpu().clone()
                      for k, v in self.model_.state_dict().items()}
                swa_state = sd if swa_state is None else \
                    {k: swa_state[k] + sd[k] for k in sd}
                n_swa += 1
            if score > best + 1e-5:
                best, bad = score, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and ep % 10 == 0:
                print(f"    [{self.name}] ep{ep:3d} val_auc={score:.4f}")

        if self.swa and n_swa > 0:
            self.model_.load_state_dict({k: (v / n_swa).to(DEVICE)
                                         for k, v in swa_state.items()})
        elif best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_auc_ = best
        self.n_params_ = sum(p.numel() for p in self.model_.parameters())
        return self

    @staticmethod
    def _infer_cards(Xc):
        return [int(Xc[:, i].max()) + 2 for i in range(Xc.shape[1])]

    @torch.no_grad()
    def _raw_proba(self, Xn, Xc):
        self.model_.eval()
        out = []
        for i in range(0, len(Xn), 4096):
            xn = torch.as_tensor(Xn[i:i + 4096], dtype=torch.float32, device=DEVICE)
            xc = torch.as_tensor(Xc[i:i + 4096], dtype=torch.long, device=DEVICE)
            out.append(torch.sigmoid(self.model_(xn, xc)).cpu().numpy())
        return np.concatenate(out)

    def predict_proba(self, X):
        Xn, Xc = X
        p = self._raw_proba(Xn, Xc)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)



# ==============================================================================
# MLP_FAMILY
# ==============================================================================
"""
MLP-family models (registry entries 1–6).

1. MLP                      — plain deep net baseline
2. Regularization Cocktails — Kadra et al. 2021: plain MLP + a stack of regularisers
3. RealMLP                  — Holzmüller et al. 2024: scaling layer, periodic
                              numeric embeddings, NTK-parametrised linear layers,
                              label smoothing, SWA-like averaging
4. TabM                     — Gorishniy et al. 2024: parameter-efficient ensemble
                              of k MLPs via BatchEnsemble-style rank-1 adapters
5. NAM                      — Agarwal et al. 2021: one subnetwork per feature,
                              outputs summed -> fully interpretable shape functions
6. NODE-GAM               — Chang et al. 2022: oblivious differentiable trees
                              restricted to per-feature (GAM) structure
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



# ----------------------------------------------------------------------
# 1. MLP
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_num, cardinalities, hidden=(256, 256, 128),
                 dropout=0.15, norm="batch", act="relu"):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities)
        d = n_num + self.emb.out_dim
        layers, prev = [], d
        for h in hidden:
            layers.append(MLPBlock(prev, h, dropout, norm, act))
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        return self.head(self.body(x)).squeeze(-1)


# ----------------------------------------------------------------------
# 2. Regularization Cocktails
# ----------------------------------------------------------------------
class RegCocktailMLP(nn.Module):
    """
    Plain MLP + shaped residual blocks + stochastic depth + shake-shake style
    branch mixing. The remaining cocktail ingredients (mixup, cutmix/swap-noise,
    weight decay, SWA, LR scheduling, label smoothing) live in the trainer and
    are switched on in registry.py.
    """

    def __init__(self, n_num, cardinalities, width=256, n_blocks=4,
                 dropout=0.2, sd_prob=0.1, shake=True):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities)
        self.inp = nn.Linear(n_num + self.emb.out_dim, width)
        self.shake = shake
        self.blocks = nn.ModuleList()
        self.sd = nn.ModuleList()
        for _ in range(n_blocks):
            branch = nn.Sequential(
                nn.BatchNorm1d(width), nn.ReLU(),
                nn.Linear(width, width), nn.Dropout(dropout),
                nn.BatchNorm1d(width), nn.ReLU(),
                nn.Linear(width, width),
            )
            branch2 = nn.Sequential(
                nn.BatchNorm1d(width), nn.ReLU(),
                nn.Linear(width, width), nn.Dropout(dropout),
                nn.BatchNorm1d(width), nn.ReLU(),
                nn.Linear(width, width),
            ) if shake else None
            self.blocks.append(nn.ModuleList([branch, branch2]))
            self.sd.append(StochasticDepth(sd_prob))
        self.head = nn.Sequential(nn.BatchNorm1d(width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x_num, x_cat):
        x = self.inp(torch.cat([x_num, self.emb(x_cat)], dim=1))
        for (b1, b2), sd in zip(self.blocks, self.sd):
            r = b1(x)
            if b2 is not None:
                a = torch.rand(x.size(0), 1, device=x.device) if self.training else 0.5
                r = a * r + (1 - a) * b2(x)
            x = sd(x, r)
        return self.head(x).squeeze(-1)


# ----------------------------------------------------------------------
# 3. RealMLP
# ----------------------------------------------------------------------
class ScalingLayer(nn.Module):
    """Learned per-feature scale — RealMLP's soft feature-selection front end."""

    def __init__(self, n):
        super().__init__()
        self.s = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return x * self.s


class NTKLinear(nn.Module):
    """NTK-parametrised linear layer: forward scaled by 1/sqrt(fan_in) so that
    a single global LR works across widths."""

    def __init__(self, din, dout, bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dout, din))
        self.b = nn.Parameter(torch.zeros(dout)) if bias else None
        self.scale = 1.0 / math.sqrt(din)

    def forward(self, x):
        y = F.linear(x, self.w * self.scale)
        return y + self.b if self.b is not None else y


class RealMLP(nn.Module):
    def __init__(self, n_num, cardinalities, hidden=(256, 256, 256),
                 d_periodic=16, dropout=0.15):
        super().__init__()
        self.scale = ScalingLayer(n_num) if n_num else None
        self.per = PeriodicEmbeddings(n_num, d_embedding=d_periodic,
                                      n_frequencies=16) if n_num else None
        self.emb = CatEmbeddings(cardinalities, dim=8)
        din = n_num + (self.per.out_dim if self.per else 0) + self.emb.out_dim
        layers, prev = [], din
        for h in hidden:
            layers += [NTKLinear(prev, h), nn.SELU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = NTKLinear(prev, 1)

    def forward(self, x_num, x_cat):
        parts = []
        if self.scale is not None:
            xs = self.scale(x_num)
            parts += [xs, self.per(xs).flatten(1)]
        parts.append(self.emb(x_cat))
        return self.head(self.body(torch.cat(parts, dim=1))).squeeze(-1)


# ----------------------------------------------------------------------
# 4. TabM
# ----------------------------------------------------------------------
class BatchEnsembleLinear(nn.Module):
    """One shared weight matrix + k rank-1 (r, s) adapters => k virtual models
    at ~1x the parameter cost. This is the core trick behind TabM."""

    def __init__(self, din, dout, k=8):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(din, dout)
        self.r = nn.Parameter(torch.randn(k, din) * 0.1 + 1.0)
        self.s = nn.Parameter(torch.randn(k, dout) * 0.1 + 1.0)
        self.b = nn.Parameter(torch.zeros(k, dout))

    def forward(self, x):                    # x: B,k,din -> B,k,dout
        return self.lin(x * self.r) * self.s + self.b


class TabM(nn.Module):
    def __init__(self, n_num, cardinalities, hidden=(256, 256), k=8, dropout=0.1):
        super().__init__()
        self.k = k
        self.emb = CatEmbeddings(cardinalities)
        din = n_num + self.emb.out_dim
        self.layers = nn.ModuleList()
        prev = din
        for h in hidden:
            self.layers.append(BatchEnsembleLinear(prev, h, k))
            prev = h
        self.drop = nn.Dropout(dropout)
        self.heads = BatchEnsembleLinear(prev, 1, k)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        x = x.unsqueeze(1).expand(-1, self.k, -1)          # B,k,din
        for l in self.layers:
            x = self.drop(F.relu(l(x)))
        out = self.heads(x).squeeze(-1)                    # B,k
        self._member_logits = out
        return out.mean(dim=1)                             # ensemble average


# ----------------------------------------------------------------------
# 5. NAM — Neural Additive Model
# ----------------------------------------------------------------------
class ExU(nn.Module):
    """Exp-centred unit from the NAM paper: f(x) = act((x - b) * exp(w)).
    Lets a single unit represent very sharp, jumpy shape functions."""

    def __init__(self, din, dout):
        super().__init__()
        self.w = nn.Parameter(torch.empty(din, dout))
        self.b = nn.Parameter(torch.zeros(din))
        nn.init.normal_(self.w, 4.0, 0.5)

    def forward(self, x):
        return torch.clamp(((x - self.b) @ torch.exp(self.w)), 0, 1)  # ReLU-n, n=1


class FeatureNet(nn.Module):
    def __init__(self, hidden=(64, 32), dropout=0.05, use_exu=True):
        super().__init__()
        first = ExU(1, hidden[0]) if use_exu else nn.Sequential(nn.Linear(1, hidden[0]), nn.ReLU())
        rest = []
        for a, b in zip(hidden, hidden[1:]):
            rest += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(first, *rest, nn.Linear(hidden[-1], 1))

    def forward(self, x):                  # B,1 -> B,1
        return self.net(x)


class NAM(nn.Module):
    """
    y = b0 + sum_j f_j(x_j).
    Every f_j is a small independent net, so the model is exactly a GAM and
    each f_j can be plotted as a shape function (see 05_lime.py).
    """

    def __init__(self, n_num, cardinalities, hidden=(64, 32),
                 dropout=0.05, l2_shape=1e-5, use_exu=True):
        super().__init__()
        self.n_num = n_num
        self.cards = list(cardinalities)
        self.nets = nn.ModuleList([FeatureNet(hidden, dropout, use_exu)
                                   for _ in range(n_num)])
        self.cat_tables = nn.ModuleList([nn.Embedding(c, 1) for c in self.cards])
        self.bias = nn.Parameter(torch.zeros(1))
        self.l2_shape = l2_shape

    def contributions(self, x_num, x_cat):
        parts = [self.nets[j](x_num[:, j:j + 1]) for j in range(self.n_num)]
        parts += [t(x_cat[:, i].clamp(0, self.cards[i] - 1))
                  for i, t in enumerate(self.cat_tables)]
        return torch.cat(parts, dim=1)               # B, n_features

    def forward(self, x_num, x_cat):
        self._contrib = self.contributions(x_num, x_cat)
        return self._contrib.sum(1) + self.bias

    def aux_loss(self):
        return self.l2_shape * (self._contrib ** 2).mean()


# ----------------------------------------------------------------------
# 6. NODE-GAM
# ----------------------------------------------------------------------
def entmax15(x, dim=-1):
    """1.5-entmax — sparse alternative to softmax used by NODE / TabNet."""
    x = x / 2
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)
    rho = torch.arange(1, x.shape[dim] + 1, device=x.device,
                       dtype=x.dtype).view([-1 if i == dim % x.dim() else 1
                                            for i in range(x.dim())])
    mean = x_sorted.cumsum(dim) / rho
    mean_sq = (x_sorted ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = torch.clamp((1 - ss) / rho, min=0)
    tau = mean - torch.sqrt(delta)
    support = (tau <= x_sorted).sum(dim=dim, keepdim=True)
    # numerical guard: an all-equal / degenerate row can yield support==0,
    # which would index -1 and crash the gather
    support = support.clamp(min=1, max=x.shape[dim])
    tau_star = tau.gather(dim, support - 1)
    return torch.clamp(x - tau_star, min=0) ** 2


class ObliviousTreeLayer(nn.Module):
    """
    Differentiable oblivious decision trees (NODE, Popov et al. 2020).
    Each tree uses `depth` shared split features; the leaf response is a
    soft product of per-level gates.

    `gam_mode=True` forces every tree to look at a single input feature,
    which turns the ensemble into a GAM (NODE-GAM).
    """

    def __init__(self, in_dim, n_trees=64, depth=4, out_dim=1,
                 gam_mode=False, feature_subset=None):
        super().__init__()
        self.n_trees, self.depth, self.out_dim = n_trees, depth, out_dim
        self.gam_mode = gam_mode
        self.feature_selection = nn.Parameter(torch.randn(in_dim, n_trees, depth))
        if gam_mode:
            # one fixed feature per tree -> additive structure
            fs = torch.zeros(in_dim, n_trees, depth)
            idx = (feature_subset if feature_subset is not None
                   else torch.arange(n_trees) % in_dim)
            for t in range(n_trees):
                fs[idx[t], t, :] = 8.0
            self.register_buffer("gam_mask", fs)
        self.thresholds = nn.Parameter(torch.randn(n_trees, depth))
        self.temps = nn.Parameter(torch.ones(n_trees, depth))
        self.leaves = nn.Parameter(torch.randn(n_trees, 2 ** depth, out_dim) * 0.1)
        bins = torch.arange(2 ** depth)
        codes = torch.stack([(bins >> d) & 1 for d in range(depth)], dim=1)
        self.register_buffer("codes", codes.float())      # 2^depth, depth

    def forward(self, x, reduce="none"):                   # B,in_dim
        """
        reduce="none" -> B, n_trees, out_dim   (NODE: every tree kept separate)
        reduce="sum"  -> B, out_dim            (NODE-GAM: additive ensemble)
        """
        logits = self.gam_mask if self.gam_mode else self.feature_selection
        w = entmax15(logits, dim=0)                        # in_dim,T,depth
        f = torch.einsum("bi,itd->btd", x, w)              # B,T,depth
        h = (f - self.thresholds) * torch.abs(self.temps).clamp(min=1e-3)
        p = torch.sigmoid(h)                               # B,T,depth  (go-right prob)
        # leaf probability = prod over depth of p or 1-p according to leaf code
        pc = p.unsqueeze(2) * self.codes + (1 - p).unsqueeze(2) * (1 - self.codes)
        leaf_p = pc.prod(dim=-1)                           # B,T,2^depth
        out = torch.einsum("btl,tlo->bto", leaf_p, self.leaves)   # B,T,out_dim
        return out.sum(1) if reduce == "sum" else out


class NODEGAM(nn.Module):
    """Stack of GAM-constrained oblivious tree layers, one group per feature."""

    def __init__(self, n_num, cardinalities, n_layers=2, trees_per_feature=4,
                 depth=4):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        self.n_in = n_num + self.emb.out_dim
        n_trees = self.n_in * trees_per_feature
        subset = torch.arange(n_trees) // trees_per_feature
        self.layers = nn.ModuleList([
            ObliviousTreeLayer(self.n_in, n_trees, depth, 1,
                               gam_mode=True, feature_subset=subset)
            for _ in range(n_layers)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        return sum(l(x, reduce="sum").squeeze(-1) for l in self.layers) + self.bias



# ==============================================================================
# ATTENTION
# ==============================================================================
"""
Attention / Transformer family (registry entries 7–16).

 7. TabNet          — sequential attentive feature selection with sparsemax masks
 8. TabTransformer  — transformer over categorical embeddings, numerics concatenated
 9. FT-Transformer  — every feature (num + cat) becomes a token; [CLS] readout
10. SAINT          — alternates self-attention over features and over ROWS
11. AutoInt        — multi-head self-attention with residual for feature interaction
12. T2G-Former     — learns a feature-relation graph, attention masked by it
13. ExcelFormer    — semi-permeable attention + attentive/interaction mixing
14. ARM-Net        — adaptive relation modelling: exponential neurons in log space
15. NPT            — non-parametric transformer: attends across datapoints AND
                     attributes, so predictions use other rows explicitly
16. TransTab       — column-name-aware tokenisation (text encoder over headers)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def sparsemax(x, dim=-1):
    srt, _ = torch.sort(x, dim=dim, descending=True)
    rng = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
    shape = [-1 if i == dim % x.dim() else 1 for i in range(x.dim())]
    rng = rng.view(shape)
    cum = srt.cumsum(dim) - 1
    support = (rng * srt > cum).to(x.dtype)
    k = support.sum(dim=dim, keepdim=True)
    tau = (cum * support).sum(dim=dim, keepdim=True) / k
    return torch.clamp(x - tau, min=0)


# ----------------------------------------------------------------------
# 7. TabNet
# ----------------------------------------------------------------------
class GLUBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.fc = nn.Linear(din, 2 * dout, bias=False)
        self.bn = nn.BatchNorm1d(2 * dout, momentum=0.02)

    def forward(self, x):
        a, b = self.bn(self.fc(x)).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class FeatTransformer(nn.Module):
    def __init__(self, din, dout, n_shared=2, n_indep=2, shared=None):
        super().__init__()
        blocks = []
        d = din
        for i in range(n_shared):
            blocks.append(shared[i] if shared else GLUBlock(d, dout))
            d = dout
        for _ in range(n_indep):
            blocks.append(GLUBlock(d, dout))
            d = dout
        self.blocks = nn.ModuleList(blocks)
        self.scale = math.sqrt(0.5)

    def forward(self, x):
        x = self.blocks[0](x)
        for b in self.blocks[1:]:
            x = (x + b(x)) * self.scale
        return x


class TabNet(nn.Module):
    def __init__(self, n_num, cardinalities, n_d=32, n_a=32, n_steps=4,
                 gamma=1.3, lambda_sparse=1e-3):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities)
        d_in = n_num + self.emb.out_dim
        self.d_in, self.n_d, self.n_a = d_in, n_d, n_a
        self.n_steps, self.gamma, self.lambda_sparse = n_steps, gamma, lambda_sparse
        self.bn = nn.BatchNorm1d(d_in, momentum=0.02)
        shared = [GLUBlock(d_in, n_d + n_a), GLUBlock(n_d + n_a, n_d + n_a)]
        self.initial = FeatTransformer(d_in, n_d + n_a, shared=shared)
        self.steps = nn.ModuleList([
            FeatTransformer(d_in, n_d + n_a, shared=shared) for _ in range(n_steps)])
        self.attn = nn.ModuleList([
            nn.Sequential(nn.Linear(n_a, d_in, bias=False),
                          nn.BatchNorm1d(d_in, momentum=0.02))
            for _ in range(n_steps)])
        self.head = nn.Linear(n_d, 1)

    def forward(self, x_num, x_cat):
        x = self.bn(torch.cat([x_num, self.emb(x_cat)], dim=1))
        prior = torch.ones_like(x)
        a = self.initial(x)[:, self.n_d:]
        out, sparse_loss = 0, 0
        self.masks_ = []
        for i in range(self.n_steps):
            m = sparsemax(self.attn[i](a) * prior)
            self.masks_.append(m.detach())
            sparse_loss = sparse_loss + (-m * torch.log(m + 1e-10)).sum(1).mean()
            prior = prior * (self.gamma - m)
            z = self.steps[i](x * m)
            d, a = z[:, :self.n_d], z[:, self.n_d:]
            out = out + F.relu(d)
        self._sparse = sparse_loss / self.n_steps
        return self.head(out).squeeze(-1)

    def aux_loss(self):
        return self.lambda_sparse * self._sparse


# ----------------------------------------------------------------------
# shared transformer block
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d, heads=8, dropout=0.1, ff_mult=4, prenorm=True,
                 geglu=True, attn_mask_module=None):
        super().__init__()
        self.prenorm = prenorm
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        inner = ff_mult * d
        self.ff = (nn.Sequential(nn.Linear(d, inner * 2), GEGLU(),
                                 nn.Dropout(dropout), nn.Linear(inner, d))
                   if geglu else
                   nn.Sequential(nn.Linear(d, inner), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(inner, d)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        h = self.n1(x) if self.prenorm else x
        a, w = self.attn(h, h, h, attn_mask=attn_mask, need_weights=True,
                         average_attn_weights=True)
        self.attn_weights_ = w.detach()
        x = x + self.drop(a)
        if not self.prenorm:
            x = self.n1(x)
        h = self.n2(x) if self.prenorm else x
        x = x + self.drop(self.ff(h))
        return x if self.prenorm else self.n2(x)


# ----------------------------------------------------------------------
# 8. TabTransformer
# ----------------------------------------------------------------------
class TabTransformer(nn.Module):
    """Transformer contextualises the CATEGORICAL embeddings only; the
    (layer-normed) numeric columns are concatenated straight into the MLP head."""

    def __init__(self, n_num, cardinalities, d_token=32, depth=6, heads=8,
                 dropout=0.1, mlp_hidden=(128, 64)):
        super().__init__()
        self.cards = list(cardinalities)
        self.embs = nn.ModuleList([nn.Embedding(c, d_token) for c in self.cards])
        self.col_emb = nn.Parameter(torch.randn(len(self.cards), d_token) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_token, heads, dropout)
                                     for _ in range(depth)])
        self.num_norm = nn.LayerNorm(n_num) if n_num else None
        d = len(self.cards) * d_token + n_num
        layers, prev = [], d
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.head = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, x_num, x_cat):
        t = torch.stack([e(x_cat[:, i].clamp(0, self.cards[i] - 1))
                         for i, e in enumerate(self.embs)], dim=1) + self.col_emb
        for b in self.blocks:
            t = b(t)
        z = [t.flatten(1)]
        if self.num_norm is not None:
            z.append(self.num_norm(x_num))
        return self.head(torch.cat(z, dim=1)).squeeze(-1)


# ----------------------------------------------------------------------
# 9. FT-Transformer
# ----------------------------------------------------------------------
class FTTransformer(nn.Module):
    def __init__(self, n_num, cardinalities, d_token=96, depth=4, heads=8,
                 dropout=0.15, ff_mult=2):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, heads, dropout, ff_mult) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        t = self.tok(x_num, x_cat, with_cls=True)
        for b in self.blocks:
            t = b(t)
        return self.head(self.norm(t[:, 0])).squeeze(-1)


# ----------------------------------------------------------------------
# 10. SAINT
# ----------------------------------------------------------------------
class SAINT(nn.Module):
    """Interleaves standard self-attention over features with 'intersample'
    attention where the batch dimension is attended over."""

    def __init__(self, n_num, cardinalities, d_token=32, depth=3, heads=4,
                 dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_token)
        n_tok = self.tok.n_tokens
        self.col_blocks = nn.ModuleList([TransformerBlock(d_token, heads, dropout)
                                         for _ in range(depth)])
        self.row_blocks = nn.ModuleList([TransformerBlock(n_tok * d_token,
                                                          heads=1, dropout=dropout,
                                                          ff_mult=1)
                                         for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(),
                                  nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        t = self.tok(x_num, x_cat, with_cls=True)           # B,T,d
        B, T, d = t.shape
        for cb, rb in zip(self.col_blocks, self.row_blocks):
            t = cb(t)                                        # attend over features
            flat = t.reshape(1, B, T * d)                    # treat batch as sequence
            flat = rb(flat)
            t = flat.reshape(B, T, d)
        return self.head(t[:, 0]).squeeze(-1)


# ----------------------------------------------------------------------
# 11. AutoInt
# ----------------------------------------------------------------------
class AutoInt(nn.Module):
    """Multi-head self-attention with residual over feature embeddings; final
    prediction is a linear map of the flattened interacting embeddings
    (+ optional deep tower)."""

    def __init__(self, n_num, cardinalities, d_token=32, n_layers=3, heads=4,
                 dropout=0.1, with_deep=True, hidden=(256, 128)):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_token)
        self.att = nn.ModuleList([
            nn.MultiheadAttention(d_token, heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)])
        self.res = nn.ModuleList([nn.Linear(d_token, d_token) for _ in range(n_layers)])
        n_tok = self.tok.n_tokens - 1
        self.out = nn.Linear(n_tok * d_token, 1)
        self.deep = None
        if with_deep:
            layers, prev = [], n_tok * d_token
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            self.deep = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, x_num, x_cat):
        t = self.tok(x_num, x_cat, with_cls=False)
        for a, r in zip(self.att, self.res):
            h, _ = a(t, t, t, need_weights=False)
            t = F.relu(h + r(t))
        flat = t.flatten(1)
        y = self.out(flat)
        if self.deep is not None:
            y = y + self.deep(flat)
        return y.squeeze(-1)


# ----------------------------------------------------------------------
# 12. T2G-Former
# ----------------------------------------------------------------------
class T2GBlock(nn.Module):
    """Graph Estimator produces a sparse feature-relation adjacency; attention
    is then restricted to that graph."""

    def __init__(self, d, heads=4, dropout=0.1, n_tokens=None):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.fr = nn.Parameter(torch.randn(n_tokens, d) * 0.02)   # relation embeds
        self.col_norm = nn.LayerNorm(d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def graph(self):
        f = self.col_norm(self.fr)
        adj = entmax15(f @ f.t() / math.sqrt(f.shape[-1]), dim=-1)
        return adj

    def forward(self, x):
        B, T, d = x.shape
        h = self.n1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        q = q.view(B, T, self.heads, -1).transpose(1, 2)
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        adj = self.graph()[:T, :T]
        self.adj_ = adj.detach()
        att = att.softmax(-1) * adj.unsqueeze(0).unsqueeze(0)
        att = att / (att.sum(-1, keepdim=True) + 1e-9)
        o = (att @ v).transpose(1, 2).reshape(B, T, d)
        x = x + self.drop(self.o(o))
        return x + self.drop(self.ff(self.n2(x)))


class T2GFormer(nn.Module):
    def __init__(self, n_num, cardinalities, d_token=64, depth=3, heads=4,
                 dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_token)
        T = self.tok.n_tokens
        self.blocks = nn.ModuleList([T2GBlock(d_token, heads, dropout, T)
                                     for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(),
                                  nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        t = self.tok(x_num, x_cat, with_cls=True)
        for b in self.blocks:
            t = b(t)
        return self.head(t[:, 0]).squeeze(-1)


# ----------------------------------------------------------------------
# 13. ExcelFormer
# ----------------------------------------------------------------------
class SemiPermeableAttention(nn.Module):
    """Information flows only from less-informative to more-informative
    features (ranked by mutual information), which suppresses rotational
    invariance issues on tabular data."""

    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rank_mask):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, self.h, -1).transpose(1, 2)
        k = k.view(B, T, self.h, -1).transpose(1, 2)
        v = v.view(B, T, self.h, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        att = att.masked_fill(rank_mask[:T, :T], float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, d)
        return self.drop(self.o(o))


class ExcelFormer(nn.Module):
    def __init__(self, n_num, cardinalities, d_token=64, depth=3, heads=4,
                 dropout=0.1, feature_order=None):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_token)
        T = self.tok.n_tokens
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("rank_mask", mask)
        self.att = nn.ModuleList([SemiPermeableAttention(d_token, heads, dropout)
                                  for _ in range(depth)])
        # attentive / interaction module: element-wise gated feature mixing
        self.mix = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(d_token), nn.Linear(d_token, 2 * d_token), GEGLU(),
            nn.Linear(d_token, d_token)) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(),
                                  nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        t = self.tok(x_num, x_cat, with_cls=True)
        for a, m, n in zip(self.att, self.mix, self.norms):
            t = t + a(n(t), self.rank_mask)
            t = t + m(t)
        return self.head(t.mean(1)).squeeze(-1)


# ----------------------------------------------------------------------
# 14. ARM-Net
# ----------------------------------------------------------------------
class ARMNet(nn.Module):
    """
    Adaptive Relation Modelling. Multiplicative feature interactions of
    arbitrary order are modelled in log space:
        o_k = exp( sum_j w_kj * log|e_j| )
    with the weights w produced by sparse (entmax) attention over features.
    """

    def __init__(self, n_num, cardinalities, d_token=16, n_exp=32, alpha=1.7,
                 hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_token)
        T = self.tok.n_tokens - 1
        self.T, self.n_exp = T, n_exp
        self.q = nn.Parameter(torch.randn(n_exp, d_token) * 0.1)
        self.k = nn.Linear(d_token, d_token)
        self.alpha = alpha
        self.bn = nn.BatchNorm1d(n_exp * d_token)
        layers, prev = [], n_exp * d_token
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers, nn.Linear(prev, 1))
        self.wide = nn.Linear(T * d_token, 1)

    def forward(self, x_num, x_cat):
        e = self.tok(x_num, x_cat, with_cls=False)             # B,T,d
        k = self.k(e)                                          # B,T,d
        att = torch.einsum("ed,btd->bet", self.q, k) / math.sqrt(k.shape[-1])
        w = entmax15(att * self.alpha, dim=-1)                 # B,n_exp,T  sparse
        log_e = torch.log(torch.abs(e) + 1e-7)                 # B,T,d
        o = torch.exp(torch.einsum("bet,btd->bed", w, log_e))  # B,n_exp,d
        o = self.bn(o.flatten(1))
        return (self.mlp(o) + self.wide(e.flatten(1))).squeeze(-1)


# ----------------------------------------------------------------------
# 15. NPT — Non-Parametric Transformer
# ----------------------------------------------------------------------
class NPT(nn.Module):
    """
    Attends between DATAPOINTS as well as attributes, so a prediction for one
    row can explicitly reference other rows in the batch (a learned,
    end-to-end analogue of k-NN). Batch order is irrelevant by construction.
    """

    def __init__(self, n_num, cardinalities, d_token=32, depth=3, heads=4,
                 dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_token)
        T = self.tok.n_tokens
        self.abd = nn.ModuleList([TransformerBlock(d_token, heads, dropout)
                                  for _ in range(depth)])   # attention between attributes
        self.abda = nn.ModuleList([TransformerBlock(T * d_token, 1, dropout, ff_mult=1)
                                   for _ in range(depth)])  # ...between datapoints
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(),
                                  nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        t = self.tok(x_num, x_cat, with_cls=True)
        B, T, d = t.shape
        for a1, a2 in zip(self.abd, self.abda):
            t = a1(t)
            t = a2(t.reshape(1, B, T * d)).reshape(B, T, d)
        return self.head(t[:, 0]).squeeze(-1)


# ----------------------------------------------------------------------
# 16. TransTab
# ----------------------------------------------------------------------
class TransTab(nn.Module):
    """
    Column-name-aware tokenisation. Each column contributes
        token_j = ColumnNameEmbedding(j) * value_gate + ValueEmbedding
    so the model transfers across tables with different schemas. Here the
    column-name embeddings are initialised from a hashed bag-of-characters of
    the real header text (no external LM needed).
    """

    def __init__(self, n_num, cardinalities, col_name_vectors=None,
                 d_token=64, depth=3, heads=8, dropout=0.1):
        super().__init__()
        self.n_num = n_num
        self.cards = list(cardinalities)
        n_col = n_num + len(self.cards)
        if col_name_vectors is None:
            col_name_vectors = torch.randn(n_col, d_token) * 0.02
        self.col_name = nn.Parameter(torch.as_tensor(col_name_vectors,
                                                     dtype=torch.float32).clone())
        self.num_proj = nn.Linear(1, d_token)
        self.cat_emb = nn.ModuleList([nn.Embedding(c, d_token) for c in self.cards])
        self.gate = nn.Linear(d_token, d_token)
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_token, heads, dropout)
                                     for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(),
                                  nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        toks = []
        if self.n_num:
            v = self.num_proj(x_num.unsqueeze(-1))                 # B,n_num,d
            toks.append(v * torch.sigmoid(self.gate(self.col_name[:self.n_num]))
                        + self.col_name[:self.n_num])
        if self.cards:
            c = torch.stack([e(x_cat[:, i].clamp(0, self.cards[i] - 1))
                             for i, e in enumerate(self.cat_emb)], dim=1)
            toks.append(c + self.col_name[self.n_num:])
        t = torch.cat(toks, dim=1)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        for b in self.blocks:
            t = b(t)
        return self.head(t[:, 0]).squeeze(-1)


def hashed_column_vectors(col_names, d=64, seed=0):
    """Deterministic character-n-gram hash embedding for column headers —
    gives TransTab its schema awareness without downloading a language model."""
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(4096, d, generator=g) * 0.05
    out = []
    for name in col_names:
        s = f"^{name.lower()}$"
        idx = [hash(s[i:i + 3]) % 4096 for i in range(len(s) - 2)] or [hash(s) % 4096]
        out.append(table[idx].mean(0))
    return torch.stack(out)



# ==============================================================================
# TREE_HYBRID
# ==============================================================================
"""
Tree-inspired / hybrid family (registry entries 17–23).

17. NODE     — Neural Oblivious Decision Ensembles (dense-block stacked)
18. DeepGBM  — GBDT distillation (CatNN + GBDT2NN): trees are trained first,
               their leaf structure is distilled into an embedding network
19. GrowNet  — gradient boosting where each weak learner is a shallow MLP
20. GANDALF  — Gated Adaptive Network for Deep Automated Learning of Features
               (Gated Feature Learning Units with learnable feature masks)
21. DANet    — Deep Abstract Networks: Abstract Layers group features via a
               sparse learnable mask, then apply shared "abstract" transforms
22. Net-DNF  — disjunctive normal form networks: literals -> conjunctions ->
               disjunctions, with feature selection, mimicking tree ensembles
23. TabNN    — recursive encoder groups correlated features into "clusters",
               each handled by a sub-network whose structure is initialised
               from a GBDT leaf-index clustering
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# ----------------------------------------------------------------------
# 17. NODE
# ----------------------------------------------------------------------
class NODE(nn.Module):
    """DenseNet-style stacking of oblivious tree layers: each layer sees the
    raw input plus every previous layer's tree outputs."""

    def __init__(self, n_num, cardinalities, n_layers=3, n_trees=64, depth=5,
                 tree_out=3):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        d = n_num + self.emb.out_dim
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(ObliviousTreeLayer(d, n_trees, depth, tree_out))
            d += n_trees * tree_out
        self.head = nn.Linear(d, 1)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        for l in self.layers:
            h = l(x, reduce="none")                 # B, n_trees, out_dim
            x = torch.cat([x, h.reshape(x.shape[0], -1)], dim=1)
        return self.head(x).squeeze(-1)


# ----------------------------------------------------------------------
# 18. DeepGBM
# ----------------------------------------------------------------------
class DeepGBM(nn.Module):
    """
    Two towers:
      CatNN   — FM + deep tower over categorical/sparse features
      GBDT2NN — an MLP trained to reproduce the leaf-index one-hot outputs of a
                pre-fitted LightGBM/HistGB ensemble, distilling its structure.

    `leaf_targets` (B, n_trees) must be supplied at fit time by the wrapper in
    registry.py; the distillation loss is exposed via aux_loss().
    """

    def __init__(self, n_num, cardinalities, n_trees=100, max_leaf=32,
                 emb_dim=8, hidden=(256, 128), dropout=0.1, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.emb = CatEmbeddings(cardinalities, dim=emb_dim)
        # ---- CatNN: FM second order + deep
        n_cat = len(list(cardinalities))
        self.fm_first = nn.ModuleList([nn.Embedding(c, 1) for c in cardinalities])
        self.cards = list(cardinalities)
        d_in = n_num + self.emb.out_dim
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.cat_deep = nn.Sequential(*layers, nn.Linear(prev, 1))
        # ---- GBDT2NN
        self.tree_mlp = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU())
        self.leaf_head = nn.Linear(128, n_trees * max_leaf)   # distillation target
        self.tree_out = nn.Linear(128, 1)
        self.n_trees, self.max_leaf = n_trees, max_leaf
        self.w = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.leaf_targets = None

    def forward(self, x_num, x_cat):
        e = self.emb(x_cat)
        x = torch.cat([x_num, e], dim=1)
        fm1 = sum(t(x_cat[:, i].clamp(0, self.cards[i] - 1)).squeeze(-1)
                  for i, t in enumerate(self.fm_first)) if self.cards else 0
        ev = e.view(e.shape[0], len(self.cards), -1) if self.cards else None
        fm2 = 0
        if ev is not None:
            s = ev.sum(1)
            fm2 = 0.5 * ((s ** 2) - (ev ** 2).sum(1)).sum(1)
        cat_logit = self.cat_deep(x).squeeze(-1) + fm1 + fm2
        h = self.tree_mlp(x)
        self._leaf_logits = self.leaf_head(h)
        tree_logit = self.tree_out(h).squeeze(-1)
        w = torch.softmax(self.w, 0)
        return w[0] * cat_logit + w[1] * tree_logit

    def aux_loss(self):
        if self.leaf_targets is None:
            return torch.zeros((), device=self._leaf_logits.device)
        lg = self._leaf_logits.view(-1, self.n_trees, self.max_leaf)
        tgt = self.leaf_targets[:lg.shape[0]]
        return self.beta * F.cross_entropy(lg.reshape(-1, self.max_leaf),
                                           tgt.reshape(-1))


# ----------------------------------------------------------------------
# 19. GrowNet
# ----------------------------------------------------------------------
class WeakMLP(nn.Module):
    """Shallow net that receives [raw features, penultimate features of the
    previous learner] — the 'corrective step' input of GrowNet."""

    def __init__(self, d_in, d_prev, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(d_in + d_prev, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, prev_hidden=None):
        z = x if prev_hidden is None else torch.cat([x, prev_hidden], dim=1)
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.out(h).squeeze(-1), h


class GrowNet(nn.Module):
    """
    Gradient boosting with neural weak learners. Stage k fits the negative
    gradient of the logistic loss w.r.t. the current ensemble score.
    Trained by the dedicated GrowNetTrainer in registry.py.
    """

    def __init__(self, n_num, cardinalities, n_stages=8, hidden=64, lr_stage=0.3):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        self.d_in = n_num + self.emb.out_dim
        self.hidden = hidden
        self.lr_stage = lr_stage
        self.learners = nn.ModuleList(
            [WeakMLP(self.d_in, 0 if i == 0 else hidden, hidden)
             for i in range(n_stages)])
        self.n_active = len(self.learners)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        score, h = 0.0, None
        for i, m in enumerate(self.learners[:self.n_active]):
            o, h = m(x, h)
            score = score + self.lr_stage * o
        return score


# ----------------------------------------------------------------------
# 20. GANDALF
# ----------------------------------------------------------------------
def t_softmax(x, t=0.5, dim=-1):
    """Temperature-controlled sparse softmax used by GANDALF's feature masks."""
    mx = x.max(dim=dim, keepdim=True).values
    z = torch.exp((x - mx) / max(t, 1e-3))
    return z / (z.sum(dim=dim, keepdim=True) + 1e-9)


class GFLU(nn.Module):
    """Gated Feature Learning Unit — soft feature mask + GRU-style gating."""

    def __init__(self, d_in, d_state, t=0.5, dropout=0.05):
        super().__init__()
        self.mask = nn.Linear(d_state, d_in)
        self.t = t
        self.Wz = nn.Linear(d_in + d_state, d_state)
        self.Wr = nn.Linear(d_in + d_state, d_state)
        self.Wh = nn.Linear(d_in + d_state, d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, h):
        m = t_softmax(self.mask(h), self.t)
        xm = x * m * x.shape[1]
        c = torch.cat([xm, h], dim=1)
        z = torch.sigmoid(self.Wz(c))
        r = torch.sigmoid(self.Wr(c))
        hh = torch.tanh(self.Wh(torch.cat([xm, r * h], dim=1)))
        return self.drop((1 - z) * h + z * hh), m


class GANDALF(nn.Module):
    def __init__(self, n_num, cardinalities, n_stages=6, d_state=32,
                 t=0.5, dropout=0.05):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        d_in = n_num + self.emb.out_dim
        self.h0 = nn.Parameter(torch.zeros(1, d_state))
        self.stages = nn.ModuleList([GFLU(d_in, d_state, t, dropout)
                                     for _ in range(n_stages)])
        self.head = nn.Sequential(nn.LayerNorm(d_state), nn.ReLU(),
                                  nn.Linear(d_state, 1))

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        h = self.h0.expand(x.shape[0], -1)
        self.masks_ = []
        for s in self.stages:
            h, m = s(x, h)
            self.masks_.append(m.detach().mean(0))
        return self.head(h).squeeze(-1)


# ----------------------------------------------------------------------
# 21. DANet
# ----------------------------------------------------------------------
class AbstractLayer(nn.Module):
    """Sparse learnable feature grouping followed by k shared 'abstract'
    transformations that are mixed per group."""

    def __init__(self, d_in, d_out, k=5, virtual_bs=256):
        super().__init__()
        self.k, self.d_in, self.d_out = k, d_in, d_out
        self.in_norm = nn.LayerNorm(d_in)
        self.mask = nn.Linear(d_in, k * d_in, bias=False)
        nn.init.xavier_uniform_(self.mask.weight, gain=0.1)
        self.proc = nn.ModuleList([nn.Sequential(
            nn.Linear(d_in, 2 * d_out), nn.BatchNorm1d(2 * d_out))
            for _ in range(k)])
        self.scale = 1.0 / math.sqrt(d_in)

    def forward(self, x):
        B, D = x.shape
        h = self.in_norm(x)
        # scaled + clamped logits keep entmax's support non-degenerate; without
        # this the sparse mask can collapse and produce NaNs on deeper stacks
        logits = (self.mask(h).view(B, self.k, D) * self.scale).clamp(-15, 15)
        m = entmax15(logits, dim=-1)
        m = m * D                                   # keep the masked input scaled
        out = 0
        for i, p in enumerate(self.proc):
            z = p(h * m[:, i, :])
            a, b = z.chunk(2, dim=-1)
            out = out + a * torch.sigmoid(b)
        return out / self.k


class DANet(nn.Module):
    def __init__(self, n_num, cardinalities, n_layers=4, d_hidden=64, k=5,
                 base_dropout=0.1):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        d_in = n_num + self.emb.out_dim
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        prev = d_in
        for _ in range(n_layers):
            self.blocks.append(AbstractLayer(prev, d_hidden, k))
            self.shortcuts.append(nn.Linear(prev, d_hidden))
            prev = d_hidden
        self.drop = nn.Dropout(base_dropout)
        self.head = nn.Sequential(nn.BatchNorm1d(d_hidden), nn.ReLU(),
                                  nn.Linear(d_hidden, 1))

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        for b, s in zip(self.blocks, self.shortcuts):
            x = self.drop(b(x) + s(x))          # basic block + shortcut (DANet-“B”)
        return self.head(x).squeeze(-1)


# ----------------------------------------------------------------------
# 22. Net-DNF
# ----------------------------------------------------------------------
class NetDNF(nn.Module):
    """
    Disjunctive Normal Form network.
      literals      : tanh(w·x + b)                    (soft half-spaces)
      conjunctions  : tanh(and_mask · literals - |and_mask|_1 + 1.5)
      disjunction   : tanh(or_mask · conjunctions + ...)
    A learnable, temperature-annealed feature-selection mask reproduces the
    column-subsampling of a random forest.
    """

    def __init__(self, n_num, cardinalities, n_formulas=64, n_literals=8,
                 feature_dropout=0.3, l1=1e-5):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        d = n_num + self.emb.out_dim
        self.n_formulas, self.n_literals = n_formulas, n_literals
        self.lit = nn.Linear(d, n_formulas * n_literals)
        self.feature_mask = nn.Parameter(torch.randn(n_formulas, d) * 0.1)
        self.and_mask = nn.Parameter(torch.randn(n_formulas, n_literals) * 0.1)
        self.or_w = nn.Linear(n_formulas, 1)
        self.l1 = l1
        self.feature_dropout = feature_dropout

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        B, D = x.shape
        fm = torch.sigmoid(self.feature_mask)                       # F,D
        if self.training and self.feature_dropout > 0:
            fm = fm * (torch.rand_like(fm) > self.feature_dropout).float()
        self._fm = fm
        xs = torch.einsum("bd,fd->bfd", x, fm)                      # B,F,D
        w = self.lit.weight.view(self.n_formulas, self.n_literals, D)
        b = self.lit.bias.view(self.n_formulas, self.n_literals)
        lits = torch.tanh(torch.einsum("bfd,fld->bfl", xs, w) + b)  # B,F,L
        am = torch.tanh(self.and_mask)
        conj = torch.tanh((lits * am).sum(-1) - am.abs().sum(-1) + 1.5)   # B,F
        return self.or_w(conj).squeeze(-1)

    def aux_loss(self):
        return self.l1 * self._fm.abs().mean()


# ----------------------------------------------------------------------
# 23. TabNN
# ----------------------------------------------------------------------
class TabNN(nn.Module):
    """
    Feature-group network. Correlated columns are placed in the same group
    (grouping is passed in from a GBDT leaf-cooccurrence clustering computed in
    registry.py; falls back to correlation-based k-means). Each group gets its
    own sub-network; group outputs are concatenated and fused.
    """

    def __init__(self, n_num, cardinalities, groups=None, n_groups=6,
                 group_hidden=64, fuse_hidden=(128,), dropout=0.1):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        d = n_num + self.emb.out_dim
        if groups is None:
            groups = [list(range(i, d, n_groups)) for i in range(n_groups)]
        groups = [g for g in groups if len(g)]
        self.groups = groups
        self.register_buffer("_dummy", torch.zeros(1))
        self.subnets = nn.ModuleList([
            nn.Sequential(nn.Linear(len(g), group_hidden), nn.ReLU(),
                          nn.Dropout(dropout),
                          nn.Linear(group_hidden, group_hidden), nn.ReLU())
            for g in groups])
        prev = group_hidden * len(groups)
        layers = []
        for h in fuse_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.fuse = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        outs = [net(x[:, g]) for net, g in zip(self.subnets, self.groups)]
        return self.fuse(torch.cat(outs, dim=1)).squeeze(-1)


def correlation_groups(X, n_groups=6, seed=0):
    """Cluster columns by absolute correlation — the fallback grouping for TabNN."""
    from sklearn.cluster import AgglomerativeClustering
    C = np.corrcoef(np.asarray(X, dtype=float).T)
    C = np.nan_to_num(C)
    D = 1 - np.abs(C)
    lab = AgglomerativeClustering(n_clusters=min(n_groups, D.shape[0]),
                                  metric="precomputed",
                                  linkage="average").fit_predict(D)
    return [list(np.where(lab == g)[0]) for g in np.unique(lab)]



# ==============================================================================
# CTR_MODELS
# ==============================================================================
"""
Interaction / CTR-derived family (registry entries 24–28).

24. Wide & Deep — memorisation (linear on crosses) + generalisation (MLP)
25. DeepFM      — factorisation machine (order-2) sharing embeddings with an MLP
26. xDeepFM     — adds a Compressed Interaction Network for explicit vector-wise
                  high-order interactions
27. DCN         — Deep & Cross Network v1: x_{l+1} = x0 * (x_l^T w_l) + b_l + x_l
28. DCNv2       — cross layer uses a full (or low-rank) matrix instead of a vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def _mlp(d_in, hidden, dropout, d_out=None):
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    if d_out is not None:
        layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers), (d_out or prev)


# ----------------------------------------------------------------------
# 24. Wide & Deep
# ----------------------------------------------------------------------
class WideAndDeep(nn.Module):
    def __init__(self, n_num, cardinalities, d_emb=16, hidden=(256, 128),
                 dropout=0.1, n_cross_hash=2048):
        super().__init__()
        self.cards = list(cardinalities)
        self.emb = CatEmbeddings(cardinalities, dim=d_emb)
        # wide part: raw numerics + one-hot cats + hashed pairwise cat crosses
        self.n_cross_hash = n_cross_hash
        self.wide_num = nn.Linear(n_num, 1)
        self.wide_cat = nn.ModuleList([nn.Embedding(c, 1) for c in self.cards])
        self.wide_cross = nn.Embedding(n_cross_hash, 1)
        nn.init.zeros_(self.wide_cross.weight)
        self.deep, d = _mlp(n_num + self.emb.out_dim, hidden, dropout, 1)

    def forward(self, x_num, x_cat):
        wide = self.wide_num(x_num).squeeze(-1)
        for i, e in enumerate(self.wide_cat):
            wide = wide + e(x_cat[:, i].clamp(0, self.cards[i] - 1)).squeeze(-1)
        n = len(self.cards)
        for i in range(n):
            for j in range(i + 1, n):
                h = (x_cat[:, i] * 131 + x_cat[:, j] * 977 + i * 31 + j) % self.n_cross_hash
                wide = wide + self.wide_cross(h).squeeze(-1)
        deep = self.deep(torch.cat([x_num, self.emb(x_cat)], dim=1)).squeeze(-1)
        return wide + deep


# ----------------------------------------------------------------------
# 25. DeepFM
# ----------------------------------------------------------------------
class DeepFM(nn.Module):
    def __init__(self, n_num, cardinalities, d_emb=16, hidden=(256, 128),
                 dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_emb)
        self.n_fields = self.tok.n_tokens - 1
        self.first_order = nn.Linear(self.n_fields, 1)
        self.first_proj = nn.Linear(d_emb, 1)
        self.deep, _ = _mlp(self.n_fields * d_emb, hidden, dropout, 1)

    def forward(self, x_num, x_cat):
        e = self.tok(x_num, x_cat, with_cls=False)          # B,F,d
        first = self.first_proj(e).squeeze(-1)              # B,F
        y1 = self.first_order(first).squeeze(-1)
        s = e.sum(1)
        y2 = 0.5 * ((s ** 2) - (e ** 2).sum(1)).sum(1)      # FM 2nd order
        y3 = self.deep(e.flatten(1)).squeeze(-1)
        return y1 + y2 + y3


# ----------------------------------------------------------------------
# 26. xDeepFM
# ----------------------------------------------------------------------
class CIN(nn.Module):
    """Compressed Interaction Network — explicit vector-wise interactions
    of increasing order, one layer per order."""

    def __init__(self, n_fields, layer_sizes=(64, 64)):
        super().__init__()
        self.convs = nn.ModuleList()
        prev = n_fields
        self.out_dim = 0
        for h in layer_sizes:
            self.convs.append(nn.Conv1d(n_fields * prev, h, 1))
            prev = h
            self.out_dim += h

    def forward(self, x0):                                   # B,F,d
        B, F0, D = x0.shape
        xk = x0
        outs = []
        for conv in self.convs:
            z = torch.einsum("bhd,bmd->bhmd", xk, x0).reshape(B, -1, D)
            xk = F.relu(conv(z))
            outs.append(xk.sum(-1))
        return torch.cat(outs, dim=1)


class xDeepFM(nn.Module):
    def __init__(self, n_num, cardinalities, d_emb=16, cin_layers=(48, 48),
                 hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_emb)
        F_ = self.tok.n_tokens - 1
        self.cin = CIN(F_, cin_layers)
        self.cin_out = nn.Linear(self.cin.out_dim, 1)
        self.lin = nn.Linear(F_ * d_emb, 1)
        self.deep, _ = _mlp(F_ * d_emb, hidden, dropout, 1)

    def forward(self, x_num, x_cat):
        e = self.tok(x_num, x_cat, with_cls=False)
        f = e.flatten(1)
        return (self.lin(f) + self.cin_out(self.cin(e)) + self.deep(f)).squeeze(-1)


# ----------------------------------------------------------------------
# 27. DCN (v1)
# ----------------------------------------------------------------------
class CrossNetV1(nn.Module):
    def __init__(self, d, n_layers=3):
        super().__init__()
        self.w = nn.ParameterList([nn.Parameter(torch.randn(d) * 0.01)
                                   for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(d))
                                   for _ in range(n_layers)])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.w, self.b):
            x = x0 * (x @ w).unsqueeze(1) + b + x
        return x


class DCN(nn.Module):
    def __init__(self, n_num, cardinalities, d_emb=16, n_cross=3,
                 hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=d_emb)
        d = n_num + self.emb.out_dim
        self.cross = CrossNetV1(d, n_cross)
        self.deep, dd = _mlp(d, hidden, dropout)
        self.head = nn.Linear(d + dd, 1)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        return self.head(torch.cat([self.cross(x), self.deep(x)], dim=1)).squeeze(-1)


# ----------------------------------------------------------------------
# 28. DCNv2
# ----------------------------------------------------------------------
class CrossNetV2(nn.Module):
    """x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l, with optional low-rank W = U V^T
    and a mixture of experts over subspaces."""

    def __init__(self, d, n_layers=3, rank=None, n_experts=1):
        super().__init__()
        self.low_rank = rank is not None
        self.n_experts = n_experts
        if self.low_rank:
            self.U = nn.ParameterList([nn.Parameter(torch.randn(n_experts, d, rank) * .02)
                                       for _ in range(n_layers)])
            self.V = nn.ParameterList([nn.Parameter(torch.randn(n_experts, rank, d) * .02)
                                       for _ in range(n_layers)])
            self.gate = nn.ModuleList([nn.Linear(d, n_experts) for _ in range(n_layers)])
        else:
            self.W = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(n_layers)])

    def forward(self, x0):
        x = x0
        for i in range(len(self.b)):
            if self.low_rank:
                g = torch.softmax(self.gate[i](x), dim=-1)        # B,E
                outs = []
                for e in range(self.n_experts):
                    h = torch.tanh(x @ self.U[i][e]) @ self.V[i][e]
                    outs.append(h)
                h = torch.stack(outs, 1)                          # B,E,d
                h = (h * g.unsqueeze(-1)).sum(1)
            else:
                h = self.W[i](x)
            x = x0 * (h + self.b[i]) + x
        return x


class DCNv2(nn.Module):
    def __init__(self, n_num, cardinalities, d_emb=16, n_cross=3, rank=32,
                 n_experts=2, hidden=(256, 128), dropout=0.1, structure="parallel"):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=d_emb)
        d = n_num + self.emb.out_dim
        self.structure = structure
        self.cross = CrossNetV2(d, n_cross, rank, n_experts)
        if structure == "parallel":
            self.deep, dd = _mlp(d, hidden, dropout)
            self.head = nn.Linear(d + dd, 1)
        else:                                                     # stacked
            self.deep, dd = _mlp(d, hidden, dropout)
            self.head = nn.Linear(dd, 1)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        c = self.cross(x)
        if self.structure == "parallel":
            return self.head(torch.cat([c, self.deep(x)], dim=1)).squeeze(-1)
        return self.head(self.deep(c)).squeeze(-1)



# ==============================================================================
# RETRIEVAL_SSL
# ==============================================================================
"""
Retrieval / in-context / pretrained  (29–35),
Sequence-backbone adapted             (36–37),
Self-supervised / representation      (38–41).

29/30. TabPFN, TabPFN v2 — prior-fitted transformers used through the official
       package when installed (they are pretrained; we never train them here).
31.    TabICL             — in-context tabular learner (official package).
32.    TabR               — retrieval-augmented MLP: a differentiable k-NN module
                            over a learned embedding space, implemented natively.
33.    CARTE              — graph-attentive pretrained model (official package).
34.    XTab               — cross-table federated pretraining (checkpoint-based).
35.    TabDPT             — decision-pretrained transformer (official package).
36/37. MambaTab / TabMamba — selective state-space backbone over the feature
                            sequence; native minimal S6 implementation so it runs
                            without CUDA kernels.
38.    VIME               — value imputation + mask estimation pretext.
39.    SubTab             — subsetting features as multi-view for contrastive +
                            reconstruction pretraining.
40.    SCARF              — corruption-based contrastive (InfoNCE) pretraining.
41.    DAE                — denoising autoencoder with swap noise, then finetune.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# ======================================================================
# 32. TabR — retrieval-augmented tabular network
# ======================================================================
class TabR(nn.Module):
    """
    y(x) = head( E(x) + sum_i  a(x, x_i) * ( V(x_i) + T(y_i) ) )

    where the sum runs over the k nearest neighbours of x inside a *context
    set* (the training rows). Attention weights come from L2 distance in a
    learned embedding space. Labels of retrieved neighbours are fed in
    explicitly, which is what separates TabR from a plain MLP.
    """

    def __init__(self, n_num, cardinalities, d=192, k=32, dropout=0.1,
                 context_size=2048):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        d_in = n_num + self.emb.out_dim
        self.k, self.d = k, d
        self.encoder = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d, d))
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d)
        self.label_emb = nn.Linear(1, d)
        self.T = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, d),
                               nn.Dropout(dropout), nn.Linear(d, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
        self.context_size = context_size
        self.register_buffer("ctx_x", torch.zeros(0, d_in))
        self.register_buffer("ctx_y", torch.zeros(0))

    def set_context(self, Xn, Xc, y, max_n=None):
        """Called once by the wrapper after fitting the preprocessor."""
        max_n = max_n or self.context_size
        n = len(y)
        idx = np.random.RandomState(0).choice(n, min(max_n, n), replace=False)
        with torch.no_grad():
            xn = torch.as_tensor(Xn[idx], dtype=torch.float32, device=DEVICE)
            xc = torch.as_tensor(Xc[idx], dtype=torch.long, device=DEVICE)
            x = torch.cat([xn, self.emb(xc)], dim=1)
        self.ctx_x = x
        self.ctx_y = torch.as_tensor(y[idx], dtype=torch.float32, device=DEVICE)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        q = self.encoder(x)                                       # B,d
        if self.ctx_x.numel() == 0:                               # no context yet
            return self.head(q).squeeze(-1)
        ctx = self.encoder(self.ctx_x)                            # N,d
        kq, kc = self.W_k(q), self.W_k(ctx)
        # negative squared L2 as the similarity (TabR's key design choice)
        sim = -(kq.pow(2).sum(1, keepdim=True) - 2 * kq @ kc.t()
                + kc.pow(2).sum(1))
        k = min(self.k, ctx.shape[0])
        top, idx = sim.topk(k, dim=1)
        a = torch.softmax(top / math.sqrt(self.d), dim=1)         # B,k
        v = self.W_v(ctx[idx])                                    # B,k,d
        lab = self.label_emb(self.ctx_y[idx].unsqueeze(-1))       # B,k,d
        ctx_val = self.T(v + lab)
        return self.head(q + (a.unsqueeze(-1) * ctx_val).sum(1)).squeeze(-1)


# ======================================================================
# 36/37. MambaTab / TabMamba
# ======================================================================
class S6Block(nn.Module):
    """
    Minimal selective state-space block (the S6 core of Mamba), written with
    an explicit recurrence so it runs on CPU without the fused CUDA kernel.
    Feature count on tabular data is small (tens), so the loop is cheap.
    """

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1,
                              groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, d_inner)
        A = torch.arange(1, d_state + 1).float().repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):                                   # B,L,d_model
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi = self.conv(xi.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xi = F.silu(xi)
        dbc = self.x_proj(xi)
        dt, Bm, Cm = dbc[..., :1], dbc[..., 1:1 + self.d_state], dbc[..., 1 + self.d_state:]
        dt = F.softplus(self.dt_proj(dt))                   # B,L,d_inner
        A = -torch.exp(self.A_log)                          # d_inner,d_state
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)                  # B,di,ds
            dB = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)         # B,di,ds
            h = dA * h + dB * xi[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + self.D * xi
        return self.out_proj(y * F.silu(z))


class MambaTab(nn.Module):
    """One learned token per feature, a single S6 block over the feature
    sequence, mean-pooled readout. Matches the 'plain and efficient' design
    of the MambaTab paper."""

    def __init__(self, n_num, cardinalities, d_model=32, d_state=16, n_blocks=1,
                 dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([S6Block(d_model, d_state) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x_num, x_cat):
        t = self.norm_in(self.tok(x_num, x_cat, with_cls=False))
        for b, n in zip(self.blocks, self.norms):
            t = t + b(n(t))
        return self.head(t.mean(1)).squeeze(-1)


class TabMamba(nn.Module):
    """Bidirectional, deeper variant: scans the feature sequence forwards and
    backwards and mixes with a small transformer-style FFN, plus a [CLS] token."""

    def __init__(self, n_num, cardinalities, d_model=48, d_state=16, n_blocks=2,
                 dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cardinalities, d_model)
        self.fwd = nn.ModuleList([S6Block(d_model, d_state) for _ in range(n_blocks)])
        self.bwd = nn.ModuleList([S6Block(d_model, d_state) for _ in range(n_blocks)])
        self.ff = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
            for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.ReLU(),
                                  nn.Linear(d_model, 1))

    def forward(self, x_num, x_cat):
        t = self.tok(x_num, x_cat, with_cls=True)
        for f, b, ff, n in zip(self.fwd, self.bwd, self.ff, self.norms):
            h = n(t)
            t = t + f(h) + torch.flip(b(torch.flip(h, [1])), [1])
            t = t + ff(t)
        return self.head(t[:, 0]).squeeze(-1)


# ======================================================================
# Self-supervised family 38–41
# ======================================================================
class TabEncoder(nn.Module):
    """Shared encoder used by all four SSL methods."""

    def __init__(self, n_num, cardinalities, d_hidden=256, d_rep=128, dropout=0.1):
        super().__init__()
        self.emb = CatEmbeddings(cardinalities, dim=4)
        self.d_in = n_num + self.emb.out_dim
        self.net = nn.Sequential(
            nn.Linear(self.d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_rep), nn.BatchNorm1d(d_rep), nn.ReLU())
        self.d_rep = d_rep

    def raw(self, x_num, x_cat):
        return torch.cat([x_num, self.emb(x_cat)], dim=1)

    def forward(self, x_num, x_cat):
        return self.net(self.raw(x_num, x_cat))


class SSLBase(nn.Module):
    """Common interface: pretrain_loss(x) for stage 1, forward() for stage 2."""

    def __init__(self, n_num, cardinalities, d_hidden=256, d_rep=128, dropout=0.1):
        super().__init__()
        self.enc = TabEncoder(n_num, cardinalities, d_hidden, d_rep, dropout)
        self.head = nn.Sequential(nn.Linear(d_rep, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x_num, x_cat):
        return self.head(self.enc(x_num, x_cat)).squeeze(-1)


def swap_noise(x, p=0.3):
    idx = torch.randperm(x.size(0), device=x.device)
    m = (torch.rand_like(x) < p)
    return torch.where(m, x[idx], x), m.float()


# ---- 38. VIME --------------------------------------------------------
class VIME(SSLBase):
    """Pretext = predict the corruption mask (mask estimation) AND recover the
    original values (value imputation)."""

    def __init__(self, n_num, cardinalities, p_corrupt=0.3, alpha=2.0, **kw):
        super().__init__(n_num, cardinalities, **kw)
        d = self.enc.d_in
        self.mask_head = nn.Linear(self.enc.d_rep, d)
        self.value_head = nn.Linear(self.enc.d_rep, d)
        self.p, self.alpha = p_corrupt, alpha

    def pretrain_loss(self, x_num, x_cat):
        x = self.enc.raw(x_num, x_cat)
        xt, m = swap_noise(x, self.p)
        z = self.enc.net(xt)
        lm = F.binary_cross_entropy_with_logits(self.mask_head(z), m)
        lr = F.mse_loss(self.value_head(z), x)
        return lm + self.alpha * lr


# ---- 39. SubTab ------------------------------------------------------
class SubTab(SSLBase):
    """Splits the feature vector into overlapping subsets ('views'), encodes
    each, and combines contrastive + reconstruction + distance losses."""

    def __init__(self, n_num, cardinalities, n_subsets=4, overlap=0.25,
                 temperature=0.1, **kw):
        super().__init__(n_num, cardinalities, **kw)
        d = self.enc.d_in
        self.n_subsets = n_subsets
        size = int(math.ceil(d / n_subsets * (1 + overlap)))
        step = max(1, (d - size) // max(1, n_subsets - 1)) if n_subsets > 1 else d
        self.subsets = [list(range(min(i * step, d - size), min(i * step, d - size) + size))
                        for i in range(n_subsets)]
        self.sub_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(size, 256), nn.ReLU(), nn.Linear(256, self.enc.d_rep))
            for _ in range(n_subsets)])
        self.dec = nn.Linear(self.enc.d_rep, d)
        self.t = temperature

    def pretrain_loss(self, x_num, x_cat):
        x = self.enc.raw(x_num, x_cat)
        zs = [e(x[:, s]) for e, s in zip(self.sub_enc, self.subsets)]
        rec = sum(F.mse_loss(self.dec(z), x) for z in zs) / len(zs)
        zn = [F.normalize(z, dim=1) for z in zs]
        cont, n = 0.0, 0
        for i in range(len(zn)):
            for j in range(i + 1, len(zn)):
                logits = zn[i] @ zn[j].t() / self.t
                tgt = torch.arange(logits.size(0), device=x.device)
                cont = cont + .5 * (F.cross_entropy(logits, tgt)
                                    + F.cross_entropy(logits.t(), tgt))
                n += 1
        dist = sum(F.mse_loss(zn[i], zn[j]) for i in range(len(zn))
                   for j in range(i + 1, len(zn))) / max(1, n)
        return rec + cont / max(1, n) + dist


# ---- 40. SCARF -------------------------------------------------------
class SCARF(SSLBase):
    """Contrastive: corrupt a random subset of features by resampling each from
    its own empirical marginal, then pull the two views together (InfoNCE)."""

    def __init__(self, n_num, cardinalities, p_corrupt=0.6, temperature=0.5, **kw):
        super().__init__(n_num, cardinalities, **kw)
        self.proj = nn.Sequential(nn.Linear(self.enc.d_rep, 128), nn.ReLU(),
                                  nn.Linear(128, 64))
        self.p, self.t = p_corrupt, temperature

    def _corrupt(self, x):
        idx = torch.randint(0, x.size(0), (x.size(0), x.size(1)), device=x.device)
        marg = torch.gather(x, 0, idx)                   # per-column resample
        m = torch.rand_like(x) < self.p
        return torch.where(m, marg, x)

    def pretrain_loss(self, x_num, x_cat):
        x = self.enc.raw(x_num, x_cat)
        z1 = F.normalize(self.proj(self.enc.net(x)), dim=1)
        z2 = F.normalize(self.proj(self.enc.net(self._corrupt(x))), dim=1)
        logits = z1 @ z2.t() / self.t
        tgt = torch.arange(logits.size(0), device=x.device)
        return .5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


# ---- 41. Denoising Autoencoder --------------------------------------
class DenoisingAE(SSLBase):
    """Bottleneck autoencoder trained with swap noise; the encoder is then
    finetuned for classification (the classic Porto-Seguro recipe)."""

    def __init__(self, n_num, cardinalities, p_corrupt=0.2, **kw):
        super().__init__(n_num, cardinalities, **kw)
        d = self.enc.d_in
        self.dec = nn.Sequential(nn.Linear(self.enc.d_rep, 256), nn.ReLU(),
                                 nn.Linear(256, d))
        self.p = p_corrupt

    def pretrain_loss(self, x_num, x_cat):
        x = self.enc.raw(x_num, x_cat)
        xt, _ = swap_noise(x, self.p)
        return F.mse_loss(self.dec(self.enc.net(xt)), x)



# ==============================================================================
# GENERATIVE
# ==============================================================================
"""
Generative family (registry entries 42–47).

These are NOT classifiers. They are evaluated the way generative models for
tabular data actually get used: synthesise extra rows, add them to the training
fold, retrain a fixed downstream classifier, and report the change in test
performance (the standard "machine-learning efficacy" / TSTR protocol).

42. CTGAN      — conditional GAN with mode-specific normalisation + training-by-sampling
43. CTAB-GAN+  — CTGAN plus a classifier/auxiliary loss and information loss
44. TVAE       — variational autoencoder with the same CTGAN data transform
45. TabDDPM    — Gaussian diffusion over numeric cols + multinomial over categoricals
46. TabSyn     — VAE into a continuous latent space, then diffusion in that latent
47. GReaT      — LLM-based row synthesis ("Age is 57, Sex is Male, ...")

CTGAN / TVAE use the `ctgan` package when present; CTAB-GAN+, TabDDPM and TabSyn
are implemented natively here so the pipeline does not depend on
research repos that are awkward to install. GReaT requires a GPU and the
`be-great` package; it degrades to a skip if unavailable.
"""

import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


warnings.filterwarnings("ignore")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# helper: mixed-type encoder shared by the native generators
# ----------------------------------------------------------------------
class MixedTransform:
    """Numeric -> standardised; categorical -> one-hot. Invertible."""

    def __init__(self, num, cat):
        self.num, self.cat = num, cat

    def fit(self, df):
        self.mu = df[self.num].mean().values
        self.sd = df[self.num].std().replace(0, 1).values
        self.levels = {c: sorted(df[c].astype(str).unique()) for c in self.cat}
        self.dims = [len(self.levels[c]) for c in self.cat]
        return self

    def transform(self, df):
        xn = ((df[self.num].values - self.mu) / self.sd).astype(np.float32)
        oh = []
        for c in self.cat:
            lv = self.levels[c]
            m = np.zeros((len(df), len(lv)), np.float32)
            idx = df[c].astype(str).map({l: i for i, l in enumerate(lv)}).fillna(0).astype(int)
            m[np.arange(len(df)), idx.values] = 1
            oh.append(m)
        return np.hstack([xn] + oh) if oh else xn

    def inverse(self, Z):
        n = len(self.num)
        xn = Z[:, :n] * self.sd + self.mu
        out = pd.DataFrame(xn, columns=self.num)
        p = n
        for c, d in zip(self.cat, self.dims):
            blk = Z[:, p:p + d]
            out[c] = [self.levels[c][i] for i in blk.argmax(1)]
            p += d
        return out

    @property
    def dim(self):
        return len(self.num) + int(sum(self.dims))


def _activate(z, n_num, dims):
    """tanh on numerics, gumbel-softmax on each categorical block."""
    parts = [torch.tanh(z[:, :n_num])]
    p = n_num
    for d in dims:
        parts.append(F.gumbel_softmax(z[:, p:p + d], tau=0.2, hard=False))
        p += d
    return torch.cat(parts, dim=1)


# ======================================================================
# 42/44. CTGAN & TVAE  (official package when available)
# ======================================================================
def synth_ctgan(df, n_rows, discrete_cols, epochs=200, kind="ctgan"):
    from ctgan import CTGAN, TVAE
    m = (CTGAN(epochs=epochs, verbose=False, cuda=torch.cuda.is_available())
         if kind == "ctgan" else
         TVAE(epochs=epochs, cuda=torch.cuda.is_available()))
    m.fit(df, discrete_columns=discrete_cols)
    return m.sample(n_rows)


# ======================================================================
# 43. CTAB-GAN+  (native)
# ======================================================================
class _Gen(nn.Module):
    def __init__(self, z_dim, out_dim, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], z_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev, out_dim)

    def forward(self, z):
        return self.out(self.net(z))


class _Disc(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256), n_class=2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(.2), nn.Dropout(.3)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.adv = nn.Linear(prev, 1)
        self.cls = nn.Linear(prev, n_class)          # CTAB-GAN+ auxiliary classifier

    def forward(self, x):
        h = self.body(x)
        return self.adv(h), self.cls(h), h


class CTABGANPlus:
    """CTGAN backbone + auxiliary classifier loss + information (feature-matching)
    loss + WGAN-GP objective — the three additions of CTAB-GAN+."""

    def __init__(self, num, cat, target, z_dim=128, epochs=300, batch=500,
                 lambda_cls=1.0, lambda_info=1.0, gp=10.0, lr=2e-4):
        self.num, self.cat, self.target = num, cat, target
        self.z_dim, self.epochs, self.batch = z_dim, epochs, batch
        self.lambda_cls, self.lambda_info, self.gp, self.lr = \
            lambda_cls, lambda_info, gp, lr

    def fit(self, df):
        self.tf = MixedTransform(self.num, self.cat).fit(df)
        X = torch.tensor(self.tf.transform(df), device=DEV)
        y = torch.tensor(df[self.target].values, dtype=torch.long, device=DEV)
        d = X.shape[1]
        self.G = _Gen(self.z_dim + 2, d).to(DEV)
        self.D = _Disc(d + 2).to(DEV)
        og = torch.optim.Adam(self.G.parameters(), self.lr, betas=(.5, .9))
        od = torch.optim.Adam(self.D.parameters(), self.lr, betas=(.5, .9))
        n = len(X)
        n_num = len(self.num)
        for ep in range(self.epochs):
            idx = torch.randint(0, n, (self.batch,), device=DEV)
            xr, yr = X[idx], y[idx]
            yoh = F.one_hot(yr, 2).float()
            # --- D step
            z = torch.randn(self.batch, self.z_dim, device=DEV)
            xf = _activate(self.G(torch.cat([z, yoh], 1)), n_num, self.tf.dims)
            dr, cr, hr = self.D(torch.cat([xr, yoh], 1))
            dfk, cf, hf = self.D(torch.cat([xf.detach(), yoh], 1))
            eps = torch.rand(self.batch, 1, device=DEV)
            xh = (eps * xr + (1 - eps) * xf.detach()).requires_grad_(True)
            dh, _, _ = self.D(torch.cat([xh, yoh], 1))
            g = torch.autograd.grad(dh.sum(), xh, create_graph=True)[0]
            gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
            loss_d = (dfk.mean() - dr.mean() + self.gp * gp
                      + self.lambda_cls * F.cross_entropy(cr, yr))
            od.zero_grad(); loss_d.backward(); od.step()
            # --- G step
            z = torch.randn(self.batch, self.z_dim, device=DEV)
            xf = _activate(self.G(torch.cat([z, yoh], 1)), n_num, self.tf.dims)
            dfk, cf, hf = self.D(torch.cat([xf, yoh], 1))
            info = (F.mse_loss(hf.mean(0), hr.mean(0).detach())
                    + F.mse_loss(hf.std(0), hr.std(0).detach()))
            loss_g = (-dfk.mean() + self.lambda_cls * F.cross_entropy(cf, yr)
                      + self.lambda_info * info)
            og.zero_grad(); loss_g.backward(); og.step()
        return self

    @torch.no_grad()
    def sample(self, n, class_ratio=None):
        self.G.eval()
        y = torch.bernoulli(torch.full((n,), class_ratio if class_ratio
                                       else .5, device=DEV)).long()
        z = torch.randn(n, self.z_dim, device=DEV)
        x = _activate(self.G(torch.cat([z, F.one_hot(y, 2).float()], 1)),
                      len(self.num), self.tf.dims)
        out = self.tf.inverse(x.cpu().numpy())
        out[self.target] = y.cpu().numpy()
        return out


# ======================================================================
# 45. TabDDPM  (native)
# ======================================================================
class TabDDPM:
    """
    Gaussian diffusion on the (standardised) numeric block and independent
    multinomial diffusion on each categorical block, conditioned on the class
    label — the exact decomposition proposed in the TabDDPM paper.
    """

    def __init__(self, num, cat, target, T=200, d_hidden=256, epochs=400,
                 batch=512, lr=1e-3):
        self.num, self.cat, self.target = num, cat, target
        self.T, self.epochs, self.batch, self.lr = T, epochs, batch, lr
        self.d_hidden = d_hidden

    def _schedule(self):
        s = 0.008
        t = torch.linspace(0, self.T, self.T + 1, device=DEV) / self.T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        betas = torch.clip(1 - ab[1:] / ab[:-1], 0, .999)
        self.betas = betas
        self.alphas = 1 - betas
        self.abar = torch.cumprod(self.alphas, 0)

    def fit(self, df):
        self.tf = MixedTransform(self.num, self.cat).fit(df)
        X = torch.tensor(self.tf.transform(df), device=DEV)
        y = torch.tensor(df[self.target].values, dtype=torch.long, device=DEV)
        d = X.shape[1]
        self._schedule()
        self.net = nn.Sequential(
            nn.Linear(d + 2 + 32, self.d_hidden), nn.SiLU(),
            nn.Linear(self.d_hidden, self.d_hidden), nn.SiLU(),
            nn.Linear(self.d_hidden, self.d_hidden), nn.SiLU(),
            nn.Linear(self.d_hidden, d)).to(DEV)
        opt = torch.optim.AdamW(self.net.parameters(), self.lr)
        n = len(X)
        for ep in range(self.epochs):
            i = torch.randint(0, n, (min(self.batch, n),), device=DEV)
            x0, yy = X[i], y[i]
            t = torch.randint(0, self.T, (len(i),), device=DEV)
            ab = self.abar[t].unsqueeze(1)
            eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            pred = self.net(torch.cat([xt, F.one_hot(yy, 2).float(),
                                       self._temb(t)], 1))
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def _temb(self, t, dim=32):
        half = dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=DEV) / half)
        a = t.float().unsqueeze(1) * f.unsqueeze(0)
        return torch.cat([a.sin(), a.cos()], dim=1)

    @torch.no_grad()
    def sample(self, n, class_ratio=0.5):
        self.net.eval()
        y = torch.bernoulli(torch.full((n,), class_ratio, device=DEV)).long()
        yoh = F.one_hot(y, 2).float()
        x = torch.randn(n, self.tf.dim, device=DEV)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=DEV, dtype=torch.long)
            eps = self.net(torch.cat([x, yoh, self._temb(tt)], 1))
            a, ab, b = self.alphas[t], self.abar[t], self.betas[t]
            x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0:
                x = x + b.sqrt() * torch.randn_like(x)
        out = self.tf.inverse(x.cpu().numpy())
        out[self.target] = y.cpu().numpy()
        return out


# ======================================================================
# 46. TabSyn  (native): VAE -> latent diffusion
# ======================================================================
class TabSyn:
    def __init__(self, num, cat, target, d_latent=16, epochs_vae=300,
                 epochs_diff=400, batch=512, lr=1e-3, beta=1e-2):
        self.num, self.cat, self.target = num, cat, target
        self.d_latent, self.beta = d_latent, beta
        self.epochs_vae, self.epochs_diff, self.batch, self.lr = \
            epochs_vae, epochs_diff, batch, lr

    def fit(self, df):
        self.tf = MixedTransform(self.num, self.cat).fit(df)
        X = torch.tensor(self.tf.transform(df), device=DEV)
        y = torch.tensor(df[self.target].values, dtype=torch.long, device=DEV)
        d, L = X.shape[1], self.d_latent
        self.enc = nn.Sequential(nn.Linear(d + 2, 256), nn.SiLU(),
                                 nn.Linear(256, 2 * L)).to(DEV)
        self.dec = nn.Sequential(nn.Linear(L + 2, 256), nn.SiLU(),
                                 nn.Linear(256, d)).to(DEV)
        opt = torch.optim.AdamW(list(self.enc.parameters()) +
                                list(self.dec.parameters()), self.lr)
        n = len(X)
        for _ in range(self.epochs_vae):                     # --- stage 1: VAE
            i = torch.randint(0, n, (min(self.batch, n),), device=DEV)
            x, yy = X[i], F.one_hot(y[i], 2).float()
            h = self.enc(torch.cat([x, yy], 1))
            mu, lv = h.chunk(2, -1)
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            rec = F.mse_loss(_activate(self.dec(torch.cat([z, yy], 1)),
                                       len(self.num), self.tf.dims), x)
            kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).mean()
            loss = rec + self.beta * kl
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():                                # --- latent bank
            mu, _ = self.enc(torch.cat([X, F.one_hot(y, 2).float()], 1)).chunk(2, -1)
        self.z_mu, self.z_sd = mu.mean(0), mu.std(0).clamp(min=1e-3)
        Z = (mu - self.z_mu) / self.z_sd

        self.diff = TabDDPM(self.num, self.cat, self.target, T=100,
                            epochs=self.epochs_diff)          # --- stage 2
        self.diff._schedule()
        self.dnet = nn.Sequential(
            nn.Linear(L + 2 + 32, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, L)).to(DEV)
        od = torch.optim.AdamW(self.dnet.parameters(), self.lr)
        for _ in range(self.epochs_diff):
            i = torch.randint(0, n, (min(self.batch, n),), device=DEV)
            z0, yy = Z[i], F.one_hot(y[i], 2).float()
            t = torch.randint(0, self.diff.T, (len(i),), device=DEV)
            ab = self.diff.abar[t].unsqueeze(1)
            eps = torch.randn_like(z0)
            zt = ab.sqrt() * z0 + (1 - ab).sqrt() * eps
            loss = F.mse_loss(self.dnet(torch.cat([zt, yy, self.diff._temb(t)], 1)), eps)
            od.zero_grad(); loss.backward(); od.step()
        return self

    @torch.no_grad()
    def sample(self, n, class_ratio=0.5):
        y = torch.bernoulli(torch.full((n,), class_ratio, device=DEV)).long()
        yoh = F.one_hot(y, 2).float()
        z = torch.randn(n, self.d_latent, device=DEV)
        for t in reversed(range(self.diff.T)):
            tt = torch.full((n,), t, device=DEV, dtype=torch.long)
            eps = self.dnet(torch.cat([z, yoh, self.diff._temb(tt)], 1))
            a, ab, b = self.diff.alphas[t], self.diff.abar[t], self.diff.betas[t]
            z = (z - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0:
                z = z + b.sqrt() * torch.randn_like(z)
        z = z * self.z_sd + self.z_mu
        x = _activate(self.dec(torch.cat([z, yoh], 1)), len(self.num), self.tf.dims)
        out = self.tf.inverse(x.cpu().numpy())
        out[self.target] = y.cpu().numpy()
        return out


# ======================================================================
# 47. GReaT
# ======================================================================
def synth_great(df, n_rows, epochs=50, model="distilgpt2"):
    """LLM row synthesis. Needs `be-great` + a GPU to be practical."""
    from be_great import GReaT
    g = GReaT(llm=model, batch_size=32, epochs=epochs)
    g.fit(df)
    return g.sample(n_samples=n_rows)



# ==============================================================================
# METRICS
# ==============================================================================
"""
Every classification metric worth reporting for a clinical risk model,
in one place.

Threshold-free : ROC AUC, PR AUC (average precision), Brier, log loss,
                 Gini, KS statistic, H-measure-style scaled Brier
Threshold-based: accuracy, balanced accuracy, precision, recall/sensitivity,
                 specificity, NPV, F1, F2, MCC, Cohen's kappa, Youden's J,
                 likelihood ratios (LR+ / LR−), diagnostic odds ratio
Calibration    : Brier decomposition (reliability/resolution/uncertainty),
                 ECE, MCE, calibration slope + intercept
Clinical       : net benefit at chosen thresholds (decision-curve analysis),
                 number needed to screen
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             fbeta_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


# ----------------------------------------------------------------------
def best_threshold(y, p, criterion="youden"):
    fpr, tpr, thr = roc_curve(y, p)
    if criterion == "youden":
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    if criterion == "f1":
        cands = np.unique(np.quantile(p, np.linspace(.01, .99, 99)))
        return float(max(cands, key=lambda t: f1_score(y, (p >= t).astype(int))))
    return 0.5


def ks_statistic(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = mce = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(y[m].mean() - p[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def calibration_slope_intercept(y, p):
    """Cox calibration: logit(y) ~ a + b*logit(p). b=1, a=0 is perfect."""
    import statsmodels.api as sm
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    X = sm.add_constant(lp)
    try:
        r = sm.Logit(y, X).fit(disp=0)
        return float(r.params[1]), float(r.params[0])
    except Exception:
        return np.nan, np.nan


def brier_decomposition(y, p, n_bins=10):
    """Murphy decomposition: Brier = reliability − resolution + uncertainty."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    base = y.mean()
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return float(rel), float(res), float(base * (1 - base))


def net_benefit(y, p, threshold):
    """Vickers & Elkin decision-curve analysis."""
    pred = (p >= threshold).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    n = len(y)
    return float(tp / n - (fp / n) * (threshold / (1 - threshold)))


def bootstrap_ci(y, p, fn=roc_auc_score, n_boot=1000, alpha=.05, seed=0):
    rs = np.random.RandomState(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        i = rs.randint(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ----------------------------------------------------------------------
def all_metrics(y, p, threshold=None, dca_thresholds=(.1, .2, .3, .4, .5)):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    thr = best_threshold(y, p) if threshold is None else threshold
    yh = (p >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()

    sens = tp / max(1, tp + fn_)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn_)
    lr_pos = sens / max(1e-9, 1 - spec)
    lr_neg = (1 - sens) / max(1e-9, spec)
    auc = roc_auc_score(y, p)
    ece, mce = expected_calibration_error(y, p)
    rel, res, unc = brier_decomposition(y, p)
    slope, intercept = calibration_slope_intercept(y, p)

    m = dict(
        # ---- threshold-free
        roc_auc=float(auc),
        gini=float(2 * auc - 1),
        avg_precision=float(average_precision_score(y, p)),
        log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        brier=float(brier_score_loss(y, p)),
        brier_skill=float(1 - brier_score_loss(y, p) / (y.mean() * (1 - y.mean()))),
        ks_stat=ks_statistic(y, p),
        # ---- threshold-based (at Youden-optimal cut-off unless given)
        threshold=float(thr),
        accuracy=float(accuracy_score(y, yh)),
        balanced_accuracy=float(balanced_accuracy_score(y, yh)),
        precision=float(ppv), recall=float(sens),
        specificity=float(spec), npv=float(npv),
        f1=float(f1_score(y, yh, zero_division=0)),
        f2=float(fbeta_score(y, yh, beta=2, zero_division=0)),
        mcc=float(matthews_corrcoef(y, yh)),
        kappa=float(cohen_kappa_score(y, yh)),
        youden_j=float(sens + spec - 1),
        lr_positive=float(lr_pos), lr_negative=float(lr_neg),
        diagnostic_or=float(lr_pos / max(1e-9, lr_neg)),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn_),
        # ---- calibration
        ece=ece, mce=mce,
        calib_slope=slope, calib_intercept=intercept,
        brier_reliability=rel, brier_resolution=res, brier_uncertainty=unc,
    )
    for t in dca_thresholds:
        m[f"net_benefit@{t}"] = net_benefit(y, p, t)
    # number needed to screen at the operating point
    m["number_needed_to_screen"] = float((tp + fp) / max(1, tp))
    return m


def metrics_with_ci(y, p, n_boot=1000, seed=0):
    m = all_metrics(y, p)
    for key, fn in [("roc_auc", roc_auc_score),
                    ("avg_precision", average_precision_score)]:
        lo, hi = bootstrap_ci(y, p, fn, n_boot, seed=seed)
        m[f"{key}_ci_lo"], m[f"{key}_ci_hi"] = lo, hi
    return m


# ----------------------------------------------------------------------
# Model-vs-model significance tests
# ----------------------------------------------------------------------
def delong_test(y, p1, p2):
    """DeLong's test for two correlated ROC AUCs. Returns (z, p_value, auc_diff)."""
    from scipy.stats import norm
    y = np.asarray(y).astype(int)
    pos, neg = p1[y == 1], p1[y == 0]
    m, n = len(pos), len(neg)

    def midrank(x):
        s = np.argsort(x)
        xs = x[s]
        r = np.zeros(len(x))
        i = 0
        while i < len(xs):
            j = i
            while j < len(xs) - 1 and xs[j + 1] == xs[i]:
                j += 1
            r[i:j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        out = np.empty(len(x))
        out[s] = r
        return out

    preds = np.vstack([p1, p2])
    k = 2
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    aucs = np.empty(k)
    for r in range(k):
        px, py = preds[r][y == 1], preds[r][y == 0]
        tx[r] = midrank(px); ty[r] = midrank(py)
        tz[r] = midrank(np.concatenate([px, py]))
        aucs[r] = (tz[r, :m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    S = s01 / m + s10 / n
    L = np.array([1, -1])
    var = L @ S @ L
    if var <= 0:
        return 0.0, 1.0, float(aucs[0] - aucs[1])
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(z), float(2 * (1 - norm.cdf(abs(z)))), float(aucs[0] - aucs[1])


def mcnemar_test(y, p1, p2, threshold=.5):
    """McNemar on the discordant classification pairs."""
    from statsmodels.stats.contingency_tables import mcnemar
    a = (p1 >= threshold).astype(int) == y
    b = (p2 >= threshold).astype(int) == y
    tbl = np.array([[(a & b).sum(), (a & ~b).sum()],
                    [(~a & b).sum(), (~a & ~b).sum()]])
    r = mcnemar(tbl, exact=False, correction=True)
    return float(r.statistic), float(r.pvalue)



# ==============================================================================
# REGISTRY
# ==============================================================================
"""
The registry: every one of the 47 requested models plus classical baselines,
each exposed through a single `.fit(X, y)` / `.predict_proba(X)` interface where
X is the tuple (X_num, X_cat).

Availability is checked at import time. Anything whose third-party package is
missing is registered as UNAVAILABLE and reported in the results table rather
than silently dropped, so the final report tells you exactly what ran.
"""

import importlib
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


warnings.filterwarnings("ignore")


def _has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


HAS = {m: _has(m) for m in
       ["pytorch_tabnet", "tabpfn", "tabicl", "transtab", "carte_ai",
        "xgboost", "lightgbm", "catboost", "ctgan", "be_great", "sdv"]}


# ======================================================================
# Adapters for non-torch / special-protocol models
# ======================================================================
class DenseSklearnAdapter(BaseEstimator, ClassifierMixin):
    """Wraps any sklearn-style estimator so it accepts (X_num, X_cat)."""

    def __init__(self, est, name="sk", use_onehot=True, cards=None):
        self.est, self.name, self.use_onehot, self.cards = est, name, use_onehot, cards

    def _flat(self, X):
        Xn, Xc = X
        if not self.use_onehot:
            return np.hstack([Xn, Xc.astype(np.float32)])
        oh = []
        for i in range(Xc.shape[1]):
            k = int(self.cards[i])
            m = np.zeros((len(Xc), k), np.float32)
            m[np.arange(len(Xc)), np.clip(Xc[:, i], 0, k - 1)] = 1
            oh.append(m)
        return np.hstack([Xn] + oh) if oh else Xn

    def fit(self, X, y, eval_set=None):
        self.est.fit(self._flat(X), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(self._flat(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


class GrowNetTrainer(BaseEstimator, ClassifierMixin):
    """
    GrowNet needs stage-wise boosting, not one joint backward pass:
      * stage k is trained to fit the negative gradient of the current ensemble
      * then all stages so far are jointly fine-tuned ("corrective step")
    """

    def __init__(self, n_stages=8, hidden=64, lr=1e-2, lr_stage=0.3,
                 epochs_per_stage=30, corrective_epochs=5, batch_size=256):
        self.n_stages, self.hidden, self.lr = n_stages, hidden, lr
        self.lr_stage = lr_stage
        self.epochs_per_stage = epochs_per_stage
        self.corrective_epochs = corrective_epochs
        self.batch_size = batch_size

    def fit(self, X, y, eval_set=None):
        from sklearn.metrics import roc_auc_score
        seed_all()
        Xn, Xc = X
        cards = [int(Xc[:, i].max()) + 2 for i in range(Xc.shape[1])]
        self.model_ = GrowNet(Xn.shape[1], cards, self.n_stages,
                                self.hidden, self.lr_stage).to(DEVICE)
        xn = torch.as_tensor(Xn, dtype=torch.float32, device=DEVICE)
        xc = torch.as_tensor(Xc, dtype=torch.long, device=DEVICE)
        yy = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
        n = len(y)

        for k in range(self.n_stages):
            self.model_.n_active = k + 1
            for p in self.model_.parameters():
                p.requires_grad_(False)
            for p in self.model_.learners[k].parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(self.model_.learners[k].parameters(), self.lr)
            with torch.no_grad():
                self.model_.n_active = k
                prev = (self.model_(xn, xc) if k > 0
                        else torch.zeros(n, device=DEVICE))
                resid = yy - torch.sigmoid(prev)          # negative gradient of BCE
                self.model_.n_active = k + 1
            for _ in range(self.epochs_per_stage):
                idx = torch.randperm(n, device=DEVICE)[:self.batch_size]
                x = torch.cat([xn[idx], self.model_.emb(xc[idx])], 1)
                h = None
                with torch.no_grad():
                    for j in range(k):
                        _, h = self.model_.learners[j](x, h)
                o, _ = self.model_.learners[k](x, h)
                loss = F.mse_loss(o, resid[idx])
                opt.zero_grad(); loss.backward(); opt.step()

            for p in self.model_.parameters():
                p.requires_grad_(True)
            copt = torch.optim.AdamW(self.model_.parameters(), self.lr * 0.1)
            for _ in range(self.corrective_epochs):
                idx = torch.randperm(n, device=DEVICE)[:self.batch_size]
                loss = F.binary_cross_entropy_with_logits(
                    self.model_(xn[idx], xc[idx]), yy[idx])
                copt.zero_grad(); loss.backward(); copt.step()

        self.classes_ = np.array([0, 1])
        self.n_params_ = sum(p.numel() for p in self.model_.parameters())
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        Xn, Xc = X
        self.model_.eval()
        out = []
        for i in range(0, len(Xn), 4096):
            xn = torch.as_tensor(Xn[i:i + 4096], dtype=torch.float32, device=DEVICE)
            xc = torch.as_tensor(Xc[i:i + 4096], dtype=torch.long, device=DEVICE)
            out.append(torch.sigmoid(self.model_(xn, xc)).cpu().numpy())
        p = np.concatenate(out)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


class SSLTrainer(TorchTabClassifier):
    """Two-stage: self-supervised pretraining on unlabelled X, then supervised
    finetuning of encoder + head."""

    def __init__(self, ssl_cls, pretrain_epochs=60, pretrain_lr=1e-3,
                 freeze_encoder=False, **kw):
        super().__init__(build_fn=lambda n, c, **k: ssl_cls(n, c, **k), **kw)
        self.ssl_cls = ssl_cls
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.freeze_encoder = freeze_encoder

    def fit(self, X, y, eval_set=None):
        seed_all()
        Xn, Xc = X
        cards = [int(Xc[:, i].max()) + 2 for i in range(Xc.shape[1])]
        model = self.ssl_cls(Xn.shape[1], cards).to(DEVICE)
        xn = torch.as_tensor(Xn, dtype=torch.float32, device=DEVICE)
        xc = torch.as_tensor(Xc, dtype=torch.long, device=DEVICE)
        opt = torch.optim.AdamW(model.parameters(), self.pretrain_lr)
        n = len(Xn)
        model.train()
        for _ in range(self.pretrain_epochs):                # ---- stage 1
            idx = torch.randperm(n, device=DEVICE)[:self.batch_size]
            if len(idx) < 4:
                continue
            loss = model.pretrain_loss(xn[idx], xc[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        self._pretrained = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def build(n_num, cardinalities, **k):                 # ---- stage 2
            m = self.ssl_cls(n_num, cardinalities, **k)
            m.load_state_dict(self._pretrained, strict=False)
            if self.freeze_encoder:
                for p in m.enc.parameters():
                    p.requires_grad_(False)
            return m

        self.build_fn = build
        return super().fit(X, y, eval_set)


class TabRTrainer(TorchTabClassifier):
    """TabR needs its retrieval context installed before training."""

    def fit(self, X, y, eval_set=None):
        Xn, Xc = X
        self._ctx = (Xn, Xc, y)
        outer = self

        def build(n_num, cards, **k):
            m = TabR(n_num, cards, **k)
            m = m.to(DEVICE)
            m.set_context(*outer._ctx)
            return m

        self.build_fn = build
        return super().fit(X, y, eval_set)


class DeepGBMTrainer(TorchTabClassifier):
    """Fits a GBDT first, extracts leaf indices, and hands them to DeepGBM as
    the distillation target."""

    def fit(self, X, y, eval_set=None):
        from sklearn.ensemble import GradientBoostingClassifier
        Xn, Xc = X
        flat = np.hstack([Xn, Xc.astype(np.float32)])
        gb = GradientBoostingClassifier(n_estimators=60, max_depth=4,
                                        random_state=SEED).fit(flat, y)
        leaves = gb.apply(flat)[:, :, 0].astype(np.int64)         # n, n_trees
        max_leaf = int(leaves.max()) + 1
        self._leaf = torch.as_tensor(leaves, device=DEVICE)
        self.kw.update(n_trees=leaves.shape[1], max_leaf=max_leaf)
        est = super().fit(X, y, eval_set)
        return est


class TabPFNAdapter(BaseEstimator, ClassifierMixin):
    """TabPFN v1/v2 are pretrained — 'fitting' just stores the context rows."""

    def __init__(self, version=2, max_train=10000):
        self.version, self.max_train = version, max_train

    def fit(self, X, y, eval_set=None):
        from tabpfn import TabPFNClassifier
        Xn, Xc = X
        Z = np.hstack([Xn, Xc.astype(np.float32)])
        n = min(self.max_train, len(y))
        idx = np.random.RandomState(SEED).choice(len(y), n, replace=False)
        self.clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
        self.clf.fit(Z[idx], y[idx])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xn, Xc = X
        return self.clf.predict_proba(np.hstack([Xn, Xc.astype(np.float32)]))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


class TabICLAdapter(TabPFNAdapter):
    def fit(self, X, y, eval_set=None):
        from tabicl import TabICLClassifier
        Xn, Xc = X
        Z = np.hstack([Xn, Xc.astype(np.float32)])
        self.clf = TabICLClassifier()
        self.clf.fit(Z, y)
        self.classes_ = np.array([0, 1])
        return self


class PyTorchTabNetAdapter(BaseEstimator, ClassifierMixin):
    """Official pytorch-tabnet implementation, for cross-checking our native one."""

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y, eval_set=None):
        from pytorch_tabnet.tab_model import TabNetClassifier
        Xn, Xc = X
        Z = np.hstack([Xn, Xc.astype(np.float32)])
        cat_idx = list(range(Xn.shape[1], Z.shape[1]))
        cat_dims = [int(Xc[:, i].max()) + 2 for i in range(Xc.shape[1])]
        self.clf = TabNetClassifier(cat_idxs=cat_idx, cat_dims=cat_dims,
                                    cat_emb_dim=4, verbose=0, seed=SEED, **self.kw)
        if eval_set is not None:
            (Vn, Vc), vy = eval_set
            V = np.hstack([Vn, Vc.astype(np.float32)])
            self.clf.fit(Z, y, eval_set=[(V, vy)], eval_metric=["auc"],
                         max_epochs=200, patience=25, batch_size=1024)
        else:
            self.clf.fit(Z, y, max_epochs=200, patience=25, batch_size=1024)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xn, Xc = X
        return self.clf.predict_proba(np.hstack([Xn, Xc.astype(np.float32)]))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


# ======================================================================
# Baselines (not in the 47, but you need them to know if deep models help)
# ======================================================================
def _baselines(cards):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {
        "LogisticRegression": lambda: DenseSklearnAdapter(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0,
                                             class_weight="balanced")),
            cards=cards),
        "RandomForest": lambda: DenseSklearnAdapter(
            RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   n_jobs=-1, random_state=SEED,
                                   class_weight="balanced_subsample"),
            cards=cards, use_onehot=False),
        "HistGradientBoosting": lambda: DenseSklearnAdapter(
            HistGradientBoostingClassifier(random_state=SEED,
                                           max_iter=400, learning_rate=.06),
            cards=cards, use_onehot=False),
    }
    if HAS["xgboost"]:
        import xgboost as xgb
        out["XGBoost"] = lambda: DenseSklearnAdapter(
            xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=.05,
                              subsample=.8, colsample_bytree=.8, n_jobs=-1,
                              eval_metric="auc", random_state=SEED,
                              tree_method="hist"),
            cards=cards, use_onehot=False)
    if HAS["lightgbm"]:
        import lightgbm as lgb
        out["LightGBM"] = lambda: DenseSklearnAdapter(
            lgb.LGBMClassifier(n_estimators=800, learning_rate=.04, num_leaves=48,
                               subsample=.8, colsample_bytree=.8, verbose=-1,
                               random_state=SEED),
            cards=cards, use_onehot=False)
    if HAS["catboost"]:
        from catboost import CatBoostClassifier
        out["CatBoost"] = lambda: DenseSklearnAdapter(
            CatBoostClassifier(iterations=800, depth=6, learning_rate=.05,
                               verbose=0, random_seed=SEED),
            cards=cards, use_onehot=False)
    return out


# ======================================================================
# THE REGISTRY
# ======================================================================
def build_registry(cards, col_names=None, n_features=None, groups=None):
    """
    Returns {name: (family, callable_or_None, note)}.
    callable() -> a fresh estimator.  None means the model is unavailable.
    """
    def torchmodel(cls, **kw):
        def f(**over):
            k = dict(kw)
            k.update(over)
            train_kw = {p: k.pop(p) for p in list(k)
                        if p in ("lr", "weight_decay", "batch_size", "max_epochs",
                                 "patience", "use_mixup", "swap_noise", "swa",
                                 "label_smoothing", "class_weight", "scheduler")}
            return TorchTabClassifier(build_fn=cls, cardinalities=list(cards),
                                      **train_kw, **k)
        return f

    reg = {}

    # ---------------- baselines ----------------
    for n, f in _baselines(cards).items():
        reg[n] = ("Baseline", f, "reference model")

    # ---------------- 1–6 MLP family ----------------
    reg["MLP"] = ("MLP-family", torchmodel(MLP, hidden=(256, 256, 128),
                                           dropout=.15), "")
    reg["RegularizationCocktails"] = (
        "MLP-family",
        torchmodel(RegCocktailMLP, width=256, n_blocks=4, dropout=.2,
                   use_mixup=True, swap_noise=.1, swa=True,
                   label_smoothing=.05, weight_decay=1e-3),
        "13-ingredient cocktail: mixup + cutmix + SD + shake-shake + SWA + LS + WD")
    reg["RealMLP"] = ("MLP-family",
                      torchmodel(RealMLP, hidden=(256, 256, 256),
                                 label_smoothing=.05, swa=True), "")
    reg["TabM"] = ("MLP-family", torchmodel(TabM, hidden=(256, 256), k=8), "")
    reg["NAM"] = ("MLP-family", torchmodel(NAM, hidden=(64, 32)),
                  "fully interpretable — shape functions plotted in 05_lime.py")
    reg["NODE-GAM"] = ("MLP-family",
                       torchmodel(NODEGAM, n_layers=2, trees_per_feature=4,
                                  depth=4, lr=2e-3), "additive, interpretable")

    # ---------------- 7–16 attention ----------------
    reg["TabNet"] = ("Attention", torchmodel(TabNet, n_d=32, n_a=32, n_steps=4),
                     "native implementation")
    if HAS["pytorch_tabnet"]:
        reg["TabNet(official)"] = ("Attention", lambda: PyTorchTabNetAdapter(),
                                   "pytorch-tabnet package")
    reg["TabTransformer"] = ("Attention",
                             torchmodel(TabTransformer, d_token=32, depth=4), "")
    reg["FT-Transformer"] = ("Attention",
                             torchmodel(FTTransformer, d_token=96, depth=4,
                                        lr=1e-4, weight_decay=1e-5), "")
    reg["SAINT"] = ("Attention", torchmodel(SAINT, d_token=32, depth=3,
                                            batch_size=128, lr=1e-4),
                    "intersample attention — batch size matters")
    reg["AutoInt"] = ("Attention", torchmodel(AutoInt, d_token=32, n_layers=3), "")
    reg["T2G-Former"] = ("Attention", torchmodel(T2GFormer, d_token=64, depth=3,
                                                 lr=1e-4), "")
    reg["ExcelFormer"] = ("Attention", torchmodel(ExcelFormer, d_token=64,
                                                  depth=3, lr=1e-4,
                                                  use_mixup=True), "")
    reg["ARM-Net"] = ("Attention", torchmodel(ARMNet, d_token=16, n_exp=32), "")
    reg["NPT"] = ("Attention", torchmodel(NPT, d_token=32, depth=3,
                                          batch_size=128, lr=1e-4), "")
    tt_kw = {}
    if col_names is not None:
        tt_kw["col_name_vectors"] = hashed_column_vectors(col_names, 64)
    reg["TransTab"] = ("Attention", torchmodel(TransTab, d_token=64, depth=3,
                                               lr=1e-4, **tt_kw),
                       "column-name-aware tokens (hashed header embeddings)")

    # ---------------- 17–23 tree hybrids ----------------
    reg["NODE"] = ("Tree-hybrid", torchmodel(NODE, n_layers=2, n_trees=64,
                                             depth=5, tree_out=2, lr=2e-3), "")
    reg["DeepGBM"] = ("Tree-hybrid",
                      lambda: DeepGBMTrainer(build_fn=DeepGBM,
                                             cardinalities=list(cards)),
                      "distils a fitted GBDT's leaf structure")
    reg["GrowNet"] = ("Tree-hybrid", lambda: GrowNetTrainer(n_stages=8),
                      "stage-wise boosting of shallow MLPs")
    reg["GANDALF"] = ("Tree-hybrid", torchmodel(GANDALF, n_stages=6,
                                                d_state=32), "")
    reg["DANet"] = ("Tree-hybrid", torchmodel(DANet, n_layers=4, d_hidden=64,
                                              k=5), "")
    reg["Net-DNF"] = ("Tree-hybrid", torchmodel(NetDNF, n_formulas=64,
                                                n_literals=8), "")
    reg["TabNN"] = ("Tree-hybrid", torchmodel(TabNN, groups=groups,
                                              n_groups=6), "correlation grouping")

    # ---------------- 24–28 CTR ----------------
    reg["Wide&Deep"] = ("CTR", torchmodel(WideAndDeep, d_emb=16), "")
    reg["DeepFM"] = ("CTR", torchmodel(DeepFM, d_emb=16), "")
    reg["xDeepFM"] = ("CTR", torchmodel(xDeepFM, d_emb=16,
                                        cin_layers=(48, 48)), "")
    reg["DCN"] = ("CTR", torchmodel(DCN, d_emb=16, n_cross=3), "")
    reg["DCNv2"] = ("CTR", torchmodel(DCNv2, d_emb=16, n_cross=3, rank=32,
                                      n_experts=2), "low-rank mixture cross layers")

    # ---------------- 29–35 retrieval / pretrained ----------------
    reg["TabPFN"] = ("Retrieval/Pretrained",
                     (lambda: TabPFNAdapter(version=1)) if HAS["tabpfn"] else None,
                     "pip install tabpfn" if not HAS["tabpfn"] else "prior-fitted, no training")
    reg["TabPFN-v2"] = ("Retrieval/Pretrained",
                        (lambda: TabPFNAdapter(version=2)) if HAS["tabpfn"] else None,
                        "pip install tabpfn" if not HAS["tabpfn"] else "prior-fitted, no training")
    reg["TabICL"] = ("Retrieval/Pretrained",
                     (lambda: TabICLAdapter()) if HAS["tabicl"] else None,
                     "pip install tabicl" if not HAS["tabicl"] else "")
    reg["TabR"] = ("Retrieval/Pretrained",
                   lambda: TabRTrainer(build_fn=TabR, cardinalities=list(cards),
                                       lr=3e-4, batch_size=256),
                   "native differentiable k-NN retrieval")
    reg["CARTE"] = ("Retrieval/Pretrained", None,
                    "needs carte-ai + pretrained checkpoint; designed for "
                    "cross-table transfer, marginal value on a single table")
    reg["XTab"] = ("Retrieval/Pretrained", None,
                   "cross-table federated pretraining; requires the authors' "
                   "checkpoints — run the FT-Transformer backbone instead")
    reg["TabDPT"] = ("Retrieval/Pretrained", None,
                     "decision-pretrained transformer; requires released weights")

    # ---------------- 36–37 sequence backbones ----------------
    reg["MambaTab"] = ("Sequence", torchmodel(MambaTab, d_model=32, d_state=16),
                       "native S6 recurrence (no CUDA kernel needed)")
    reg["TabMamba"] = ("Sequence", torchmodel(TabMamba, d_model=48, n_blocks=2,
                                              lr=5e-4), "bidirectional variant")

    # ---------------- 38–41 self-supervised ----------------
    reg["VIME"] = ("Self-supervised", lambda: SSLTrainer(VIME,
                                                         pretrain_epochs=80), "")
    reg["SubTab"] = ("Self-supervised", lambda: SSLTrainer(SubTab,
                                                           pretrain_epochs=80), "")
    reg["SCARF"] = ("Self-supervised", lambda: SSLTrainer(SCARF,
                                                          pretrain_epochs=80), "")
    reg["DenoisingAE"] = ("Self-supervised", lambda: SSLTrainer(DenoisingAE,
                                                                pretrain_epochs=100), "")

    return reg


GENERATIVE = {
    "CTGAN":     ("ctgan package",  "42"),
    "CTAB-GAN+": ("native",         "43"),
    "TVAE":      ("ctgan package",  "44"),
    "TabDDPM":   ("native",         "45"),
    "TabSyn":    ("native",         "46"),
    "GReaT":     ("be-great pkg",   "47"),
}


def availability_report(reg):
    import pandas as pd
    rows = [dict(model=k, family=v[0],
                 status="available" if v[1] is not None else "UNAVAILABLE",
                 note=v[2]) for k, v in reg.items()]
    return pd.DataFrame(rows)



# ==============================================================================
# 01_EDA
# ==============================================================================
"""
STEP 1 — Exploratory Data Analysis.

Produces figs/eda_*.png and outputs/eda_*.csv
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
PAL = {0: "#4C72B0", 1: "#C44E52"}


def _save_eda(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"eda_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figs/eda_{name}.png")


# ----------------------------------------------------------------------
def eda_profile_table(df, num, cat):
    """Structural eda_profile_table: dtypes, missingness, cardinality, descriptives."""
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append(dict(
            column=c, dtype=str(s.dtype), n_unique=s.nunique(),
            n_missing=int(s.isna().sum()),
            pct_missing=round(100 * s.isna().mean(), 3),
            mean=round(float(s.mean()), 3) if c in num else np.nan,
            std=round(float(s.std()), 3) if c in num else np.nan,
            min=float(s.min()) if c in num else np.nan,
            q25=float(s.quantile(.25)) if c in num else np.nan,
            median=float(s.median()) if c in num else np.nan,
            q75=float(s.quantile(.75)) if c in num else np.nan,
            max=float(s.max()) if c in num else np.nan,
            skew=round(float(s.skew()), 3) if c in num else np.nan,
            kurtosis=round(float(s.kurtosis()), 3) if c in num else np.nan,
            top_value=str(s.mode().iloc[0]) if len(s.mode()) else "",
        ))
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "eda_profile.csv", index=False)
    print("  outputs/eda_profile.csv")
    return p


def target_balance(df):
    vc = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["No disease (0)", "Disease (1)"], vc.values,
              color=[PAL[0], PAL[1]])
    for i, v in enumerate(vc.values):
        ax[0].text(i, v, f"{v}\n({v/len(df):.1%})", ha="center", va="bottom")
    ax[0].set_title("Class distribution")
    ax[0].set_ylabel("count")
    ax[1].pie(vc.values, labels=["No disease", "Disease"], autopct="%1.1f%%",
              colors=[PAL[0], PAL[1]], startangle=90,
              wedgeprops=dict(width=.45, edgecolor="w"))
    ax[1].set_title(f"Imbalance ratio = {vc[0]/vc[1]:.2f} : 1")
    _save_eda(fig, "01_target_balance")
    return vc


def numeric_distributions(df, num):
    n = len(num)
    ncol, nrow = 5, int(np.ceil(n / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.9 * nrow))
    for ax, c in zip(axes.ravel(), num):
        for k, g in df.groupby(TARGET):
            sns.kdeplot(g[c], ax=ax, fill=True, alpha=.35,
                        color=PAL[k], label=f"y={k}", warn_singular=False)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Numeric feature densities by outcome", y=1.002)
    _save_eda(fig, "02_numeric_kde")


def numeric_boxplots(df, num):
    n = len(num)
    ncol, nrow = 5, int(np.ceil(n / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow))
    for ax, c in zip(axes.ravel(), num):
        sns.boxplot(data=df, x=TARGET, y=c, ax=ax, palette=PAL,
                    hue=TARGET, legend=False, fliersize=1.5)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Numeric features — box plots by outcome (outlier view)", y=1.002)
    _save_eda(fig, "03_numeric_box")


def categorical_rates(df, cat):
    n = len(cat)
    ncol, nrow = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow))
    tabs = {}
    for ax, c in zip(np.atleast_1d(axes).ravel(), cat):
        ct = pd.crosstab(df[c].astype(str), df[TARGET], normalize="index")
        cnt = df[c].astype(str).value_counts()
        tabs[c] = pd.crosstab(df[c].astype(str), df[TARGET])
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=[PAL[0], PAL[1]], legend=False, width=.7)
        for i, lvl in enumerate(ct.index):
            ax.text(i, 1.01, f"n={cnt[lvl]}", ha="center", fontsize=7)
        ax.axhline(df[TARGET].mean(), ls="--", c="k", lw=1)
        ax.set_title(f"{c}  (dashed = base rate {df[TARGET].mean():.1%})",
                     fontsize=9)
        ax.set_ylabel("proportion")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    _save_eda(fig, "04_categorical_rates")
    pd.concat(tabs, names=["feature", "level"]).to_csv(
        OUT / "eda_categorical_crosstabs.csv")
    print("  outputs/eda_categorical_crosstabs.csv")


def correlation(df, num):
    corr = df[num + [TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.42 * len(corr) + 3, 0.38 * len(corr) + 2))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, ax=ax,
                square=True, linewidths=.3, cbar_kws=dict(shrink=.6),
                annot=False, vmin=-1, vmax=1)
    ax.set_title("Spearman correlation matrix")
    _save_eda(fig, "05_correlation_heatmap")
    corr.to_csv(OUT / "eda_correlation_spearman.csv")

    tc = corr[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(tc) + 2))
    ax.barh(tc.index, tc.values,
            color=[PAL[1] if v > 0 else PAL[0] for v in tc.values])
    ax.axvline(0, c="k", lw=.8)
    ax.set_title("Spearman correlation with target")
    _save_eda(fig, "06_target_correlation")

    # highly collinear pairs
    up = corr.where(mask)
    pairs = (up.stack().rename("rho").reset_index()
               .rename(columns={"level_0": "a", "level_1": "b"}))
    pairs["abs_rho"] = pairs.rho.abs()
    pairs = pairs[pairs.abs_rho > .8].sort_values("abs_rho", ascending=False)
    pairs.to_csv(OUT / "eda_collinear_pairs.csv", index=False)
    print(f"  outputs/eda_collinear_pairs.csv  ({len(pairs)} pairs |rho|>0.8)")
    return corr


def outliers_iqr(df, num):
    rows = []
    for c in num:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        m = (df[c] < lo) | (df[c] > hi)
        z = np.abs((df[c] - df[c].mean()) / df[c].std())
        rows.append(dict(feature=c, n_iqr_outliers=int(m.sum()),
                         pct_iqr=round(100 * m.mean(), 2),
                         n_z_gt3=int((z > 3).sum()),
                         lower_fence=round(lo, 2), upper_fence=round(hi, 2)))
    o = pd.DataFrame(rows).sort_values("pct_iqr", ascending=False)
    o.to_csv(OUT / "eda_outliers.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(o) + 2))
    ax.barh(o.feature, o.pct_iqr, color="#DD8452")
    ax.set_xlabel("% rows outside 1.5·IQR fences")
    ax.set_title("Outlier burden per numeric feature")
    ax.invert_yaxis()
    _save_eda(fig, "07_outliers")
    return o


def age_sex_risk(df):
    d = df.copy()
    d["age_band"] = pd.cut(d.age, [0, 35, 45, 55, 65, 75, 200],
                           labels=["<35", "35-44", "45-54", "55-64", "65-74", "75+"])
    piv = d.pivot_table(index="age_band", columns="sex",
                        values=TARGET, aggfunc="mean", observed=True)
    cnt = d.pivot_table(index="age_band", columns="sex",
                        values=TARGET, aggfunc="size", observed=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv.plot(kind="bar", ax=ax[0], color=["#C44E52", "#4C72B0"])
    ax[0].set_ylabel("P(heart disease)")
    ax[0].set_title("Prevalence by age band × sex")
    ax[0].tick_params(axis="x", rotation=0)
    sns.heatmap(piv, annot=True, fmt=".2%", cmap="Reds", ax=ax[1],
                cbar_kws=dict(label="prevalence"))
    ax[1].set_title("Same, as heatmap")
    _save_eda(fig, "08_age_sex_risk")
    piv.to_csv(OUT / "eda_age_sex_prevalence.csv")
    cnt.to_csv(OUT / "eda_age_sex_counts.csv")


def bivariate_scatter(df):
    pairs = [("age", "max_heart_rate_achieved"),
             ("bmi", "hba1c"),
             ("cholesterol_total", "hdl"),
             ("resting_bp_systolic", "resting_bp_diastolic"),
             ("exercise_minutes_per_week", "daily_steps"),
             ("stress_score", "sleep_hours")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (a, b) in zip(axes.ravel(), pairs):
        for k, g in df.groupby(TARGET):
            ax.scatter(g[a], g[b], s=6, alpha=.35, color=PAL[k], label=f"y={k}")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.legend(fontsize=7, markerscale=2)
    fig.suptitle("Selected bivariate relationships coloured by outcome")
    _save_eda(fig, "09_bivariate_scatter")


def mutual_information(df, num, cat):
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import OrdinalEncoder
    X = df[num].copy()
    for c in cat:
        X[c] = OrdinalEncoder().fit_transform(df[[c]].astype(str))
    disc = [c in cat for c in X.columns]
    mi = mutual_info_classif(X, df[TARGET], discrete_features=disc,
                             random_state=SEED)
    s = pd.Series(mi, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 2))
    ax.barh(s.index, s.values, color="#55A868")
    ax.set_title("Mutual information with target (nats)")
    _save_eda(fig, "10_mutual_information")
    s.sort_values(ascending=False).to_csv(OUT / "eda_mutual_information.csv",
                                          header=["mutual_info"])
    return s


def pca_umap(df, num):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    X = StandardScaler().fit_transform(df[num])
    p = PCA().fit(X)
    Z = p.transform(X)[:, :2]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].plot(np.arange(1, len(p.explained_variance_ratio_) + 1),
               np.cumsum(p.explained_variance_ratio_), "o-")
    ax[0].axhline(.95, ls="--", c="r")
    ax[0].set_xlabel("components")
    ax[0].set_ylabel("cumulative explained variance")
    ax[0].set_title("PCA scree")
    for k in (0, 1):
        m = df[TARGET] == k
        ax[1].scatter(Z[m, 0], Z[m, 1], s=6, alpha=.35, color=PAL[k], label=f"y={k}")
    ax[1].set_title("PCA projection")
    ax[1].legend(markerscale=2)
    sub = np.random.RandomState(SEED).choice(len(X), min(3000, len(X)), replace=False)
    T = TSNE(n_components=2, random_state=SEED, init="pca",
             perplexity=30).fit_transform(X[sub])
    yy = df[TARGET].values[sub]
    for k in (0, 1):
        m = yy == k
        ax[2].scatter(T[m, 0], T[m, 1], s=6, alpha=.4, color=PAL[k], label=f"y={k}")
    ax[2].set_title("t-SNE (3k subsample)")
    ax[2].legend(markerscale=2)
    _save_eda(fig, "11_pca_tsne")


def gbdt_importance(df, num, cat):
    """Quick tree-based signal check — tells you the ceiling before deep models."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split
    X = df[num + cat].copy()
    for c in cat:
        X[c] = OrdinalEncoder().fit_transform(df[[c]].astype(str))
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25,
                                          stratify=y, random_state=SEED)
    m = HistGradientBoostingClassifier(
        random_state=SEED, categorical_features=[X.columns.get_loc(c) for c in cat]
    ).fit(Xtr, ytr)
    r = permutation_importance(m, Xte, yte, n_repeats=8,
                               random_state=SEED, scoring="roc_auc")
    s = pd.Series(r.importances_mean, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 2))
    ax.barh(s.index, s.values, xerr=r.importances_std[np.argsort(r.importances_mean)],
            color="#8172B3")
    ax.set_title("Permutation importance (HistGB, ΔAUC)")
    _save_eda(fig, "12_permutation_importance")
    s.sort_values(ascending=False).to_csv(OUT / "eda_permutation_importance.csv",
                                          header=["delta_auc"])
    from sklearn.metrics import roc_auc_score
    print(f"  [sanity] HistGB holdout AUC = "
          f"{roc_auc_score(yte, m.predict_proba(Xte)[:,1]):.4f}")
    return s


def duplicate_check(df):
    feats = [c for c in df.columns if c != TARGET]
    dup_full = int(df.duplicated().sum())
    dup_feat = int(df.duplicated(subset=feats).sum())
    conflict = 0
    if dup_feat:
        g = df.groupby(feats, observed=True)[TARGET].nunique()
        conflict = int((g > 1).sum())
    print(f"  duplicates: exact={dup_full}, feature-identical={dup_feat}, "
          f"contradictory-label groups={conflict}")
    return dict(exact=dup_full, feature_identical=dup_feat, contradictory=conflict)


# ----------------------------------------------------------------------
def eda_main():
    print("STEP 1 — EDA")
    raw = load_raw()
    df = engineer(raw)
    num, cat = column_types(df)
    print(f"  shape={df.shape}  numeric={len(num)}  categorical={len(cat)}")

    eda_profile_table(df, num, cat)
    duplicate_check(df)
    target_balance(df)
    numeric_distributions(df, num)
    numeric_boxplots(df, num)
    categorical_rates(df, cat)
    correlation(df, num)
    outliers_iqr(df, num)
    age_sex_risk(df)
    bivariate_scatter(df)
    mutual_information(df, num, cat)
    pca_umap(df, num)
    gbdt_importance(df, num, cat)
    print("EDA complete.\n")



# ==============================================================================
# 02_STATISTICS
# ==============================================================================
"""
STEP 2 — Classical statistical analysis.

Covers the "statistical methods" pillar:
   Distributional screening      (normality, variance homogeneity)
  B. Univariate association tests  (t / Welch / Mann–Whitney, chi² / Fisher,
                                    with effect sizes and FDR correction)
   Multicollinearity             (VIF, condition number)
  D. Inferential multivariable model (statsmodels logistic regression:
                                    coefficients, odds ratios, 95% CI, Wald p,
                                    Hosmer–Lemeshow goodness of fit)
  E. Stratified / interaction analysis (Cochran–Mantel–Haenszel, Breslow–Day)
  F. Power analysis and bootstrap CIs

Outputs: outputs/stats_*.csv, figs/stats_*.png
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.contingency_tables import StratifiedTable


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def _save_stats(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"stats_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figs/stats_{name}.png")


# ======================================================================
#  Distributional screening
# ======================================================================
def normality_and_variance(df, num):
    rows = []
    g0 = df[df[TARGET] == 0]
    g1 = df[df[TARGET] == 1]
    for c in num:
        x = df[c].dropna().values
        xs = x if len(x) <= 5000 else np.random.RandomState(SEED).choice(x, 5000, False)
        sw_s, sw_p = st.shapiro(xs)
        dag_s, dag_p = st.normaltest(x)
        ks_s, ks_p = st.kstest((x - x.mean()) / x.std(), "norm")
        a0 = g0[c].to_numpy(dtype=float)
        a1 = g1[c].to_numpy(dtype=float)
        lev_s, lev_p = st.levene(a0, a1, center="median")
        bar_s, bar_p = st.bartlett(a0, a1)
        rows.append(dict(
            feature=c, shapiro_W=sw_s, shapiro_p=sw_p,
            dagostino_K2=dag_s, dagostino_p=dag_p,
            ks_D=ks_s, ks_p=ks_p,
            skew=st.skew(x), kurtosis=st.kurtosis(x),
            levene_p=lev_p, bartlett_p=bar_p,
            normal_at_05=bool(sw_p > .05),
            equal_var_at_05=bool(lev_p > .05),
        ))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "stats_A_normality.csv", index=False)
    print(f"  outputs/stats_A_normality.csv  "
          f"({int(t.normal_at_05.sum())}/{len(t)} features pass Shapiro at α=.05)")

    # Q-Q grid
    n = len(num)
    ncol, nrow = 5, int(np.ceil(n / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.8 * nrow))
    for ax, c in zip(axes.ravel(), num):
        st.probplot(df[c].dropna(), dist="norm", plot=ax)
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_lines()[0].set_markersize(2)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Q–Q plots vs normal", y=1.002)
    _save_stats(fig, "A_qq_plots")
    return t


# ======================================================================
# B. Univariate association tests
# ======================================================================
def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def rank_biserial(a, b):
    u = st.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 1 - 2 * u / (len(a) * len(b))


def cramers_v(ct):
    chi2 = st.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = chi2 / n
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1)))


def numeric_tests(df, num, norm_tbl):
    g0 = df[df[TARGET] == 0]
    g1 = df[df[TARGET] == 1]
    rows = []
    for c in num:
        a, b = g1[c].values, g0[c].values          # a = disease, b = healthy
        eq = bool(norm_tbl.loc[norm_tbl.feature == c, "equal_var_at_05"].iloc[0])
        t_s, t_p = st.ttest_ind(a, b, equal_var=eq)
        u_s, u_p = st.mannwhitneyu(a, b, alternative="two-sided")
        ks_s, ks_p = st.ks_2samp(a, b)
        # bootstrap CI on the difference in means
        rs = np.random.RandomState(SEED)
        boot = [rs.choice(a, len(a)).mean() - rs.choice(b, len(b)).mean()
                for _ in range(1000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(
            feature=c,
            mean_disease=a.mean(), mean_healthy=b.mean(),
            median_disease=np.median(a), median_healthy=np.median(b),
            mean_diff=a.mean() - b.mean(), ci95_lo=lo, ci95_hi=hi,
            test_used="Student t" if eq else "Welch t",
            t_stat=t_s, t_p=t_p,
            mannwhitney_U=u_s, mannwhitney_p=u_p,
            ks2_D=ks_s, ks2_p=ks_p,
            cohens_d=cohens_d(a, b), rank_biserial=rank_biserial(a, b),
        ))
    t = pd.DataFrame(rows)
    for col, new in [("t_p", "t_p_fdr"), ("mannwhitney_p", "mw_p_fdr")]:
        t[new] = multipletests(t[col], method="fdr_bh")[1]
    t["signif_fdr_05"] = t.mw_p_fdr < .05
    t["effect_size_label"] = pd.cut(
        t.cohens_d.abs(), [-.01, .2, .5, .8, 99],
        labels=["negligible", "small", "medium", "large"])
    t = t.sort_values("mw_p_fdr")
    t.to_csv(OUT / "stats_B_numeric_tests.csv", index=False)
    print(f"  outputs/stats_B_numeric_tests.csv  "
          f"({int(t.signif_fdr_05.sum())}/{len(t)} significant after BH-FDR)")

    s = t.set_index("feature").cohens_d.sort_values()
    fig, ax = plt.subplots(figsize=(7, .28 * len(s) + 2))
    ax.barh(s.index, s.values,
            color=["#C44E52" if v > 0 else "#4C72B0" for v in s.values])
    for x in (-.8, -.5, -.2, .2, .5, .8):
        ax.axvline(x, ls=":", c="grey", lw=.7)
    ax.axvline(0, c="k", lw=1)
    ax.set_xlabel("Cohen's d  (disease − healthy)")
    ax.set_title("Standardised effect sizes (dotted = .2/.5/.8 thresholds)")
    _save_stats(fig, "B_effect_sizes")

    # volcano
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = t.cohens_d.values
    y = -np.log10(np.clip(t.mw_p_fdr.values, 1e-300, None))
    ax.scatter(x, y, s=45, c=np.where(t.signif_fdr_05, "#C44E52", "grey"),
               alpha=.8, edgecolor="k", linewidth=.4)
    ax.axhline(-np.log10(.05), ls="--", c="k", lw=1)
    for _, r in t.head(12).iterrows():
        ax.annotate(r.feature, (r.cohens_d, -np.log10(max(r.mw_p_fdr, 1e-300))),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("−log10(FDR-adjusted p)")
    ax.set_title("Volcano plot — univariate screening")
    _save_stats(fig, "B_volcano")
    return t


def categorical_tests(df, cat):
    rows = []
    for c in cat:
        ct = pd.crosstab(df[c].astype(str), df[TARGET])
        chi2, p, dof, exp = st.chi2_contingency(ct)
        g, gp, _, _ = st.chi2_contingency(ct, lambda_="log-likelihood")
        min_exp = exp.min()
        fisher_p = np.nan
        odds = np.nan
        if ct.shape == (2, 2):
            odds, fisher_p = st.fisher_exact(ct)
        rows.append(dict(
            feature=c, levels=ct.shape[0], chi2=chi2, dof=dof, chi2_p=p,
            g_test=g, g_p=gp, fisher_p=fisher_p, odds_ratio_2x2=odds,
            cramers_v=cramers_v(ct), min_expected_count=min_exp,
            chi2_assumption_ok=bool(min_exp >= 5),
        ))
    t = pd.DataFrame(rows)
    t["chi2_p_fdr"] = multipletests(t.chi2_p, method="fdr_bh")[1]
    t["signif_fdr_05"] = t.chi2_p_fdr < .05
    t = t.sort_values("chi2_p_fdr")
    t.to_csv(OUT / "stats_B_categorical_tests.csv", index=False)
    print(f"  outputs/stats_B_categorical_tests.csv  "
          f"({int(t.signif_fdr_05.sum())}/{len(t)} significant)")

    fig, ax = plt.subplots(figsize=(7, .4 * len(t) + 2))
    ax.barh(t.feature, t.cramers_v, color="#55A868")
    ax.set_xlabel("Cramér's V")
    ax.set_title("Categorical association strength with target")
    ax.invert_yaxis()
    _save_stats(fig, "B_cramers_v")
    return t


# ======================================================================
#  Multicollinearity
# ======================================================================
def vif_analysis(df, num):
    X = sm.add_constant(df[num].astype(float))
    vals = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    t = pd.DataFrame(dict(feature=num, VIF=vals)).sort_values("VIF", ascending=False)
    t["severity"] = pd.cut(t.VIF, [0, 5, 10, 1e9],
                           labels=["ok", "moderate", "severe"])
    t.to_csv(OUT / "stats_C_vif.csv", index=False)
    Xs = (df[num] - df[num].mean()) / df[num].std()
    cond = np.linalg.cond(Xs.values)
    print(f"  outputs/stats_C_vif.csv  (condition number = {cond:.1f}; "
          f"{int((t.VIF>10).sum())} features VIF>10)")

    fig, ax = plt.subplots(figsize=(7, .28 * len(t) + 2))
    colors = ["#C44E52" if v > 10 else "#DD8452" if v > 5 else "#55A868"
              for v in t.VIF]
    ax.barh(t.feature, np.clip(t.VIF, 0, 60), color=colors)
    ax.axvline(5, ls="--", c="grey")
    ax.axvline(10, ls="--", c="k")
    ax.set_xlabel("VIF (clipped at 60)")
    ax.set_title("Variance inflation factors")
    ax.invert_yaxis()
    _save_stats(fig, "C_vif")
    return t, cond


# ======================================================================
# D. Multivariable logistic regression (inference, not prediction)
# ======================================================================
def logistic_inference(df, num, cat, vif_tbl, max_vif=10.0):
    keep = vif_tbl.loc[vif_tbl.VIF <= max_vif, "feature"].tolist()
    dropped = sorted(set(num) - set(keep))
    X = df[keep].astype(float).copy()
    X = (X - X.mean()) / X.std()                     # standardised -> comparable ORs
    D = pd.get_dummies(df[cat].astype(str), drop_first=True).astype(float)
    X = pd.concat([X, D], axis=1)
    X = sm.add_constant(X)
    y = df[TARGET].astype(float)

    res = sm.Logit(y, X).fit(disp=0, maxiter=200)
    out = pd.DataFrame(dict(
        term=res.params.index, coef=res.params.values,
        std_err=res.bse.values, z=res.tvalues.values, p_value=res.pvalues.values,
        odds_ratio=np.exp(res.params.values),
        or_ci_lo=np.exp(res.conf_int()[0].values),
        or_ci_hi=np.exp(res.conf_int()[1].values),
    ))
    out["p_fdr"] = multipletests(out.p_value, method="fdr_bh")[1]
    out = out.sort_values("p_value")
    out.to_csv(OUT / "stats_D_logit_coefficients.csv", index=False)

    fit = dict(
        n=int(res.nobs), df_model=int(res.df_model),
        llf=float(res.llf), llnull=float(res.llnull),
        pseudo_r2_mcfadden=float(res.prsquared),
        aic=float(res.aic), bic=float(res.bic),
        lr_stat=float(res.llr), lr_p=float(res.llr_pvalue),
        dropped_for_vif=", ".join(dropped),
    )
    # Hosmer–Lemeshow
    p = res.predict(X)
    dfr = pd.DataFrame(dict(y=y.values, p=p.values))
    dfr["decile"] = pd.qcut(dfr.p, 10, duplicates="drop", labels=False)
    g = dfr.groupby("decile").agg(obs=("y", "sum"), n=("y", "size"), exp=("p", "sum"))
    hl = (((g.obs - g.exp) ** 2) / (g.exp * (1 - g.exp / g.n))).sum()
    hl_dof = len(g) - 2
    fit["hosmer_lemeshow_chi2"] = float(hl)
    fit["hosmer_lemeshow_p"] = float(1 - st.chi2.cdf(hl, hl_dof))
    pd.Series(fit).to_csv(OUT / "stats_D_logit_fit.csv", header=["value"])
    print(f"  outputs/stats_D_logit_coefficients.csv / _fit.csv  "
          f"(McFadden R²={fit['pseudo_r2_mcfadden']:.3f}, "
          f"HL p={fit['hosmer_lemeshow_p']:.3f})")

    # forest plot
    top = out[out.term != "const"].nsmallest(20, "p_value").sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(8, .35 * len(top) + 2))
    ypos = np.arange(len(top))
    ax.errorbar(top.odds_ratio, ypos,
                xerr=[top.odds_ratio - top.or_ci_lo, top.or_ci_hi - top.odds_ratio],
                fmt="o", color="#4C72B0", ecolor="grey", capsize=3)
    ax.axvline(1, ls="--", c="k")
    ax.set_yticks(ypos)
    ax.set_yticklabels(top.term)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (log scale, 95% CI) — per 1 SD for numeric terms")
    ax.set_title("Multivariable logistic regression — top 20 terms")
    _save_stats(fig, "D_forest_odds_ratios")

    # calibration of the inferential model
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(g.exp / g.n, g.obs / g.n, "o-", color="#C44E52")
    ax.plot([0, 1], [0, 1], "--", c="k")
    ax.set_xlabel("mean predicted probability (decile)")
    ax.set_ylabel("observed fraction")
    ax.set_title(f"Hosmer–Lemeshow calibration (p={fit['hosmer_lemeshow_p']:.3f})")
    _save_stats(fig, "D_hl_calibration")
    return out, fit


# ======================================================================
# E. Stratified analysis / confounding
# ======================================================================
def stratified_analysis(df):
    """CMH test: is exercise-induced angina associated with disease after
    stratifying by sex and age band (classic confounder control)?"""
    d = df.copy()
    d["age_band"] = pd.cut(d.age, [0, 45, 55, 65, 200],
                           labels=["<45", "45-54", "55-64", "65+"])
    tables, keys = [], []
    for (s, ab), g in d.groupby(["sex", "age_band"], observed=True):
        ct = pd.crosstab(g["exercise_induced_angina"], g[TARGET])
        if ct.shape == (2, 2) and (ct.values > 0).all():
            tables.append(ct.values)
            keys.append(f"{s}/{ab}")
    stt = StratifiedTable(np.dstack(tables))
    res = dict(
        strata=len(tables),
        pooled_or_mh=float(stt.oddsratio_pooled),
        or_ci_lo=float(stt.oddsratio_pooled_confint()[0]),
        or_ci_hi=float(stt.oddsratio_pooled_confint()[1]),
        cmh_chi2=float(stt.test_null_odds().statistic),
        cmh_p=float(stt.test_null_odds().pvalue),
        breslow_day_chi2=float(stt.test_equal_odds().statistic),
        breslow_day_p=float(stt.test_equal_odds().pvalue),
    )
    crude = pd.crosstab(d["exercise_induced_angina"], d[TARGET]).values
    res["crude_or"] = float((crude[1, 1] * crude[0, 0]) / (crude[1, 0] * crude[0, 1]))
    res["confounding_pct_change"] = 100 * (res["crude_or"] - res["pooled_or_mh"]) / res["crude_or"]
    pd.Series(res).to_csv(OUT / "stats_E_cmh_angina.csv", header=["value"])
    print(f"  outputs/stats_E_cmh_angina.csv  "
          f"(crude OR={res['crude_or']:.2f} -> MH OR={res['pooled_or_mh']:.2f}, "
          f"Breslow–Day p={res['breslow_day_p']:.3f})")
    return res


def interaction_scan(df, num, cat, top_k=6):
    """Likelihood-ratio tests for pairwise interactions among top features."""
    from scipy.stats import chi2 as chi2_dist
    imp = pd.read_csv(OUT / "eda_mutual_information.csv", index_col=0)
    feats = [f for f in imp.index[:top_k] if f in num]
    y = df[TARGET].astype(float)
    base_cols = df[feats].astype(float)
    base_cols = (base_cols - base_cols.mean()) / base_cols.std()
    rows = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            a, b = feats[i], feats[j]
            X0 = sm.add_constant(base_cols[[a, b]])
            X1 = X0.copy()
            X1[f"{a}:{b}"] = base_cols[a] * base_cols[b]
            m0 = sm.Logit(y, X0).fit(disp=0)
            m1 = sm.Logit(y, X1).fit(disp=0)
            lr = 2 * (m1.llf - m0.llf)
            rows.append(dict(term=f"{a} × {b}", lr_stat=lr,
                             p_value=1 - chi2_dist.cdf(lr, 1),
                             interaction_coef=m1.params[f"{a}:{b}"]))
    t = pd.DataFrame(rows)
    t["p_fdr"] = multipletests(t.p_value, method="fdr_bh")[1]
    t = t.sort_values("p_value")
    t.to_csv(OUT / "stats_E_interactions.csv", index=False)
    print(f"  outputs/stats_E_interactions.csv  "
          f"({int((t.p_fdr<.05).sum())}/{len(t)} interactions significant)")
    return t


# ======================================================================
# F. Power analysis
# ======================================================================
def power_analysis(df, numeric_tbl):
    from statsmodels.stats.power import TTestIndPower, NormalIndPower
    n1 = int((df[TARGET] == 1).sum())
    n0 = int((df[TARGET] == 0).sum())
    ratio = n0 / n1
    pw = TTestIndPower()
    rows = []
    for _, r in numeric_tbl.iterrows():
        power = pw.power(effect_size=abs(r.cohens_d), nobs1=n1,
                         alpha=.05, ratio=ratio)
        need = np.nan
        if abs(r.cohens_d) > .01:
            try:
                need = pw.solve_power(effect_size=abs(r.cohens_d), power=.8,
                                      alpha=.05, ratio=ratio)
            except Exception:
                pass
        rows.append(dict(feature=r.feature, cohens_d=r.cohens_d,
                         achieved_power=power, n1_needed_for_80pct=need))
    t = pd.DataFrame(rows).sort_values("achieved_power")
    t.to_csv(OUT / "stats_F_power.csv", index=False)
    mde = pw.solve_power(effect_size=None, nobs1=n1, alpha=.05,
                         power=.8, ratio=ratio)
    print(f"  outputs/stats_F_power.csv  (n1={n1}, n0={n0}; "
          f"minimum detectable d at 80% power = {mde:.3f})")
    fig, ax = plt.subplots(figsize=(7, .28 * len(t) + 2))
    ax.barh(t.feature, t.achieved_power, color="#4C72B0")
    ax.axvline(.8, ls="--", c="r")
    ax.set_xlabel("achieved power (α=.05)")
    ax.set_title("Statistical power per feature")
    _save_stats(fig, "F_power")
    return t


# ======================================================================
def stats_main():
    print("STEP 2 — Statistical analysis")
    df = engineer(load_raw())
    num, cat = column_types(df)

    norm = normality_and_variance(df, num)
    numt = numeric_tests(df, num, norm)
    categorical_tests(df, cat)
    vif, cond = vif_analysis(df, num)
    logistic_inference(df, num, cat, vif)
    stratified_analysis(df)
    interaction_scan(df, num, cat)
    power_analysis(df, numt)
    print("Statistics complete.\n")



# ==============================================================================
# 03_TRAIN
# ==============================================================================
"""
STEP 3 — Train every model with stratified 5-fold CV + a held-out test set.

Protocol
--------
  * 20% of rows held out once as a true test set, never touched during training.
  * On the remaining 80%: stratified 5-fold CV. Each fold trains on 4/5 and uses
    the held-out 1/5 for early stopping AND as the out-of-fold prediction.
  * Fold models are then averaged into an ensemble prediction on the test set,
    so every model gets both a CV score (with a std) and a test score.
  * Preprocessors are re-fitted inside every fold on the training part only —
    no target or distribution leakage across folds.

Outputs
-------
  outputs/results_summary.csv    one row per model, all metrics
  outputs/oof_predictions.npz    out-of-fold probabilities (for stats + LIME)
  outputs/test_predictions.npz   test-set probabilities
  outputs/fold_scores.csv        per-fold AUC/AP, for significance testing
"""

import argparse
import time
import traceback
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


def train_all(models=None, quick=False, max_epochs=None, skip=()):
    ds = Dataset()
    y = ds.y
    print(f"Dataset: {ds.summary()}")

    # groups for TabNN, computed once from the training portion
    Xn0, Xc0, prep0 = ds.mixed()
    groups = correlation_groups(np.hstack([Xn0, Xc0.astype(float)])[ds.tr_idx], 6)

    reg = build_registry(prep0.cardinalities, col_names=ds.num + ds.cat,
                         groups=groups)
    avail = availability_report(reg)
    avail.to_csv(OUT / "model_availability.csv", index=False)
    print(f"Registry: {(avail.status=='available').sum()} available, "
          f"{(avail.status=='UNAVAILABLE').sum()} unavailable\n")

    names = models or (QUICK_MODELS if quick else list(reg))
    names = [n for n in names if n in reg and reg[n][1] is not None and n not in skip]

    oof = {}
    testp = {}
    rows = []
    fold_rows = []

    for name in names:
        fam, fn, note = reg[name]
        print(f"[{name}] ({fam})")
        t0 = time.time()
        try:
            oof_p = np.full(len(y), np.nan)
            test_fold_p = []
            f_scores = []

            for k, (tr, va) in enumerate(ds.folds):
                prep = MixedPrep(ds.num, ds.cat).fit(ds.raw, tr)
                Xn, Xc = prep.transform(ds.raw)
                est = fn()
                if max_epochs and hasattr(est, "max_epochs"):
                    est.max_epochs = max_epochs
                try:
                    est.fit((Xn[tr], Xc[tr]), y[tr],
                            eval_set=((Xn[va], Xc[va]), y[va]))
                except TypeError:
                    est.fit((Xn[tr], Xc[tr]), y[tr])
                p_va = est.predict_proba((Xn[va], Xc[va]))[:, 1]
                oof_p[va] = p_va
                test_fold_p.append(est.predict_proba((Xn[ds.te_idx],
                                                      Xc[ds.te_idx]))[:, 1])
                m = all_metrics(y[va], p_va)
                f_scores.append(m)
                fold_rows.append(dict(model=name, fold=k, **m))
                print(f"   fold {k}: auc={m['roc_auc']:.4f} ap={m['avg_precision']:.4f}")

            p_test = np.mean(test_fold_p, axis=0)
            oof[name] = oof_p
            testp[name] = p_test

            mask = ~np.isnan(oof_p)
            cv = all_metrics(y[mask], oof_p[mask])
            te = all_metrics(y[ds.te_idx], p_test)
            row = dict(model=name, family=fam, note=note,
                       runtime_s=round(time.time() - t0, 1),
                       n_params=getattr(est, "n_params_", np.nan))
            row.update({f"cv_{k}": v for k, v in cv.items()})
            row.update({f"test_{k}": v for k, v in te.items()})
            for k in ("roc_auc", "avg_precision"):
                row[f"fold_{k}_mean"] = float(np.mean([s[k] for s in f_scores]))
                row[f"fold_{k}_std"] = float(np.std([s[k] for s in f_scores]))
            rows.append(row)
            print(f"   -> CV AUC {cv['roc_auc']:.4f} "
                  f"(±{row['fold_roc_auc_std']:.4f})  "
                  f"TEST AUC {te['roc_auc']:.4f}  [{row['runtime_s']}s]\n")

        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            rows.append(dict(model=name, family=fam, note=f"FAILED: {e}"))

    res = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
    res.to_csv(OUT / "results_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_scores.csv", index=False)
    np.savez_compressed(OUT / "oof_predictions.npz", y=y, **oof)
    np.savez_compressed(OUT / "test_predictions.npz",
                        y=y[ds.te_idx], idx=ds.te_idx, **testp)

    print("\n" + "=" * 78)
    cols = ["model", "family", "cv_roc_auc", "fold_roc_auc_std",
            "test_roc_auc", "test_avg_precision", "test_brier", "runtime_s"]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].head(25).to_string(index=False))
    print("=" * 78)
    print("\noutputs/results_summary.csv, fold_scores.csv, "
          "oof_predictions.npz, test_predictions.npz")
    return res



# ==============================================================================
# 04_EVALUATE_VIZ
# ==============================================================================
"""
STEP 4 — Evaluation, visualisation and model comparison.

Reads outputs/{results_summary,fold_scores,oof_predictions,test_predictions}
and produces the full comparison figure set plus formal significance_tests tests.

Figures
-------
  eval_01_leaderboard            ranked test AUC with bootstrap CIs
  eval_02_family_comparison      box plots by architecture family
  eval_03_roc_curves             ROC for the top-k models
  eval_04_pr_curves              precision-recall for the top-k
  eval_05_calibration            reliability diagrams + histogram
  eval_06_confusion_matrices     confusion matrix grid for the top-k
  eval_07_threshold_sweep        metric-vs-threshold for the best model
  eval_08_decision_curve         net benefit vs treat-all / treat-none
  eval_09_metric_heatmap         all models × all metrics, z-scored
  eval_10_critical_difference    Nemenyi CD diagram over folds
  eval_11_prediction_correlation model agreement clustermap
  eval_12_runtime_vs_auc         accuracy/compute trade-off
  eval_13_lift_gain              lift and cumulative gain curves
  eval_14_error_analysis         where the best model fails
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_curve, average_precision_score, roc_auc_score)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
TOPK = 8


def _save_eval(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"eval_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figs/eval_{name}.png")


def load_results():
    res = pd.read_csv(OUT / "results_summary.csv")
    res = res[res.test_roc_auc.notna()].sort_values("test_roc_auc", ascending=False)
    folds = pd.read_csv(OUT / "fold_scores.csv")
    te = np.load(OUT / "test_predictions.npz")
    y = te["y"]
    preds = {k: te[k] for k in te.files if k not in ("y", "idx")}
    return res, folds, y, preds


# ----------------------------------------------------------------------
def leaderboard(res, y, preds):
    rows = []
    for _, r in res.iterrows():
        p = preds.get(r.model)
        if p is None:
            continue
        lo, hi = bootstrap_ci(y, p, roc_auc_score, 500, seed=SEED)
        rows.append(dict(model=r.model, family=r.family, auc=r.test_roc_auc,
                         lo=lo, hi=hi))
    t = pd.DataFrame(rows).sort_values("auc")
    t.to_csv(OUT / "eval_leaderboard_ci.csv", index=False)

    fams = sorted(t.family.unique())
    cmap = dict(zip(fams, sns.color_palette("tab10", len(fams))))
    fig, ax = plt.subplots(figsize=(9, .3 * len(t) + 2))
    ypos = np.arange(len(t))
    ax.barh(ypos, t.auc, color=[cmap[f] for f in t.family],
            xerr=[t.auc - t.lo, t.hi - t.auc], error_kw=dict(lw=1, ecolor="k"))
    ax.set_yticks(ypos)
    ax.set_yticklabels(t.model, fontsize=8)
    ax.set_xlim(max(.5, t.lo.min() - .02), 1.0)
    ax.set_xlabel("Test ROC AUC (95% bootstrap CI)")
    ax.set_title("Model leaderboard")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[f]) for f in fams]
    ax.legend(handles, fams, fontsize=7, loc="lower right")
    _save_eval(fig, "01_leaderboard")
    return t


def family_comparison(res):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    order = res.groupby("family").test_roc_auc.median().sort_values().index
    sns.boxplot(data=res, y="family", x="test_roc_auc", order=order, ax=ax[0],
                hue="family", legend=False, palette="tab10")
    sns.stripplot(data=res, y="family", x="test_roc_auc", order=order, ax=ax[0],
                  color="k", size=4, alpha=.6)
    ax[0].set_title("Test AUC by architecture family")
    sns.boxplot(data=res, y="family", x="runtime_s", order=order, ax=ax[1],
                hue="family", legend=False, palette="tab10")
    ax[1].set_xscale("log")
    ax[1].set_title("Training cost by family (log seconds)")
    _save_eval(fig, "02_family_comparison")


def roc_pr_curves(res, y, preds, k=TOPK):
    top = res.head(k).model.tolist()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for m in top:
        fpr, tpr, _ = roc_curve(y, preds[m])
        ax.plot(fpr, tpr, lw=1.6,
                label=f"{m} ({roc_auc_score(y, preds[m]):.4f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC curves — top {k} models")
    ax.legend(fontsize=8, loc="lower right")
    _save_eval(fig, "03_roc_curves")

    base = y.mean()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for m in top:
        pr, rc, _ = precision_recall_curve(y, preds[m])
        ax.plot(rc, pr, lw=1.6,
                label=f"{m} ({average_precision_score(y, preds[m]):.4f})")
    ax.axhline(base, ls="--", c="grey", label=f"no-skill ({base:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–recall curves — top {k} models")
    ax.legend(fontsize=8, loc="lower left")
    _save_eval(fig, "04_pr_curves")
    return top


def calibration_plots(res, y, preds, top):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5),
                           gridspec_kw=dict(width_ratios=[1.2, 1]))
    for m in top:
        pt, pp = calibration_curve(y, preds[m], n_bins=12, strategy="quantile")
        ax[0].plot(pp, pt, "o-", ms=4, lw=1.4, label=m)
    ax[0].plot([0, 1], [0, 1], "--", c="k")
    ax[0].set_xlabel("mean predicted probability")
    ax[0].set_ylabel("observed frequency")
    ax[0].set_title("Reliability diagram (quantile bins)")
    ax[0].legend(fontsize=7)
    for m in top[:4]:
        ax[1].hist(preds[m], bins=40, alpha=.45, label=m)
    ax[1].set_xlabel("predicted probability")
    ax[1].set_title("Prediction distributions")
    ax[1].legend(fontsize=7)
    _save_eval(fig, "05_calibration")

    t = pd.DataFrame([dict(model=m, **{k: v for k, v in all_metrics(y, preds[m]).items()
                                       if k.startswith(("ece", "mce", "calib", "brier"))})
                      for m in res.model if m in preds])
    t.to_csv(OUT / "eval_calibration.csv", index=False)


def confusions(y, preds, top):
    n = len(top)
    nc = 4
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.3 * nr))
    for ax, m in zip(np.atleast_1d(axes).ravel(), top):
        mt = all_metrics(y, preds[m])
        cm = confusion_matrix(y, (preds[m] >= mt["threshold"]).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["pred 0", "pred 1"],
                    yticklabels=["true 0", "true 1"])
        ax.set_title(f"{m}\nsens={mt['recall']:.3f} spec={mt['specificity']:.3f}",
                     fontsize=8)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    _save_eval(fig, "06_confusion_matrices")


def threshold_sweep(y, preds, best):
    p = preds[best]
    ts = np.linspace(.02, .98, 97)
    rec = []
    for t in ts:
        m = all_metrics(y, p, threshold=t)
        rec.append(dict(t=t, **{k: m[k] for k in
                                ("precision", "recall", "specificity", "f1",
                                 "balanced_accuracy", "mcc", "youden_j", "npv")}))
    d = pd.DataFrame(rec)
    d.to_csv(OUT / "eval_threshold_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c in d.columns.drop("t"):
        ax.plot(d.t, d[c], lw=1.5, label=c)
    ax.axvline(d.t[d.youden_j.idxmax()], ls="--", c="k",
               label=f"Youden opt = {d.t[d.youden_j.idxmax()]:.2f}")
    ax.set_xlabel("classification threshold")
    ax.set_title(f"Threshold sensitivity — {best}")
    ax.legend(fontsize=8, ncol=2)
    _save_eval(fig, "07_threshold_sweep")


def decision_curve(y, preds, top):
    ts = np.linspace(.02, .7, 60)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    prev = y.mean()
    ax.plot(ts, [prev - (1 - prev) * t / (1 - t) for t in ts], "--",
            c="grey", label="treat all")
    ax.axhline(0, c="k", lw=1, label="treat none")
    for m in top[:5]:
        ax.plot(ts, [net_benefit(y, preds[m], t) for t in ts], lw=1.6, label=m)
    ax.set_ylim(-.05, prev + .02)
    ax.set_xlabel("threshold probability (harm/benefit ratio)")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision-curve analysis")
    ax.legend(fontsize=8)
    _save_eval(fig, "08_decision_curve")


def metric_heatmap(res, y, preds):
    keys = ["roc_auc", "avg_precision", "brier_skill", "ks_stat",
            "balanced_accuracy", "f1", "mcc", "kappa", "youden_j",
            "recall", "specificity", "precision", "npv", "ece"]
    rows = []
    for m in res.model:
        if m not in preds:
            continue
        mt = all_metrics(y, preds[m])
        rows.append(dict(model=m, **{k: mt[k] for k in keys}))
    d = pd.DataFrame(rows).set_index("model")
    d.to_csv(OUT / "eval_all_metrics.csv")
    z = (d - d.mean()) / d.std().replace(0, 1)
    z["ece"] *= -1                                    # lower ECE is better
    fig, ax = plt.subplots(figsize=(.9 * len(keys) + 3, .3 * len(z) + 3))
    sns.heatmap(z, cmap="RdYlGn", center=0, ax=ax, linewidths=.3,
                cbar_kws=dict(label="z-score (higher = better)"))
    ax.set_title("All models × all metrics (column-standardised)")
    _save_eval(fig, "09_metric_heatmap")
    return d


def critical_difference(folds):
    """Friedman + Nemenyi post-hoc over per-fold AUC ranks."""
    from scipy.stats import friedmanchisquare
    piv = folds.pivot_table(index="fold", columns="model", values="roc_auc")
    piv = piv.dropna(axis=1)
    if piv.shape[1] < 3:
        return None
    stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    k, n = piv.shape[1], piv.shape[0]
    q_alpha = 3.219 if k <= 10 else 3.9                # approx q for alpha=.05
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(piv.values)
        nem.index = nem.columns = piv.columns
        nem.to_csv(OUT / "eval_nemenyi_pvalues.csv")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, .22 * len(ranks) + 3))
    ax.barh(ranks.index, ranks.values, color="#4C72B0")
    ax.set_xlabel("mean rank across folds (lower = better)")
    ax.set_title(f"Friedman χ²={stat:.1f}, p={p:.2e}  |  Nemenyi CD={cd:.2f}")
    ax.invert_yaxis()
    _save_eval(fig, "10_critical_difference")
    pd.DataFrame(dict(model=ranks.index, mean_rank=ranks.values)).to_csv(
        OUT / "eval_mean_ranks.csv", index=False)
    return dict(friedman_chi2=float(stat), friedman_p=float(p), cd=float(cd))


def prediction_correlation(res, preds, k=20):
    top = [m for m in res.head(k).model if m in preds]
    P = pd.DataFrame({m: preds[m] for m in top})
    corr = P.corr(method="spearman")
    corr.to_csv(OUT / "eval_prediction_correlation.csv")
    g = sns.clustermap(corr, cmap="viridis", figsize=(11, 10),
                       annot=False, linewidths=.2)
    g.fig.suptitle("Model agreement (Spearman of predicted probabilities)", y=1.01)
    g.savefig(FIGS / "eval_11_prediction_correlation.png", dpi=130,
              bbox_inches="tight")
    plt.close("all")
    print("  figs/eval_11_prediction_correlation.png")


def runtime_tradeoff(res):
    fig, ax = plt.subplots(figsize=(9, 6))
    fams = sorted(res.family.unique())
    cmap = dict(zip(fams, sns.color_palette("tab10", len(fams))))
    for f in fams:
        s = res[res.family == f]
        ax.scatter(s.runtime_s, s.test_roc_auc, s=70, label=f,
                   color=cmap[f], edgecolor="k", linewidth=.5)
    for _, r in res.iterrows():
        ax.annotate(r.model, (r.runtime_s, r.test_roc_auc), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("total 5-fold training time (s, log)")
    ax.set_ylabel("test ROC AUC")
    ax.set_title("Accuracy / compute trade-off")
    ax.legend(fontsize=7)
    _save_eval(fig, "12_runtime_vs_auc")


def lift_gain(y, preds, top):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    base = y.mean()
    for m in top[:6]:
        o = np.argsort(-preds[m])
        ys = y[o]
        cum = np.cumsum(ys) / ys.sum()
        pct = np.arange(1, len(ys) + 1) / len(ys)
        ax[0].plot(pct, cum, lw=1.5, label=m)
        ax[1].plot(pct, (np.cumsum(ys) / np.arange(1, len(ys) + 1)) / base,
                   lw=1.5, label=m)
    ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set_xlabel("fraction of population screened")
    ax[0].set_ylabel("fraction of cases captured")
    ax[0].set_title("Cumulative gain")
    ax[0].legend(fontsize=7)
    ax[1].axhline(1, ls="--", c="grey")
    ax[1].set_xlabel("fraction screened")
    ax[1].set_ylabel("lift over base rate")
    ax[1].set_title("Lift curve")
    ax[1].legend(fontsize=7)
    _save_eval(fig, "13_lift_gain")


def error_analysis(y, preds, best):
    ds = Dataset()
    te = np.load(OUT / "test_predictions.npz")
    idx = te["idx"]
    df = ds.raw.iloc[idx].copy()
    p = preds[best]
    thr = all_metrics(y, p)["threshold"]
    df["p"] = p
    df["err"] = np.where((p >= thr) & (y == 0), "false positive",
                         np.where((p < thr) & (y == 1), "false negative", "correct"))
    df.to_csv(OUT / "eval_error_rows.csv", index=False)

    feats = ["age", "pct_max_hr", "st_depression", "bmi", "chol_hdl_ratio",
             "stress_score"]
    feats = [f for f in feats if f in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.ravel(), feats):
        sns.boxplot(data=df, x="err", y=f, ax=ax, hue="err", legend=False,
                    order=["correct", "false positive", "false negative"],
                    palette=["#55A868", "#DD8452", "#C44E52"])
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12, labelsize=8)
    fig.suptitle(f"Where {best} goes wrong")
    _save_eval(fig, "14_error_analysis")


def significance_tests(res, y, preds):
    """DeLong + McNemar of every model against the best one."""
    best = res.iloc[0].model
    rows = []
    for m in res.model:
        if m == best or m not in preds:
            continue
        z, p, d = delong_test(y, preds[best], preds[m])
        s, mp = mcnemar_test(y, preds[best], preds[m],
                             all_metrics(y, preds[best])["threshold"])
        rows.append(dict(reference=best, model=m, auc_diff=d, delong_z=z,
                         delong_p=p, mcnemar_stat=s, mcnemar_p=mp,
                         significant_05=bool(p < .05)))
    t = pd.DataFrame(rows).sort_values("delong_p")
    from statsmodels.stats.multitest import multipletests
    if len(t):
        t["delong_p_holm"] = multipletests(t.delong_p, method="holm")[1]
    t.to_csv(OUT / "eval_significance_vs_best.csv", index=False)
    n_sig = int(t.significant_05.sum()) if len(t) else 0
    print(f"  outputs/eval_significance_vs_best.csv  "
          f"({n_sig}/{len(t)} models differ significantly from {best})")
    return t


# ----------------------------------------------------------------------
def eval_main():
    print("STEP 4 — Evaluation & visualisation")
    res, folds, y, preds = load_results()
    print(f"  {len(res)} models with predictions, test n={len(y)}")

    leaderboard(res, y, preds)
    family_comparison(res)
    top = roc_pr_curves(res, y, preds)
    calibration_plots(res, y, preds, top)
    confusions(y, preds, top)
    threshold_sweep(y, preds, top[0])
    decision_curve(y, preds, top)
    metric_heatmap(res, y, preds)
    critical_difference(folds)
    prediction_correlation(res, preds)
    runtime_tradeoff(res)
    lift_gain(y, preds, top)
    error_analysis(y, preds, top[0])
    significance_tests(res, y, preds)
    print("Evaluation complete.\n")



# ==============================================================================
# 05_LIME
# ==============================================================================
"""
STEP 5 — Explainability.

LIME is the centrepiece, with three cross-checks so you can tell whether a LIME
explanation is trustworthy or an artefact of its local surrogate:

   LIME local explanations for individual patients (TP / TN / FP / FN)
  B. LIME global view — aggregate |weight| over a sample of patients
   LIME lime_stability_analysis — repeat the same explanation N times and measure variance
     (LIME resamples, so unstable explanations are a real and common failure)
  D. Cross-checks: SHAP kernel values, permutation importance, and the models
     that are intrinsically interpretable (NAM shape functions, TabNet masks,
     GANDALF gates, logistic odds ratios)

Everything runs against the SAME preprocessing the models were trained with,
wrapped so LIME sees human-readable raw feature values.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def _save_lime(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"lime_{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figs/lime_{name}.png")


# ======================================================================
# Bridge: raw dataframe values -> model inputs
# ======================================================================
class RawPredictor:
    """
    LIME perturbs a numeric matrix. This wraps (encoding + model) so LIME can
    work in the ORIGINAL feature space: numeric columns stay numeric, and
    categoricals are passed as integer level codes that LIME treats as
    categorical features.
    """

    def __init__(self, model, prep: MixedPrep, num, cat, raw_df):
        self.model, self.prep, self.num, self.cat = model, prep, num, cat
        self.feature_names = list(num) + list(cat)
        self.cat_index = list(range(len(num), len(num) + len(cat)))
        self.cat_levels = {i + len(num): list(map(str, cs))
                           for i, cs in enumerate(prep.oe.categories_)}
        self.raw_df = raw_df

    def matrix(self, df):
        """Raw dataframe -> the plain numeric matrix LIME operates on."""
        Xn = df[self.num].to_numpy(dtype=float)
        codes = self.prep.oe.transform(df[self.cat].astype(str))
        return np.hstack([Xn, codes])

    def predict_proba(self, M):
        M = np.atleast_2d(M)
        n = len(self.num)
        Xn = self.prep.sc.transform(M[:, :n])
        Xc = np.clip(M[:, n:].round().astype(np.int64) + 1, 0, None)
        p = self.model.predict_proba((Xn.astype(np.float32), Xc))
        return p


# ======================================================================
# A / B / C — LIME
# ======================================================================
def build_explainer(rp, train_matrix, y_train):
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(
        training_data=train_matrix,
        feature_names=rp.feature_names,
        class_names=["no disease", "heart disease"],
        categorical_features=rp.cat_index,
        categorical_names=rp.cat_levels,
        training_labels=y_train,
        discretize_continuous=True,
        discretizer="quartile",
        mode="classification",
        random_state=SEED,
    )


def local_explanations(exp, rp, M_test, y_test, p_test, n_features=12):
    """One panel each for a confident TP, TN, FP and FN."""
    thr = all_metrics(y_test, p_test)["threshold"]
    yhat = (p_test >= thr).astype(int)
    picks = {}
    for label, mask, order in [
        ("true positive", (y_test == 1) & (yhat == 1), -p_test),
        ("true negative", (y_test == 0) & (yhat == 0), p_test),
        ("false positive", (y_test == 0) & (yhat == 1), -p_test),
        ("false negative", (y_test == 1) & (yhat == 0), p_test),
    ]:
        idx = np.where(mask)[0]
        if len(idx):
            picks[label] = idx[np.argsort(order[idx])[0]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    records = []
    for ax, (label, i) in zip(axes.ravel(), picks.items()):
        e = exp.explain_instance(M_test[i], rp.predict_proba,
                                 num_features=n_features, num_samples=5000)
        pairs = e.as_list()
        names = [p[0] for p in pairs][::-1]
        vals = [p[1] for p in pairs][::-1]
        ax.barh(names, vals,
                color=["#C44E52" if v > 0 else "#4C72B0" for v in vals])
        ax.axvline(0, c="k", lw=.8)
        ax.set_title(f"{label}  |  true={y_test[i]}  p={p_test[i]:.3f}  "
                     f"(R²={e.score:.2f})", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        for nm, v in pairs:
            records.append(dict(case=label, row=int(i), feature=nm, weight=v,
                                local_r2=e.score, pred=p_test[i], truth=int(y_test[i])))
    fig.suptitle("LIME local explanations — one representative case per outcome type")
    _save_lime(fig, "01_local_explanations")
    pd.DataFrame(records).to_csv(OUT / "lime_local_explanations.csv", index=False)
    return picks


def global_from_lime(exp, rp, M, n_samples=150, n_features=15, seed=SEED):
    """Aggregate LIME weights over many patients to get a global picture."""
    rs = np.random.RandomState(seed)
    idx = rs.choice(len(M), min(n_samples, len(M)), replace=False)
    acc, signed, r2 = {}, {}, []
    for i in idx:
        e = exp.explain_instance(M[i], rp.predict_proba,
                                 num_features=n_features, num_samples=2000)
        r2.append(e.score)
        for fid, w in e.as_map()[1]:
            nm = rp.feature_names[fid]
            acc[nm] = acc.get(nm, 0) + abs(w)
            signed[nm] = signed.get(nm, 0) + w
    n = len(idx)
    t = pd.DataFrame(dict(
        feature=list(acc), mean_abs_weight=[v / n for v in acc.values()],
        mean_signed_weight=[signed[k] / n for k in acc])
    ).sort_values("mean_abs_weight", ascending=False)
    t.to_csv(OUT / "lime_global_importance.csv", index=False)
    print(f"  mean local surrogate R² = {np.mean(r2):.3f} "
          f"(low values mean LIME's linear fit is a poor local approximation)")

    s = t.head(20).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(15, .38 * len(s) + 2))
    ax[0].barh(s.feature, s.mean_abs_weight, color="#4C72B0")
    ax[0].set_xlabel("mean |LIME weight|")
    ax[0].set_title(f"LIME global importance (n={n} patients)")
    ax[1].barh(s.feature, s.mean_signed_weight,
               color=["#C44E52" if v > 0 else "#4C72B0" for v in s.mean_signed_weight])
    ax[1].axvline(0, c="k", lw=.8)
    ax[1].set_xlabel("mean signed weight (+ = pushes toward disease)")
    ax[1].set_title("Direction of effect")
    _save_lime(fig, "02_global_importance")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(r2, bins=25, color="#55A868", edgecolor="k")
    ax.axvline(np.mean(r2), c="r", ls="--", label=f"mean={np.mean(r2):.3f}")
    ax.set_xlabel("local surrogate R²")
    ax.set_title("How well LIME's linear model fits locally")
    ax.legend()
    _save_lime(fig, "03_surrogate_fit_quality")
    return t


def lime_stability_analysis(exp, rp, M, row, n_repeats=25, n_features=10):
    """LIME resamples every call — quantify how much the explanation moves."""
    runs = []
    for r in range(n_repeats):
        e = exp.explain_instance(M[row], rp.predict_proba,
                                 num_features=n_features, num_samples=2000)
        runs.append({rp.feature_names[f]: w for f, w in e.as_map()[1]})
    d = pd.DataFrame(runs).fillna(0)
    summ = pd.DataFrame(dict(mean=d.mean(), std=d.std(),
                             cv=(d.std() / d.mean().abs().replace(0, np.nan)),
                             appearance_rate=(d != 0).mean()))
    summ = summ.sort_values("mean", key=abs, ascending=False)
    summ.to_csv(OUT / "lime_stability.csv")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    s = summ.iloc[:12].iloc[::-1]
    ax[0].barh(s.index, s["mean"], xerr=s["std"], color="#4C72B0",
               error_kw=dict(ecolor="k", lw=1))
    ax[0].axvline(0, c="k", lw=.8)
    ax[0].set_title(f"LIME weight ± SD over {n_repeats} repeat explanations\n"
                    f"(same patient, same model)")
    sns.boxplot(data=d[s.index[::-1]], orient="h", ax=ax[1], color="#DD8452")
    ax[1].axvline(0, c="k", lw=.8)
    ax[1].set_title("Full distribution per feature")
    _save_lime(fig, "04_stability")
    return summ


# ======================================================================
# D — cross-checks
# ======================================================================
def shap_crosscheck(rp, M_bg, M_test, n_bg=100, n_test=120):
    try:
        import shap
    except ImportError:
        print("  (shap not installed — skipping cross-check)")
        return None
    rs = np.random.RandomState(SEED)
    bg = M_bg[rs.choice(len(M_bg), min(n_bg, len(M_bg)), replace=False)]
    te = M_test[rs.choice(len(M_test), min(n_test, len(M_test)), replace=False)]
    ex = shap.KernelExplainer(lambda z: rp.predict_proba(z)[:, 1], bg)
    sv = ex.shap_values(te, nsamples=200, silent=True)
    imp = pd.Series(np.abs(sv).mean(0), index=rp.feature_names).sort_values()
    imp.to_csv(OUT / "shap_importance.csv", header=["mean_abs_shap"])
    fig, ax = plt.subplots(figsize=(7, .3 * len(imp) + 2))
    ax.barh(imp.index, imp.values, color="#8172B3")
    ax.set_title("Mean |SHAP value| (KernelSHAP)")
    _save_lime(fig, "05_shap_importance")
    return imp


def compare_attributions(lime_tbl):
    """Do LIME, SHAP and permutation importance agree? If not, be suspicious."""
    frames = {"LIME": lime_tbl.set_index("feature").mean_abs_weight}
    for path, col, name in [
        (OUT / "shap_importance.csv", "mean_abs_shap", "SHAP"),
        (OUT / "eda_permutation_importance.csv", "delta_auc", "Permutation"),
        (OUT / "eda_mutual_information.csv", "mutual_info", "MutualInfo"),
    ]:
        if path.exists():
            frames[name] = pd.read_csv(path, index_col=0)[col]
    d = pd.DataFrame(frames).dropna(how="all")
    if d.shape[1] < 2:
        return None
    ranks = d.rank(ascending=False)
    ranks.to_csv(OUT / "attribution_rank_comparison.csv")
    corr = d.corr(method="spearman")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                ax=ax[0], vmin=-1, vmax=1)
    ax[0].set_title("Spearman agreement between attribution methods")
    top = ranks.loc[ranks.mean(1).nsmallest(15).index]
    sns.heatmap(top, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax[1])
    ax[1].set_title("Rank of each feature by method (1 = most important)")
    _save_lime(fig, "06_attribution_agreement")
    return corr


def nam_shape_functions(ds, prep, n_points=200):
    """NAM is a GAM — plot the learned f_j(x_j) directly. This is ground truth
    for what an additive model believes, and a hard check on LIME."""
    import torch
    reg = build_registry(prep.cardinalities, col_names=ds.num + ds.cat)
    est = reg["NAM"][1]()
    Xn, Xc = prep.transform(ds.raw)
    est.fit((Xn[ds.tr_idx], Xc[ds.tr_idx]), ds.y[ds.tr_idx])
    model = est.model_.eval()

    n = len(ds.num)
    ncol, nrow = 5, int(np.ceil(n / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow))
    rows = []
    for ax, (j, col) in zip(axes.ravel(), enumerate(ds.num)):
        lo, hi = np.percentile(Xn[:, j], [1, 99])
        grid = np.linspace(lo, hi, n_points)
        xb = np.zeros((n_points, Xn.shape[1]), np.float32)
        xb[:, j] = grid
        cb = np.zeros((n_points, Xc.shape[1]), np.int64)
        with torch.no_grad():
            f = model.nets[j](torch.tensor(grid, dtype=torch.float32,
                                           device=DEVICE).unsqueeze(1))
        f = f.squeeze(-1).cpu().numpy()
        raw = prep.sc.inverse_transform(xb)[:, j]
        ax.plot(raw, f - f.mean(), lw=1.8, color="#C44E52")
        ax.axhline(0, ls="--", c="grey", lw=.8)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
        rows += [dict(feature=col, x=float(a), f=float(b))
                 for a, b in zip(raw, f - f.mean())]
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("NAM learned shape functions  f_j(x_j)  "
                 "(centred; y = log-odds contribution)", y=1.002)
    _save_lime(fig, "07_nam_shape_functions")
    pd.DataFrame(rows).to_csv(OUT / "nam_shape_functions.csv", index=False)


def tabnet_masks(ds, prep):
    """TabNet's attention masks are a native, per-step feature-selection
    explanation — free interpretability, no surrogate needed."""
    import torch
    reg = build_registry(prep.cardinalities, col_names=ds.num + ds.cat)
    est = reg["TabNet"][1]()
    Xn, Xc = prep.transform(ds.raw)
    est.fit((Xn[ds.tr_idx], Xc[ds.tr_idx]), ds.y[ds.tr_idx])
    m = est.model_.eval()
    with torch.no_grad():
        sub = ds.te_idx[:512]
        m(torch.tensor(Xn[sub], device=DEVICE),
          torch.tensor(Xc[sub], device=DEVICE))
    masks = [x.cpu().numpy() for x in m.masks_]
    names = list(ds.num) + [f"{c}[emb]" for c in ds.cat]
    agg = np.mean(masks, axis=0).mean(0)
    names = (names + [f"emb{i}" for i in range(len(agg))])[:len(agg)]

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    step_imp = np.stack([mm.mean(0) for mm in masks])
    sns.heatmap(step_imp, ax=ax[0], cmap="viridis",
                yticklabels=[f"step {i}" for i in range(len(masks))],
                xticklabels=names, cbar_kws=dict(label="mean mask weight"))
    ax[0].set_title("TabNet per-step feature selection masks")
    ax[0].tick_params(axis="x", rotation=90, labelsize=6)
    s = pd.Series(agg, index=names).sort_values().tail(20)
    ax[1].barh(s.index, s.values, color="#55A868")
    ax[1].set_title("Aggregate TabNet feature importance")
    ax[1].tick_params(labelsize=7)
    _save_lime(fig, "08_tabnet_masks")
    pd.Series(agg, index=names).sort_values(ascending=False).to_csv(
        OUT / "tabnet_mask_importance.csv", header=["mask_weight"])


# ======================================================================
def lime_main(model_name=None):
    print("STEP 5 — LIME & explainability")
    ds = Dataset()
    prep = MixedPrep(ds.num, ds.cat).fit(ds.raw, ds.tr_idx)
    Xn, Xc = prep.transform(ds.raw)

    # pick the best model from step 3 unless told otherwise
    if model_name is None:
        try:
            res = pd.read_csv(OUT / "results_summary.csv")
            res = res[res.test_roc_auc.notna()]
            model_name = res.sort_values("test_roc_auc", ascending=False).iloc[0].model
        except Exception:
            model_name = "MLP"
    print(f"  explaining: {model_name}")

    reg = build_registry(prep.cardinalities, col_names=ds.num + ds.cat)
    est = reg[model_name][1]()
    est.fit((Xn[ds.tr_idx], Xc[ds.tr_idx]), ds.y[ds.tr_idx])

    rp = RawPredictor(est, prep, ds.num, ds.cat, ds.raw)
    M_train = rp.matrix(ds.raw.iloc[ds.tr_idx])
    M_test = rp.matrix(ds.raw.iloc[ds.te_idx])
    y_test = ds.y[ds.te_idx]
    p_test = rp.predict_proba(M_test)[:, 1]
    print(f"  sanity: test AUC via LIME wrapper = "
          f"{all_metrics(y_test, p_test)['roc_auc']:.4f}")

    exp = build_explainer(rp, M_train, ds.y[ds.tr_idx])
    picks = local_explanations(exp, rp, M_test, y_test, p_test)
    lime_tbl = global_from_lime(exp, rp, M_test, n_samples=150)
    lime_stability_analysis(exp, rp, M_test, list(picks.values())[0])
    shap_crosscheck(rp, M_train, M_test)
    compare_attributions(lime_tbl)
    nam_shape_functions(ds, prep)
    tabnet_masks(ds, prep)
    print("Explainability complete.\n")



# ==============================================================================
# ORCHESTRATION
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Heart Disease Risk 2026 pipeline")
    ap.add_argument("--steps", nargs="*", type=int, default=[1, 2, 3, 4, 5],
                    help="which steps to run, e.g. --steps 1 2")
    ap.add_argument("--quick", action="store_true",
                    help="train a small fast subset of models")
    ap.add_argument("--models", nargs="*", default=None,
                    help="explicit model list for step 3")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--skip", nargs="*", default=["SAINT", "NPT"],
                    help="models to skip (these two are the slow ones)")
    ap.add_argument("--explain-model", default=None,
                    help="which model step 5 should explain (default: best)")
    a = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("HEART DISEASE RISK 2026 — FULL PIPELINE")
    print(f"device: {DEVICE}   |   steps: {a.steps}")
    print("=" * 78 + "\n")

    if 1 in a.steps:
        eda_main()
    if 2 in a.steps:
        stats_main()
    if 3 in a.steps:
        train_all(a.models, a.quick, a.max_epochs, set(a.skip))
    if 4 in a.steps:
        eval_main()
    if 5 in a.steps:
        lime_main(a.explain_model)

    print("=" * 78)
    print(f"PIPELINE COMPLETE in {time.time() - t0:.0f}s")
    print(f"  figures -> {FIGS}")
    print(f"  tables  -> {OUT}")
    print("=" * 78)


if __name__ == "__main__":
    main()
