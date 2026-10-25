"""Ore grade classifier (high / medium / low) with SHAP explanations."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .synthetic import ELEMENTS

CLASSES = ["low", "medium", "high"]


def engineer(df):
    """Log-transform assays (they are skewed) and add common pathfinder ratios."""
    X = pd.DataFrame(index=df.index)
    for e in ELEMENTS:
        X[f"log_{e}"] = np.log1p(df[e].clip(lower=0))
    X["Cu_Zn_ratio"] = np.log1p(df["Cu_ppm"]) - np.log1p(df["Zn_ppm"])
    X["Au_As_ratio"] = np.log1p(df["Au_ppb"]) - np.log1p(df["As_ppm"])
    X["S_Fe_ratio"] = df["S_pct"] / (df["Fe_pct"] + 1e-6)
    return X


def _model(seed=42):
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.85,
                         colsample_bytree=0.85, objective="multi:softprob",
                         eval_metric="mlogloss", random_state=seed, n_jobs=-1)


class GradeClassifier:
    def __init__(self, seed=42):
        self.seed = seed
        self.model = _model(seed)
        self.features = None

    def fit(self, df):
        X = engineer(df)
        y = df["grade"].map({c: i for i, c in enumerate(CLASSES)}).values
        self.features = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, df):
        return np.array(CLASSES)[self.model.predict(engineer(df)[self.features])]

    def predict_proba(self, df):
        return self.model.predict_proba(engineer(df)[self.features])

    def evaluate(self, df):
        y_true = df["grade"].values
        y_pred = self.predict(df)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
            "report": classification_report(y_true, y_pred, labels=CLASSES,
                                            output_dict=True, zero_division=0),
        }

    def shap_values(self, df):
        import shap
        X = engineer(df)[self.features]
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(X)
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[0] == len(CLASSES):   # (class, n, f)
            sv = np.transpose(sv, (1, 2, 0))
        return X, sv  # (n, features, classes)

    def global_importance(self, df):
        _, sv = self.shap_values(df)
        imp = np.abs(sv).mean(axis=(0, 2))
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def explain_sample(self, df_row, top=4):
        X, sv = self.shap_values(df_row)
        k = int(np.argmax(self.model.predict_proba(X)[0]))
        contrib = pd.Series(sv[0, :, k], index=self.features)
        top_c = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top]
        return CLASSES[k], top_c

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        obj = cls()
        obj.model, obj.features = d["model"], d["features"]
        return obj


def cross_validate(df, folds=5, seed=42):
    y = df["grade"].values
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(df, y):
        m = GradeClassifier(seed).fit(df.iloc[tr])
        scores.append(m.evaluate(df.iloc[te])["macro_f1"])
    return np.array(scores)


def train_and_test(df, test_size=0.25, seed=42):
    tr, te = train_test_split(df, test_size=test_size, stratify=df["grade"], random_state=seed)
    m = GradeClassifier(seed).fit(tr)
    return m, m.evaluate(te), tr, te
