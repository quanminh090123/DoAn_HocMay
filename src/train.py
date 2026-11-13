
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CATEGORY_VALUES,
    CODED_COLUMNS,
    NUMERIC_COLUMNS,
    POSITIVE_LABEL,
    PRIMARY_SCORING,
    RANDOM_STATE,
    RAW_DATA_PATH,
    ROOT,
    TABLES_DIR,
    TARGET,
)
from src.preprocess import clean_features, encode_target

MODELS_DIR = ROOT / "models"

SCORING = {
    "recall_ckd": make_scorer(recall_score, pos_label=POSITIVE_LABEL),
    "precision": make_scorer(precision_score, pos_label=POSITIVE_LABEL),
    "f1": make_scorer(f1_score, pos_label=POSITIVE_LABEL),
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}

CODING_OPTIONS = ("ordinal", "onehot")

SIMPLE_MODEL_ORDER = {"logreg": 0, "knn": 1, "dt": 2, "rf": 3}



def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(RAW_DATA_PATH)
    X = raw.drop(columns=[TARGET])
    y = encode_target(raw[TARGET])
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def make_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)



def make_preprocessor(coding: str = "ordinal") -> ColumnTransformer:
    if coding not in CODING_OPTIONS:
        raise ValueError(f"coding phải là một trong {CODING_OPTIONS}, nhận: {coding!r}")

    numeric = list(NUMERIC_COLUMNS)
    coded = list(CODED_COLUMNS)
    categorical = list(CATEGORY_VALUES)

    if coding == "ordinal":
        coded_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        coded_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("coded", coded_pipe, coded),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(classifier, coding: str = "ordinal") -> Pipeline:
    return Pipeline([
        ("clean", FunctionTransformer(clean_features, validate=False)),
        ("prepare", make_preprocessor(coding)),
        ("clf", classifier),
    ])


def baseline_models() -> dict:
    return {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logreg": lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "dt": lambda: DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "knn": lambda: KNeighborsClassifier(),
    }


PARAM_GRIDS = {
    "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
    "dt": {"clf__max_depth": [3, 5, 8, None]},
    "rf": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8]},
    "knn": {"clf__n_neighbors": [3, 5, 7, 11], "clf__weights": ["uniform", "distance"]},
}



def run_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    print("\n[8.1] Chạy baseline (5 mô hình, coding=ordinal, 15 fold)...")
    rows = []
    for name, factory in baseline_models().items():
        started = time.perf_counter()
        pipe = make_pipeline(factory(), coding="ordinal")
        scores = cross_validate(
            pipe, X_train, y_train,
            cv=make_cv(),
            scoring=SCORING,
            n_jobs=1,
            return_train_score=False,
        )
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(f"        - {name:7s} recall_ckd={row['recall_ckd_mean']:.3f}"
              f" ± {row['recall_ckd_std']:.3f}  ({row['seconds']}s)")

    table = pd.DataFrame(rows).sort_values("recall_ckd_mean", ascending=False)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "cv_baseline.csv", index=False, encoding="utf-8-sig")
    print(f"        → đã lưu {TABLES_DIR / 'cv_baseline.csv'}")
    return table


def run_grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    print("\n[8.2] GridSearchCV: 4 mô hình × 2 kiểu mã hóa sg/al/su ...")
    records = []
    for model_name, grid in PARAM_GRIDS.items():
        for coding in CODING_OPTIONS:
            started = time.perf_counter()
            pipe = make_pipeline(baseline_models()[model_name](), coding=coding)
            search = GridSearchCV(
                pipe,
                param_grid=grid,
                cv=make_cv(),
                scoring=SCORING,
                refit=False,
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            res = search.cv_results_

            best_idx = int(np.argmin(res["rank_test_recall_ckd"]))
            records.append({
                "model": model_name,
                "coding": coding,
                "recall_ckd_mean": res["mean_test_recall_ckd"][best_idx],
                "recall_ckd_std": res["std_test_recall_ckd"][best_idx],
                "precision_mean": res["mean_test_precision"][best_idx],
                "f1_mean": res["mean_test_f1"][best_idx],
                "roc_auc_mean": res["mean_test_roc_auc"][best_idx],
                "accuracy_mean": res["mean_test_accuracy"][best_idx],
                "best_params": json.dumps(res["params"][best_idx], ensure_ascii=False),
                "n_candidates": len(res["params"]),
                "seconds": round(time.perf_counter() - started, 1),
            })
            last = records[-1]
            print(f"        - {model_name:6s} | {coding:7s} | "
                  f"recall={last['recall_ckd_mean']:.3f} ± {last['recall_ckd_std']:.3f} "
                  f"| f1={last['f1_mean']:.3f} | auc={last['roc_auc_mean']:.3f} "
                  f"| {last['seconds']}s")

    table = pd.DataFrame(records)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "cv_results.csv", index=False, encoding="utf-8-sig")
    print(f"        → đã lưu {TABLES_DIR / 'cv_results.csv'}")
    return table


def choose_winner(results: pd.DataFrame) -> pd.Series:
    per_model_rows = []
    for model_name, group in results.groupby("model"):
        group = group.sort_values("recall_ckd_mean", ascending=False)
        top = group.iloc[0]
        if len(group) > 1:
            runner_up = group.iloc[1]
            gap = top["recall_ckd_mean"] - runner_up["recall_ckd_mean"]
            if gap < top["recall_ckd_std"] and (group["coding"] == "ordinal").any():
                top = group.loc[group["coding"] == "ordinal"].iloc[0]
        per_model_rows.append(top)
    per_model = pd.DataFrame(per_model_rows).reset_index(drop=True)

    best_row = per_model.loc[per_model["recall_ckd_mean"].idxmax()]
    threshold = best_row["recall_ckd_mean"] - best_row["recall_ckd_std"]
    candidates = per_model[per_model["recall_ckd_mean"] >= threshold].copy()
    candidates["_simplicity"] = candidates["model"].map(SIMPLE_MODEL_ORDER)
    winner = candidates.sort_values(
        ["_simplicity", "recall_ckd_mean"], ascending=[True, False]
    ).iloc[0]

    print("\n[8.3] Chọn mô hình theo quy tắc: recall cao nhất, "
          "chênh < 1 độ lệch chuẩn thì ưu tiên đơn giản:")
    print(per_model[["model", "coding", "recall_ckd_mean", "recall_ckd_std",
                     "f1_mean", "roc_auc_mean", "best_params"]]
          .to_string(index=False))
    print(f"        → THẮNG: {winner['model']} (coding={winner['coding']}, "
          f"recall={winner['recall_ckd_mean']:.3f} ± {winner['recall_ckd_std']:.3f})")
    return winner


def fit_winner_on_full_train(X_train: pd.DataFrame, y_train: pd.Series,
                             winner: pd.Series):
    print("\n[8.4] Fit model cuối trên toàn bộ tập train ...")
    pipeline = make_pipeline(
        baseline_models()[winner["model"]](), coding=winner["coding"]
    )
    search = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRIDS[winner["model"]],
        cv=make_cv(),
        scoring=SCORING,
        refit="recall_ckd",
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    print(f"        siêu tham số thắng: {json.dumps(search.best_params_, ensure_ascii=False)}")
    return search.best_estimator_, search


def save_artifacts(pipeline, winner: pd.Series, search: GridSearchCV,
                   n_train: int, n_test: int) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib_path = MODELS_DIR / "pipeline.joblib"
    joblib.dump(pipeline, joblib_path)

    best_idx = int(np.argmin(search.cv_results_["rank_test_recall_ckd"]))
    cvres = search.cv_results_
    metadata = {
        "model": winner["model"],
        "coding": winner["coding"],
        "best_params": json.loads(winner["best_params"]),
        "cv_recall_ckd_mean": round(float(winner["recall_ckd_mean"]), 4),
        "cv_recall_ckd_std": round(float(winner["recall_ckd_std"]), 4),
        "cv_f1_mean": round(float(winner["f1_mean"]), 4),
        "cv_roc_auc_mean": round(float(winner["roc_auc_mean"]), 4),
        "cv_precision_mean": round(float(winner["precision_mean"]), 4),
        "cv_detail": {
            "recall_ckd": round(float(cvres["mean_test_recall_ckd"][best_idx]), 4),
            "f1": round(float(cvres["mean_test_f1"][best_idx]), 4),
            "roc_auc": round(float(cvres["mean_test_roc_auc"][best_idx]), 4),
        },
        "scoring_primary": PRIMARY_SCORING,
        "positive_label": POSITIVE_LABEL,
        "random_state": RANDOM_STATE,
        "n_train": n_train,
        "n_test": n_test,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "trained_on": "train only (test kept untouched for evaluate.py)",
    }
    meta_path = MODELS_DIR / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n        → đã lưu {joblib_path}")
    print(f"        → đã lưu {meta_path}")



def main() -> None:
    total_started = time.perf_counter()

    print("=" * 68)
    print("BƯỚC 6 — Nạp dữ liệu và chia train/test (80/20, stratify)")
    print("=" * 68)
    X, y = load_xy()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"  tổng mẫu      : {len(X)}  (ckd={int(y.sum())}, notckd={int((y == 0).sum())})")
    print(f"  train         : {len(X_train)}  (ckd={int(y_train.sum())}, "
          f"notckd={int((y_train == 0).sum())})")
    print(f"  test (giữ lại): {len(X_test)}  (ckd={int(y_test.sum())}, "
          f"notckd={int((y_test == 0).sum())})  ← evaluate.py sẽ dùng")
    print(f"  {PRIMARY_SCORING=}, {POSITIVE_LABEL=}, {RANDOM_STATE=}")

    print("\n" + "=" * 68)
    print("BƯỚC 7–8 — Huấn luyện và chọn mô hình CHỈ trên tập train")
    print("=" * 68)
    baseline = run_baseline(X_train, y_train)

    results = run_grid_search(X_train, y_train)
    winner = choose_winner(results)

    pipeline, search = fit_winner_on_full_train(X_train, y_train, winner)
    save_artifacts(pipeline, winner, search, n_train=len(X_train), n_test=len(X_test))

    print("\n" + "=" * 68)
    print("HOÀN THÀNH BƯỚC 6–8")
    print("=" * 68)
    print(baseline[["model", "recall_ckd_mean", "recall_ckd_std", "f1_mean", "roc_auc_mean"]]
          .to_string(index=False))
    print(f"\n  Mô hình được chọn: {winner['model']} (coding={winner['coding']})")
    print("  Tiếp theo:  python -X utf8 -m src.evaluate  "
          "(chấm điểm tập test đúng 1 lần)")
    print(f"  Tổng thời gian: {time.perf_counter() - total_started:.1f}s")


if __name__ == "__main__":
    main()
