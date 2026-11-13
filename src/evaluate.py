
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, POSITIVE_LABEL, TABLES_DIR
from src.train import MODELS_DIR, load_xy, split_train_test

CLASS_NAMES = ["Not CKD (0)", "CKD (1)"]



def load_model_and_test():
    joblib_path = MODELS_DIR / "pipeline.joblib"
    meta_path = MODELS_DIR / "metadata.json"
    if not joblib_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Chưa tìm thấy {joblib_path.name}/{meta_path.name}. "
            "Hãy chạy trước: python -X utf8 -m src.train"
        )
    pipeline = joblib.load(joblib_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    X, y = load_xy()
    _, X_test, _, y_test = split_train_test(X, y)

    if len(X_test) != metadata["n_test"]:
        raise ValueError(
            f"Số mẫu test ({len(X_test)}) không khớp metadata "
            f"({metadata['n_test']}) — CSV hoặc seed đã bị đổi sau khi train!"
        )
    return pipeline, metadata, X_test, y_test



def compute_metrics(y_true: pd.Series, y_pred: np.ndarray,
                    y_proba: np.ndarray) -> pd.DataFrame:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_ckd": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL,
                                         zero_division=0),
        "recall_ckd": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL,
                                   zero_division=0),
        "f1_ckd": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL,
                           zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return pd.DataFrame(
        {"metric": list(metrics), "value": [round(v, 4) for v in metrics.values()]}
    )



def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center",
                fontsize=16, color="white" if value > cm.max() / 2 else "black")

    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Ma trận nhầm lẫn (test, n={len(y_true)})")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.text(0, -0.45, f"TN={cm[0, 0]}  FP={cm[0, 1]}", ha="center", fontsize=9)
    ax.text(1, -0.45, f"FN={cm[1, 0]}  TP={cm[1, 1]}", ha="center", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"        → đã lưu {path}")


def plot_roc_curve(y_true, y_proba, auc_value: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=POSITIVE_LABEL)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ngẫu nhiên (0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Recall (True Positive Rate) — lớp CKD")
    ax.set_title("ROC curve (test)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"        → đã lưu {path}")



def export_predictions(X_test: pd.DataFrame, y_true, y_pred, y_proba,
                       path: Path) -> pd.DataFrame:
    table = pd.DataFrame({
        "id": X_test["id"].values if "id" in X_test.columns else X_test.index,
        "y_true": y_true.values,
        "y_pred": y_pred,
        "prob_ckd": np.round(y_proba, 4),
    })
    table["loại_lỗi"] = np.select(
        [(table.y_true == 1) & (table.y_pred == 0),
         (table.y_true == 0) & (table.y_pred == 1)],
        ["FN — bỏ sót CKD", "FP — báo nhầm"],
        default="đúng",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"        → đã lưu {path}")
    return table


def export_feature_importance(pipeline, path: Path) -> None | pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    feature_names = pipeline.named_steps["prepare"].get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        coef = np.abs(np.asarray(clf.coef_))
        importances = coef.mean(axis=0) if coef.ndim > 1 else coef
    else:
        print(f"        (mô hình {type(clf).__name__} không có "
              "feature_importances_/coef_ → bỏ qua bước này)")
        return None

    table = pd.DataFrame({"feature": feature_names, "importance": importances})
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table["importance"] = table["importance"].round(4)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"        → đã lưu {path} (top 5: "
          + ", ".join(table.feature.head(5)) + ")")
    return table



def main() -> None:
    print("=" * 68)
    print("BƯỚC 9 — Đánh giá mô hình trên tập test (MỘT LẦN, không train)")
    print("=" * 68)

    pipeline, metadata, X_test, y_test = load_model_and_test()
    print(f"  mô hình: {metadata['model']} (coding={metadata['coding']}, "
          f"sklearn {metadata['sklearn_version']})")
    print(f"  test   : {len(X_test)} mẫu — "
          f"ckd={int(y_test.sum())}, notckd={int((y_test == 0).sum())}")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n[9.1] Bảng chỉ số")
    metrics = compute_metrics(y_test, y_pred, y_proba)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))
    print(f"        → đã lưu {TABLES_DIR / 'test_metrics.csv'}")

    test_recall = float(metrics.loc[metrics.metric == "recall_ckd", "value"].iloc[0])
    print(f"\n        Điểm CV (train, 15 fold): "
          f"recall={metadata['cv_recall_ckd_mean']:.3f} "
          f"± {metadata['cv_recall_ckd_std']:.3f}")
    print(f"        Điểm test (80 mẫu)      : recall={test_recall:.3f}")
    print("        (80 mẫu → mỗi mẫu ≈ 1,25% — chênh CV/test trong khoảng "
          "±1 độ lệch chuẩn là bình thường)")

    print("\n[9.2] Biểu đồ")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"        TN={cm[0, 0]}  FP={cm[0, 1]}  FN={cm[1, 0]}  TP={cm[1, 1]}"
          "   (FN = bỏ sót ca — đọc kỹ ô này)")
    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, float(
        metrics.loc[metrics.metric == "roc_auc", "value"].iloc[0]),
        FIGURES_DIR / "roc_curve.png")

    print("\n[9.3] Phân tích lỗi")
    predictions = export_predictions(
        X_test, y_test, y_pred, y_proba, TABLES_DIR / "test_predictions.csv"
    )
    fn = predictions[predictions["loại_lỗi"] == "FN — bỏ sót CKD"]
    fp = predictions[predictions["loại_lỗi"] == "FP — báo nhầm"]
    print(f"        FN (bỏ sót CKD) = {len(fn)} mẫu"
          + (f" → id {fn['id'].tolist()}" if len(fn) else ""))
    print(f"        FP (báo nhầm)    = {len(fp)} mẫu"
          + (f" → id {fp['id'].tolist()}" if len(fp) else ""))

    print("\n[9.4] Tầm quan trọng thuộc tính")
    export_feature_importance(pipeline, TABLES_DIR / "feature_importance.csv")

    print("\n" + "=" * 68)
    print("HOÀN THÀNH BƯỚC 9")
    print("=" * 68)
    print("  Ghi vào báo cáo: đọc test_metrics.csv kèm cv_results.csv,")
    print("  nêu giới hạn (test 80 mẫu, EDA đã xem toàn tập) và phân tích FN.")


if __name__ == "__main__":
    main()
