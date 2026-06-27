"""
Full working EmoPiano Music Emotion Recognition pipeline
=========================================================
No hard-coded results are used. All metrics, tables, plots and saved models are
computed from the dataset at runtime.

Workflow:
1) Download/load EmoPiano dataset
2) Discover audio files and labels from metadata CSV or folder names
3) Extract MFCC + Chroma features with librosa
4) Impute + Z-score normalize + PCA retain 95% variance
5) Tune SVM/RF/XGBoost hyperparameters using a practical EPO-style optimizer
6) Train Hybrid Voting Ensemble Classifier
7) Optional SVR valence-arousal regression when metadata has valence/arousal columns
8) Save metrics, confusion matrix, feature importance, predictions, plots, and models

Run:
    python emopiano_epo_hvec_pipeline.py --download

Or with a local dataset folder:
    python emopiano_epo_hvec_pipeline.py --data_dir "path/to/EmoPiano"

Optional faster smoke test:
    python emopiano_epo_hvec_pipeline.py --download --max_files 300 --epo_iter 5 --epo_pop 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Matplotlib setup requested commonly for manuscript figures
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 7)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGBOOST = False

try:
    import librosa
except Exception as exc:
    raise ImportError(
        "librosa is required for audio feature extraction. Install with: pip install librosa soundfile"
    ) from exc

KAGGLE_SLUG = "ziya07/emopiano-dataset-for-emotion-recognition-in-piano"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}
EMOTION_WORDS = ["happy", "sad", "angry", "relaxed"]


@dataclass
class Config:
    data_dir: Optional[Path]
    output_dir: Path
    download: bool
    sample_rate: int
    n_mfcc: int
    n_chroma: int
    duration: Optional[float]
    test_size: float
    val_size: float
    random_state: int
    pca_variance: float
    epo_pop: int
    epo_iter: int
    cv_folds: int
    max_files: Optional[int]
    use_cache: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_emopiano(output_dir: Path) -> Path:
    """Download dataset using kagglehub. Requires kagglehub and Kaggle access if the dataset is gated."""
    try:
        import kagglehub
    except Exception as exc:
        raise ImportError(
            "kagglehub is not installed. Install it using: pip install kagglehub\n"
            "Then rerun with --download, or pass --data_dir to an already downloaded dataset."
        ) from exc

    print(f"Downloading Kaggle dataset: {KAGGLE_SLUG}")
    dataset_path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    target = output_dir / "dataset" / "emopiano"
    safe_mkdir(target.parent)
    if target.exists():
        return target
    try:
        shutil.copytree(dataset_path, target)
        return target
    except Exception:
        # If copy fails due to existing files or links, use original cache path.
        return dataset_path


def normalize_colname(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_metadata_csv(data_dir: Path) -> Optional[Path]:
    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        return None

    preferred_keywords = ["metadata", "label", "annotation", "emopiano", "data"]
    for csv_path in csv_files:
        lname = csv_path.name.lower()
        if any(k in lname for k in preferred_keywords):
            return csv_path
    return csv_files[0]


def infer_label_from_path(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    stem = path.stem.lower()
    search_text = " ".join(parts + [stem])
    for emotion in EMOTION_WORDS:
        if emotion in search_text:
            return emotion
    return None


def resolve_audio_path(row: pd.Series, data_dir: Path, path_col: str) -> Optional[Path]:
    raw = str(row[path_col]).strip()
    candidates = [data_dir / raw, data_dir / Path(raw).name]
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in AUDIO_EXTENSIONS:
            return candidate

    # Search by basename if metadata stores only filename.
    basename = Path(raw).name
    matches = list(data_dir.rglob(basename))
    for match in matches:
        if match.suffix.lower() in AUDIO_EXTENSIONS:
            return match
    return None


def discover_dataset(data_dir: Path, max_files: Optional[int] = None) -> pd.DataFrame:
    """Return DataFrame with audio_path, label and optional valence/arousal columns."""
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    metadata_csv = find_metadata_csv(data_dir)
    records: List[Dict[str, object]] = []

    if metadata_csv is not None:
        print(f"Found metadata CSV: {metadata_csv}")
        meta = pd.read_csv(metadata_csv)
        original_cols = list(meta.columns)
        meta.columns = [normalize_colname(c) for c in meta.columns]

        path_candidates = [c for c in meta.columns if any(k in c for k in ["path", "file", "filename", "audio", "name"])]
        label_candidates = [c for c in meta.columns if any(k in c for k in ["emotion", "label", "class", "mood"])]
        val_candidates = [c for c in meta.columns if "valence" in c or c in {"v", "val"}]
        aro_candidates = [c for c in meta.columns if "arousal" in c or c in {"a", "aro"}]

        if path_candidates and label_candidates:
            path_col = path_candidates[0]
            label_col = label_candidates[0]
            val_col = val_candidates[0] if val_candidates else None
            aro_col = aro_candidates[0] if aro_candidates else None

            for _, row in meta.iterrows():
                audio_path = resolve_audio_path(row, data_dir, path_col)
                if audio_path is None:
                    continue
                label = str(row[label_col]).strip().lower()
                if not label or label == "nan":
                    label = infer_label_from_path(audio_path) or "unknown"
                item = {"audio_path": str(audio_path), "label": label}
                if val_col is not None:
                    item["valence"] = pd.to_numeric(row[val_col], errors="coerce")
                if aro_col is not None:
                    item["arousal"] = pd.to_numeric(row[aro_col], errors="coerce")
                records.append(item)
        else:
            print(
                f"Metadata CSV columns were not enough for path/label detection. Columns: {original_cols}. "
                "Falling back to folder/filename label inference."
            )

    if not records:
        audio_files = [p for p in data_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
        for p in audio_files:
            label = infer_label_from_path(p)
            if label is not None:
                records.append({"audio_path": str(p), "label": label})

    if not records:
        raise RuntimeError(
            "No labeled audio files found. Expected metadata CSV with file path + label columns, "
            "or folders/filenames containing emotion names: happy, sad, angry, relaxed."
        )

    df = pd.DataFrame(records).drop_duplicates(subset=["audio_path"]).reset_index(drop=True)
    df = df[df["label"].astype(str).str.lower() != "unknown"].reset_index(drop=True)

    if max_files is not None and len(df) > max_files:
        # Stratified subsample for fast testing without class imbalance distortion.
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(max(1, int(max_files * len(x) / len(df))), random_state=42))
            .reset_index(drop=True)
        )
        if len(df) > max_files:
            df = df.sample(max_files, random_state=42).reset_index(drop=True)

    print("Dataset discovered:")
    print(df["label"].value_counts())
    return df


def extract_features_one(
    audio_path: str,
    sample_rate: int,
    n_mfcc: int,
    n_chroma: int,
    duration: Optional[float] = None,
) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError(f"Empty audio signal: {audio_path}")

    # Trim silence and normalize amplitude safely.
    y, _ = librosa.effects.trim(y, top_db=30)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma, n_fft=2048, hop_length=512)

    def summarize(feature_matrix: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.mean(feature_matrix, axis=1),
                np.std(feature_matrix, axis=1),
                np.min(feature_matrix, axis=1),
                np.max(feature_matrix, axis=1),
            ]
        )

    return np.concatenate([summarize(mfcc), summarize(chroma)]).astype(np.float32)


def build_feature_matrix(df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, List[str]]:
    cache_path = cfg.output_dir / "features_cache.npz"
    if cfg.use_cache and cache_path.exists():
        print(f"Loading feature cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], list(data["feature_names"])

    feature_rows: List[np.ndarray] = []
    valid_indices: List[int] = []
    errors = []
    total = len(df)
    for idx, row in df.iterrows():
        if idx % 25 == 0:
            print(f"Extracting features: {idx + 1}/{total}")
        try:
            feat = extract_features_one(
                row["audio_path"], cfg.sample_rate, cfg.n_mfcc, cfg.n_chroma, cfg.duration
            )
            feature_rows.append(feat)
            valid_indices.append(idx)
        except Exception as exc:
            errors.append((row["audio_path"], str(exc)))

    if not feature_rows:
        raise RuntimeError("Feature extraction failed for all files.")

    if errors:
        pd.DataFrame(errors, columns=["audio_path", "error"]).to_csv(cfg.output_dir / "feature_extraction_errors.csv", index=False)
        print(f"Warning: {len(errors)} files failed. See feature_extraction_errors.csv")

    X = np.vstack(feature_rows)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    df_valid.to_csv(cfg.output_dir / "dataset_index_used.csv", index=False)

    names = []
    for prefix, count in [("mfcc", cfg.n_mfcc), ("chroma", cfg.n_chroma)]:
        for stat in ["mean", "std", "min", "max"]:
            for i in range(count):
                names.append(f"{prefix}_{i + 1}_{stat}")

    np.savez_compressed(cache_path, X=X, feature_names=np.array(names, dtype=object))
    return X, names


def split_train_val_test(
    X: np.ndarray, y: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    relative_val = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=cfg.random_state,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def specificity_gmean_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    recalls = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specs.append(specificity)
        recalls.append(recall)
    return float(np.mean(specs)), float(np.sqrt(np.mean(recalls) * np.mean(specs))), float(matthews_corrcoef(y_true, y_pred))


def make_xgb(params: Dict[str, object], random_state: int, n_classes: int):
    if HAS_XGBOOST:
        return XGBClassifier(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
    # fallback when xgboost is not installed
    return HistGradientBoostingClassifier(
        max_iter=int(params.get("n_estimators", 100)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_leaf_nodes=31,
        max_depth=int(params.get("max_depth", 6)),
        random_state=random_state,
    )


def build_pipeline(params: Dict[str, object], cfg: Config, n_classes: int) -> Pipeline:
    svm = SVC(
        C=float(params["svm_C"]),
        gamma=float(params["svm_gamma"]),
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=cfg.random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=int(params["rf_n_estimators"]),
        max_depth=int(params["rf_max_depth"]),
        min_samples_split=int(params["rf_min_samples_split"]),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    xgb_params = {
        "n_estimators": int(params["xgb_n_estimators"]),
        "max_depth": int(params["xgb_max_depth"]),
        "learning_rate": float(params["xgb_learning_rate"]),
        "subsample": float(params["xgb_subsample"]),
        "colsample_bytree": float(params["xgb_colsample"]),
    }
    xgb = make_xgb(xgb_params, cfg.random_state, n_classes)

    voting = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf), ("xgb", xgb)],
        voting="soft",
        n_jobs=None,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=cfg.pca_variance, random_state=cfg.random_state)),
            ("model", voting),
        ]
    )


BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "svm_C": (0.1, 80.0, "float"),
    "svm_gamma": (0.0001, 1.0, "log_float"),
    "rf_n_estimators": (80, 350, "int"),
    "rf_max_depth": (3, 30, "int"),
    "rf_min_samples_split": (2, 10, "int"),
    "xgb_n_estimators": (80, 350, "int"),
    "xgb_max_depth": (2, 10, "int"),
    "xgb_learning_rate": (0.01, 0.25, "float"),
    "xgb_subsample": (0.65, 1.0, "float"),
    "xgb_colsample": (0.65, 1.0, "float"),
}


def sample_param(rng: np.random.Generator, low: float, high: float, kind: str):
    if kind == "int":
        return int(rng.integers(int(low), int(high) + 1))
    if kind == "log_float":
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))
    return float(rng.uniform(low, high))


def vector_to_params(vector: np.ndarray) -> Dict[str, object]:
    params = {}
    for value, (name, (low, high, kind)) in zip(vector, BOUNDS.items()):
        clipped = float(np.clip(value, low, high))
        if kind == "int":
            params[name] = int(round(clipped))
        else:
            params[name] = clipped
    return params


def random_vector(rng: np.random.Generator) -> np.ndarray:
    values = []
    for _, (low, high, kind) in BOUNDS.items():
        values.append(sample_param(rng, low, high, kind))
    return np.array(values, dtype=float)


def clamp_vector(vector: np.ndarray) -> np.ndarray:
    out = vector.copy()
    for i, (_, (low, high, _)) in enumerate(BOUNDS.items()):
        out[i] = np.clip(out[i], low, high)
    return out


def evaluate_params(
    params: Dict[str, object], X_train: np.ndarray, y_train: np.ndarray, cfg: Config, n_classes: int
) -> float:
    """Fitness = mean macro-F1 over stratified CV."""
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        pipe = build_pipeline(params, cfg, n_classes)
        pipe.fit(X_train[tr_idx], y_train[tr_idx])
        pred = pipe.predict(X_train[va_idx])
        scores.append(f1_score(y_train[va_idx], pred, average="macro", zero_division=0))
    return float(np.mean(scores))


def epo_optimize(
    X_train: np.ndarray, y_train: np.ndarray, cfg: Config, n_classes: int
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Practical Emperor Penguin Optimizer-style hyperparameter search.

    This implementation uses a population of candidate hyperparameter vectors. During each iteration,
    candidates move toward the current best vector with stochastic exploration and boundary control.
    """
    rng = np.random.default_rng(cfg.random_state)
    population = np.vstack([random_vector(rng) for _ in range(cfg.epo_pop)])
    fitness = np.full(cfg.epo_pop, -np.inf)
    history = []

    best_vector = None
    best_score = -np.inf

    for iteration in range(1, cfg.epo_iter + 1):
        print(f"EPO iteration {iteration}/{cfg.epo_iter}")
        for i in range(cfg.epo_pop):
            params = vector_to_params(population[i])
            score = evaluate_params(params, X_train, y_train, cfg, n_classes)
            fitness[i] = score
            if score > best_score:
                best_score = score
                best_vector = population[i].copy()
                print(f"  New best macro-F1={best_score:.5f}: {vector_to_params(best_vector)}")

        history.append({"iteration": iteration, "best_macro_f1": best_score})

        # EPO-inspired movement: early broad exploration, later local exploitation.
        a = 2.0 * (1.0 - iteration / max(1, cfg.epo_iter))
        for i in range(cfg.epo_pop):
            r1 = rng.random(len(BOUNDS))
            r2 = rng.random(len(BOUNDS))
            distance = np.abs(best_vector - population[i])
            direction = np.sign(best_vector - population[i])
            exploration_noise = rng.normal(0, 1, len(BOUNDS)) * a * 0.05 * np.maximum(np.abs(best_vector), 1.0)
            population[i] = population[i] + r1 * direction * distance + r2 * exploration_noise
            # occasional random restart for diversity
            if rng.random() < 0.08 and iteration < cfg.epo_iter:
                population[i] = random_vector(rng)
            population[i] = clamp_vector(population[i])

    return vector_to_params(best_vector), pd.DataFrame(history)


def evaluate_classifier(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    specificity, gmean, mcc = specificity_gmean_mcc(y_test, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "specificity_macro": specificity,
        "gmean_macro": gmean,
        "mcc": mcc,
    }

    try:
        y_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        metrics["roc_auc_ovr_macro"] = np.nan

    pd.DataFrame([metrics]).to_csv(output_dir / "classification_metrics.csv", index=False)
    report = classification_report(y_test, y_pred, target_names=list(class_names), output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(output_dir / "test_predictions_encoded.csv", index=False)

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - EPO-HVEC")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=0)
    plt.yticks(np.arange(len(class_names)), class_names)
    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=18)
    plt.colorbar()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=800, bbox_inches="tight")
    plt.close()

    return metrics


def save_feature_importance(model: Pipeline, feature_names: List[str], output_dir: Path) -> None:
    try:
        pca = model.named_steps["pca"]
        voting = model.named_steps["model"]
        rf_model = voting.named_estimators_["rf"]
        if not hasattr(rf_model, "feature_importances_"):
            return
        # RF importances are in PCA component space. Map them approximately to original features.
        pca_importance = np.asarray(rf_model.feature_importances_)
        original_importance = np.abs(pca.components_).T @ pca_importance
        original_importance = original_importance / original_importance.sum()
        fi = pd.DataFrame({"feature": feature_names, "importance": original_importance})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi.to_csv(output_dir / "feature_importance_rf_pca_mapped.csv", index=False)

        top = fi.head(20).iloc[::-1]
        plt.figure()
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Mapped Importance")
        plt.ylabel("Feature")
        plt.title("Top 20 RF Feature Importances")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance_top20.png", dpi=800, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Feature importance skipped: {exc}")


def pca_variance_plot(model: Pipeline, output_dir: Path) -> None:
    pca: PCA = model.named_steps["pca"]
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pd.DataFrame({"component": np.arange(1, len(evr) + 1), "explained_variance": evr, "cumulative_variance": cum}).to_csv(
        output_dir / "pca_variance.csv", index=False
    )
    plt.figure()
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_cumulative_variance.png", dpi=800, bbox_inches="tight")
    plt.close()


def run_svr_regression(
    X: np.ndarray,
    df_used: pd.DataFrame,
    cfg: Config,
    fitted_preprocessor: Pipeline,
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """Run valence/arousal regression only if real labels are present in metadata."""
    if not {"valence", "arousal"}.issubset(df_used.columns):
        print("Valence/arousal columns not found in dataset metadata. SVR regression skipped.")
        return None

    targets = df_used[["valence", "arousal"]].apply(pd.to_numeric, errors="coerce")
    mask = targets.notna().all(axis=1).values
    if mask.sum() < 20:
        print("Not enough valid valence/arousal samples. SVR regression skipped.")
        return None

    X_reg = X[mask]
    y_reg = targets.loc[mask].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Fit preprocessing independently for regression to avoid leakage.
    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=cfg.pca_variance, random_state=cfg.random_state)),
        ]
    )
    X_train_p = preprocess.fit_transform(X_train)
    X_test_p = preprocess.transform(X_test)

    rows = []
    preds = {}
    for idx, target_name in enumerate(["valence", "arousal"]):
        svr = SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale")
        svr.fit(X_train_p, y_train[:, idx])
        pred = svr.predict(X_test_p)
        preds[target_name] = pred
        rows.append(
            {
                "target": target_name,
                "MAE": mean_absolute_error(y_test[:, idx], pred),
                "MSE": mean_squared_error(y_test[:, idx], pred),
                "RMSE": math.sqrt(mean_squared_error(y_test[:, idx], pred)),
                "R2": r2_score(y_test[:, idx], pred),
            }
        )
        joblib.dump(svr, output_dir / f"svr_{target_name}.joblib")

        plt.figure()
        plt.scatter(y_test[:, idx], pred, alpha=0.7)
        low = min(np.min(y_test[:, idx]), np.min(pred))
        high = max(np.max(y_test[:, idx]), np.max(pred))
        plt.plot([low, high], [low, high], linestyle="--")
        plt.xlabel(f"Actual {target_name.title()}")
        plt.ylabel(f"Predicted {target_name.title()}")
        plt.title(f"SVR {target_name.title()} Prediction")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(output_dir / f"svr_{target_name}_actual_vs_predicted.png", dpi=800, bbox_inches="tight")
        plt.close()

    joblib.dump(preprocess, output_dir / "svr_preprocessor.joblib")
    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(output_dir / "svr_regression_metrics.csv", index=False)
    return reg_df


def make_excel_summary(output_dir: Path) -> None:
    excel_path = output_dir / "all_results_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for csv_name, sheet_name in [
            ("dataset_index_used.csv", "Dataset_Index"),
            ("classification_metrics.csv", "Classification_Metrics"),
            ("classification_report.csv", "Class_Report"),
            ("confusion_matrix.csv", "Confusion_Matrix"),
            ("pca_variance.csv", "PCA_Variance"),
            ("epo_history.csv", "EPO_History"),
            ("feature_importance_rf_pca_mapped.csv", "Feature_Importance"),
            ("svr_regression_metrics.csv", "SVR_Regression"),
        ]:
            path = output_dir / csv_name
            if path.exists():
                pd.read_csv(path).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f"Excel summary saved: {excel_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EmoPiano EPO-HVEC full pipeline without hard-coded results")
    parser.add_argument("--data_dir", type=Path, default=None, help="Path to already downloaded EmoPiano dataset")
    parser.add_argument("--download", action="store_true", help="Download EmoPiano from Kaggle using kagglehub")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_emopiano_epo_hvec"))
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_mfcc", type=int, default=13)
    parser.add_argument("--n_chroma", type=int, default=12)
    parser.add_argument("--duration", type=float, default=None, help="Optional seconds per audio file; use for faster runs")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--pca_variance", type=float, default=0.95)
    parser.add_argument("--epo_pop", type=int, default=8)
    parser.add_argument("--epo_iter", type=int, default=8)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap for debugging only")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        download=args.download,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_chroma=args.n_chroma,
        duration=args.duration,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        pca_variance=args.pca_variance,
        epo_pop=args.epo_pop,
        epo_iter=args.epo_iter,
        cv_folds=args.cv_folds,
        max_files=args.max_files,
        use_cache=not args.no_cache,
    )

    seed_everything(cfg.random_state)
    safe_mkdir(cfg.output_dir)

    if cfg.download:
        data_dir = download_emopiano(cfg.output_dir)
    elif cfg.data_dir is not None:
        data_dir = cfg.data_dir
    else:
        raise ValueError("Use --download or provide --data_dir")

    start = time.time()
    df = discover_dataset(data_dir, cfg.max_files)
    X, feature_names = build_feature_matrix(df, cfg)

    # Re-read index used because failed files may be removed during feature extraction.
    df_used = pd.read_csv(cfg.output_dir / "dataset_index_used.csv")
    y_text = df_used["label"].astype(str).str.lower().values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = label_encoder.classes_
    pd.DataFrame({"class_id": np.arange(len(class_names)), "class_name": class_names}).to_csv(
        cfg.output_dir / "label_mapping.csv", index=False
    )
    joblib.dump(label_encoder, cfg.output_dir / "label_encoder.joblib")

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, cfg)
    # EPO sees train + validation through CV; test remains untouched.
    X_tune = np.vstack([X_train, X_val])
    y_tune = np.concatenate([y_train, y_val])

    best_params, epo_history = epo_optimize(X_tune, y_tune, cfg, len(class_names))
    epo_history.to_csv(cfg.output_dir / "epo_history.csv", index=False)
    with open(cfg.output_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("Training final EPO-HVEC model on train+validation set...")
    final_model = build_pipeline(best_params, cfg, len(class_names))
    final_model.fit(X_tune, y_tune)
    joblib.dump(final_model, cfg.output_dir / "epo_hvec_model.joblib")

    metrics = evaluate_classifier(final_model, X_test, y_test, class_names, cfg.output_dir)
    pca_variance_plot(final_model, cfg.output_dir)
    save_feature_importance(final_model, feature_names, cfg.output_dir)
    run_svr_regression(X, df_used, cfg, final_model, cfg.output_dir)
    make_excel_summary(cfg.output_dir)

    runtime = time.time() - start
    run_info = {
        "dataset_dir": str(data_dir),
        "n_samples_used": int(len(df_used)),
        "n_features_raw": int(X.shape[1]),
        "classes": list(class_names),
        "runtime_seconds": runtime,
        "classification_metrics": metrics,
        "best_hyperparameters": best_params,
        "xgboost_available": HAS_XGBOOST,
    }
    with open(cfg.output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("\nDONE ✅")
    print(f"Outputs saved in: {cfg.output_dir.resolve()}")
    print(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
