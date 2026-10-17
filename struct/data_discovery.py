import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .report_labeler import audit_report_labeler_on_gold, parse_clinical_report

META_FEATURE_COLS = ["meta_fluid", "meta_fat", "meta_plane_sag", "meta_plane_cor", "meta_plane_ax"]
WEIGHT_COLS = [f"weight_{c}" for c in [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]]


@dataclass
class DiscoveredDataset:
    comp_dir: str
    train_gold_studies_df: pd.DataFrame
    train_weak_studies_df: pd.DataFrame
    val_gold_studies_df: pd.DataFrame
    series_df: pd.DataFrame
    test_studies_df: Optional[pd.DataFrame]
    test_series_df: Optional[pd.DataFrame]
    npz_map: Dict[str, str]
    has_valid_metadata: bool
    metadata_cols: List[str]
    pos_weights: np.ndarray
    priors: np.ndarray
    diagnostics_report: Dict[str, Any]


def _normalize_id(val: Any) -> str:
    s = str(val).strip()
    if s.endswith(".npz") or s.endswith(".npy"):
        s = s[:-4]
    return s


def _parse_binary_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(np.float32)
    s_clean = series.astype(str).str.strip().str.upper()
    mapping = {
        "1": 1.0, "1.0": 1.0, "TRUE": 1.0, "YES": 1.0, "POS": 1.0, "POSITIVE": 1.0,
        "0": 0.0, "0.0": 0.0, "FALSE": 0.0, "NO": 0.0, "NEG": 0.0, "NEGATIVE": 0.0,
    }
    return s_clean.map(mapping).combine_first(pd.to_numeric(series, errors="coerce")).astype(np.float32)


def locate_competition_files(input_root: str) -> Tuple[str, Dict[str, str], List[str]]:
    comp_dir = None
    npz_map: Dict[str, str] = {}
    all_npz_list: List[str] = []

    for root, _, files in os.walk(input_root):
        if "train.csv" in files and "train_series.csv" in files and comp_dir is None:
            comp_dir = root
        for f in files:
            if f.endswith(".npz"):
                full_path = os.path.join(root, f)
                all_npz_list.append(full_path)
                stem = f[:-4].strip()
                npz_map[stem] = full_path
                parent = os.path.basename(root).strip()
                npz_map[f"{parent}_{stem}"] = full_path
                npz_map[f"{parent}/{stem}"] = full_path

    if comp_dir is None:
        raise FileNotFoundError(f"Could not find train.csv and train_series.csv under {input_root}")

    return comp_dir, npz_map, all_npz_list


def extract_and_validate_metadata(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, List[str]]:
    has_valid_meta = False
    if "Fluid_Sensitive" in df.columns:
        parsed = _parse_binary_series(df["Fluid_Sensitive"])
        if parsed.notna().sum() > 0:
            has_valid_meta = True
        df["meta_fluid"] = parsed.fillna(0.0).astype(np.float32)
    else:
        df["meta_fluid"] = np.float32(0.0)

    if "Fat_Suppression" in df.columns:
        parsed = _parse_binary_series(df["Fat_Suppression"])
        if parsed.notna().sum() > 0:
            has_valid_meta = True
        df["meta_fat"] = parsed.fillna(0.0).astype(np.float32)
    else:
        df["meta_fat"] = np.float32(0.0)

    if "Anatomical_Plane" in df.columns:
        plane_str = df["Anatomical_Plane"].astype(str).str.upper().str.strip()
        df["meta_plane_sag"] = plane_str.str.startswith("SAG").astype(np.float32)
        df["meta_plane_cor"] = plane_str.str.startswith("COR").astype(np.float32)
        df["meta_plane_ax"] = (plane_str.str.startswith("AX") | plane_str.str.startswith("TRA")).astype(np.float32)
        has_valid_meta = True
    else:
        df["meta_plane_sag"] = np.float32(0.0)
        df["meta_plane_cor"] = np.float32(0.0)
        df["meta_plane_ax"] = np.float32(0.0)

    return df, has_valid_meta, list(META_FEATURE_COLS)


def create_gold_study_split(
    gold_studies_df: pd.DataFrame, labels: List[str], num_folds: int = 5, val_fold: int = 0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    study_labels_clean = gold_studies_df[labels].fillna(0.0)
    pos_freq = study_labels_clean.sum(axis=0).values
    rarity_order = np.argsort(pos_freq)

    primary_bucket = np.zeros(len(gold_studies_df), dtype=np.int64)
    for rank_idx, label_idx in enumerate(rarity_order):
        col = labels[label_idx]
        mask = (study_labels_clean[col].values > 0.5) & (primary_bucket == 0)
        primary_bucket[mask] = rank_idx + 1

    burden = study_labels_clean.sum(axis=1).astype(int).values
    strata_key = primary_bucket * 100 + burden

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(gold_studies_df))
    order = np.argsort(strata_key[perm], kind="stable")
    folds_perm = np.zeros(len(gold_studies_df), dtype=np.int64)
    folds_perm[order] = np.arange(len(gold_studies_df)) % num_folds

    folds = np.zeros(len(gold_studies_df), dtype=np.int64)
    folds[perm] = folds_perm

    is_val = folds == val_fold
    is_train = ~is_val
    return is_train, is_val


def discover_and_prepare_data(cfg: PipelineConfig, rank: int = 0) -> DiscoveredDataset:
    comp_dir, npz_map, all_npz_list = locate_competition_files(cfg.input_root)

    patient_df = pd.read_csv(os.path.join(comp_dir, "train.csv"))
    series_df = pd.read_csv(os.path.join(comp_dir, "train_series.csv"))

    patient_df["StudyInstanceUID"] = patient_df["StudyInstanceUID"].map(_normalize_id)
    series_df["StudyInstanceUID"] = series_df["StudyInstanceUID"].map(_normalize_id)
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].map(_normalize_id)

    # Resolve .npz paths for all 24,371 series
    s_uids = series_df["StudyInstanceUID"].tolist()
    ser_uids = series_df["SeriesInstanceUID"].tolist()
    series_df["npz_path"] = [
        npz_map.get(ser) or npz_map.get(f"{st}_{ser}") or npz_map.get(st)
        for st, ser in zip(s_uids, ser_uids)
    ]
    series_df = series_df[series_df["npz_path"].notna()].copy().reset_index(drop=True)
    series_df, has_valid_meta, meta_cols = extract_and_validate_metadata(series_df)

    valid_study_set = set(series_df["StudyInstanceUID"].unique())
    patient_df = patient_df[patient_df["StudyInstanceUID"].isin(valid_study_set)].copy().reset_index(drop=True)

    for col in cfg.labels:
        patient_df[col] = _parse_binary_series(patient_df[col])

    # Separate the 58 Gold-Standard studies from the 4,349 Report-Only studies
    is_gold = patient_df[cfg.labels].notna().any(axis=1).values
    gold_df = patient_df.loc[is_gold].copy().reset_index(drop=True)
    weak_raw_df = patient_df.loc[~is_gold].copy().reset_index(drop=True)

    # Gold studies get confidence weight = 1.0 on all non-NaN labels
    for col, w_col in zip(cfg.labels, WEIGHT_COLS):
        gold_df[w_col] = gold_df[col].notna().astype(np.float32)
    gold_df["is_gold"] = 1

    # Split the 58 Gold studies into 46 Train Gold and 12 Held-out Val Gold
    g_train_mask, g_val_mask = create_gold_study_split(
        gold_df, cfg.labels, num_folds=cfg.num_folds, val_fold=cfg.fold, seed=cfg.seed
    )
    train_gold_df = gold_df.loc[g_train_mask].reset_index(drop=True)
    val_gold_df = gold_df.loc[g_val_mask].reset_index(drop=True)

    # Parse bilingual clinical reports for the 4,349 unlabeled studies
    weak_studies_df = pd.DataFrame()
    parser_audit = {}
    if cfg.use_report_supervision and "Report" in weak_raw_df.columns and len(weak_raw_df) > 0:
        parser_audit = audit_report_labeler_on_gold(patient_df, cfg.labels)
        parsed_Pairs = [parse_clinical_report(r, cfg.labels) for r in weak_raw_df["Report"]]
        weak_targets = np.vstack([p[0] for p in parsed_Pairs])
        weak_confs = np.vstack([p[1] for p in parsed_Pairs]) * float(cfg.weak_label_weight_scale)

        weak_studies_df = weak_raw_df[["StudyInstanceUID"]].copy()
        for idx, (col, w_col) in enumerate(zip(cfg.labels, WEIGHT_COLS)):
            weak_studies_df[col] = weak_targets[:, idx].astype(np.float32)
            weak_studies_df[w_col] = weak_confs[:, idx].astype(np.float32)
        weak_studies_df["is_gold"] = 0

        # Keep only weak studies that yielded at least 1 confident label AND strictly exclude val_gold_df
        val_uid_set = set(val_gold_df["StudyInstanceUID"])
        has_weak_signal = (weak_studies_df[WEIGHT_COLS].values > 0).any(axis=1)
        not_in_val = ~weak_studies_df["StudyInstanceUID"].isin(val_uid_set)
        weak_studies_df = weak_studies_df.loc[has_weak_signal & not_in_val].reset_index(drop=True)

    # Strict Zero-Leakage Assertion: Held-out validation studies never appear in train_gold or train_weak
    val_uids = set(val_gold_df["StudyInstanceUID"])
    train_gold_uids = set(train_gold_df["StudyInstanceUID"])
    train_weak_uids = set(weak_studies_df["StudyInstanceUID"]) if len(weak_studies_df) > 0 else set()
    assert val_uids.isdisjoint(train_gold_uids), "LEAKAGE between val_gold and train_gold!"
    assert val_uids.isdisjoint(train_weak_uids), "LEAKAGE between val_gold and train_weak!"

    # Compute class priors & pos_weights combining train_gold + train_weak
    combined_for_stats = pd.concat([train_gold_df[cfg.labels], weak_studies_df[cfg.labels]], ignore_index=True) if len(weak_studies_df) > 0 else train_gold_df[cfg.labels]
    pos_counts = combined_for_stats.fillna(0.0).sum(axis=0).values.astype(np.float64)
    valid_counts = combined_for_stats.notna().sum(axis=0).values.astype(np.float64)
    neg_counts = np.maximum(valid_counts - pos_counts, 1.0)

    raw_pos_weights = np.sqrt(neg_counts / np.maximum(pos_counts, 1.0))
    pos_weights = np.clip(raw_pos_weights, 1.0, cfg.pos_weight_cap).astype(np.float32)
    priors = np.clip(pos_counts / np.maximum(valid_counts, 1.0), 1e-4, 0.999).astype(np.float32)

    # Test set discovery
    test_studies_df, test_series_df = None, None
    test_ser_path = os.path.join(comp_dir, "test_series.csv")
    test_stu_path = os.path.join(comp_dir, "test.csv")
    if os.path.exists(test_ser_path):
        t_ser = pd.read_csv(test_ser_path)
        t_ser["StudyInstanceUID"] = t_ser["StudyInstanceUID"].map(_normalize_id)
        t_ser["SeriesInstanceUID"] = t_ser["SeriesInstanceUID"].map(_normalize_id)
        t_ser["npz_path"] = [
            npz_map.get(ser) or npz_map.get(f"{st}_{ser}") or npz_map.get(st)
            for st, ser in zip(t_ser["StudyInstanceUID"], t_ser["SeriesInstanceUID"])
        ]
        t_ser, _, _ = extract_and_validate_metadata(t_ser)
        test_series_df = t_ser
        if os.path.exists(test_stu_path):
            test_studies_df = pd.read_csv(test_stu_path)
            test_studies_df["StudyInstanceUID"] = test_studies_df["StudyInstanceUID"].map(_normalize_id)
        else:
            test_studies_df = pd.DataFrame({"StudyInstanceUID": t_ser["StudyInstanceUID"].unique()})

    report = {
        "total_npz_on_disk": len(all_npz_list),
        "total_studies_in_train_csv": len(patient_df),
        "total_series_matched": len(series_df),
        "gold_train_studies": len(train_gold_df),
        "gold_val_studies": len(val_gold_df),
        "weak_report_train_studies": len(weak_studies_df),
        "report_parser_gold_mean_acc": round(parser_audit.get("mean_acc", 0.0), 4),
        "report_parser_gold_mean_f1": round(parser_audit.get("mean_f1", 0.0), 4),
    }

    if rank == 0:
        print("=== STUDY-LEVEL DATASET & WEAK-SUPERVISION AUDIT ===")
        for k, v in report.items():
            print(f"  {k}: {v}")

    return DiscoveredDataset(
        comp_dir=comp_dir,
        train_gold_studies_df=train_gold_df,
        train_weak_studies_df=weak_studies_df,
        val_gold_studies_df=val_gold_df,
        series_df=series_df,
        test_studies_df=test_studies_df,
        test_series_df=test_series_df,
        npz_map=npz_map,
        has_valid_metadata=has_valid_meta,
        metadata_cols=meta_cols,
        pos_weights=pos_weights,
        priors=priors,
        diagnostics_report=report,
    )
