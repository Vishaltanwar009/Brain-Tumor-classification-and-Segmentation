"""
Data splitting protocols:
1. Patient-level splitting (StratifiedGroupKFold) with strict zero-leakage assertions.
2. Slice-level splitting (stratified) for direct literature replication with leakage auditing.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def audit_patient_leakage(train_pids: list, test_pids: list) -> dict:
    """
    Checks whether any patient IDs appear in both train and test splits.
    """
    tr_set = set(str(p).strip() for p in train_pids)
    te_set = set(str(p).strip() for p in test_pids)
    overlap = tr_set.intersection(te_set)
    return {
        "n_train_patients": len(tr_set),
        "n_test_patients": len(te_set),
        "n_overlapping_patients": len(overlap),
        "overlapping_pids": sorted(list(overlap))[:10],
        "is_leak_free": len(overlap) == 0,
    }


def create_patient_level_splits(
    y: np.ndarray,
    pids: np.ndarray,
    n_test_splits: int = 5,
    n_cv_splits: int = 5,
    seed: int = 42,
    num_tries: int = 300,
) -> dict:
    """
    Creates patient-level hold-out test set (~20%) and 5-fold CV splits on the remaining patients.
    Screens multiple seeds to find the candidate with optimal class balance while guaranteeing
    zero patient leakage.
    """
    pids = np.array([str(p).strip() for p in pids])
    pid_unique = np.unique(pids)
    nc = len(np.unique(y))

    # Patient majority label for stratification
    pid_majority = {p: int(np.bincount(y[pids == p], minlength=nc).argmax()) for p in pid_unique}
    y_strat = np.array([pid_majority[p] for p in pids])

    # Hold-out test split
    best_test_fold = None
    best_score = float("inf")
    target_dist = np.bincount(y, minlength=nc) / len(y)

    for t in range(num_tries):
        rs = seed + t
        sgkf = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=rs)
        try:
            folds = [te for _, te in sgkf.split(np.zeros(len(y)), y_strat, pids)]
        except ValueError:
            continue
        test_idx = folds[0]
        dist = np.bincount(y[test_idx], minlength=nc) / len(test_idx)
        score = float(np.mean((dist - target_dist) ** 2))
        if score < best_score:
            best_score = score
            best_test_fold = test_idx

    test_idx = np.sort(best_test_fold)
    train_val_idx = np.sort(np.setdiff1d(np.arange(len(y)), test_idx))

    # Assert zero patient leakage
    test_pids = set(pids[test_idx])
    train_val_pids = set(pids[train_val_idx])
    overlap = test_pids.intersection(train_val_pids)
    assert len(overlap) == 0, f"Critical error: {len(overlap)} patients leaked between train and test: {overlap}"

    # CV folds on remaining train_val data
    y_tv = y_strat[train_val_idx]
    pids_tv = pids[train_val_idx]
    sgkf_cv = StratifiedGroupKFold(n_splits=n_cv_splits, shuffle=True, random_state=seed)
    cv_folds = []
    for tr, va in sgkf_cv.split(np.zeros(len(y_tv)), y_tv, pids_tv):
        tr_full = train_val_idx[tr]
        va_full = train_val_idx[va]
        assert len(set(pids[tr_full]).intersection(set(pids[va_full]))) == 0
        cv_folds.append((tr_full, va_full))

    return {
        "test_idx": test_idx,
        "train_val_idx": train_val_idx,
        "cv_folds": cv_folds,
        "n_test_images": len(test_idx),
        "n_test_patients": len(test_pids),
    }
