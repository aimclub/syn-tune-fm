"""Human-readable labels for logging and unique artifact names (OpenML id vs generic name)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

OPENML_ID_TO_NAME = {
    24: "mushroom",
    31: "credit_g",
    37: "diabetes",
    59: "ionosphere",
    61: "iris",
    179: "adult",
    1049: "pc4",
    1056: "mc1",
    1461: "bank_marketing",
    1464: "blood_transfusion",
    1488: "parkinsons",
    42178: "telco_churn",
    531: "boston_housing",
    183: "abalone",
    42225: "diamonds",
    42165: "house_prices",
    42570: "mercedes_benz",
    42688: "brazilian_houses",
}


def dataset_label_from_cfg(dataset_cfg: Any) -> str:
    """Stable string: which data we actually use (not just 'openml')."""
    name = getattr(dataset_cfg, "name", None) or (dataset_cfg or {}).get("name", "unknown")
    params = getattr(dataset_cfg, "params", None) or (dataset_cfg or {}).get("params") or {}
    if isinstance(params, dict):
        p = params
    else:
        try:
            p = dict(params)
        except Exception:
            p = {}

    if str(name) == "openml" or "dataset_id" in p:
        did = p.get("dataset_id", "?")
        return f"openml_{did}"

    if str(name) == "csv":
        train_path = p.get("train_path", "train")
        return f"csv_{Path(str(train_path)).stem}"

    return str(name)


def safe_filename_part(s: str, max_len: int = 120) -> str:
    import re

    out = re.sub(r"[^\w.\-]+", "_", str(s).strip())
    return out[:max_len] if len(out) > max_len else out


def dataset_display_name_from_cfg(dataset_cfg: Any) -> str:
    """Human-friendly name for charts/titles while keeping IDs available."""
    name = getattr(dataset_cfg, "name", None) or (dataset_cfg or {}).get("name", "unknown")
    params = getattr(dataset_cfg, "params", None) or (dataset_cfg or {}).get("params") or {}
    if isinstance(params, dict):
        p = params
    else:
        try:
            p = dict(params)
        except Exception:
            p = {}

    if str(name) == "openml" or "dataset_id" in p:
        did_raw = p.get("dataset_id")
        try:
            did = int(did_raw)
        except (TypeError, ValueError):
            return f"openml_{did_raw}"
        return OPENML_ID_TO_NAME.get(did, f"openml_{did}")

    if str(name) == "csv":
        train_path = p.get("train_path", "train")
        return Path(str(train_path)).stem

    return str(name)
