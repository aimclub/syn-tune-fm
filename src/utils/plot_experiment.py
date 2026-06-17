"""Bar charts for experiment variants (same layout as run_experiment.plot_experiment_results)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils.experiment_labels import (
    dataset_display_name_from_cfg,
    dataset_label_from_cfg,
    safe_filename_part,
)


VARIANTS_WIDE_C2 = [
    "C0",
    "C1",
    "C2_ctgan",
    "C2_tvae",
    "C2_tabddpm",
    "C3",
    "C4",
]


def plot_experiment_results(
    results: dict,
    cfg: DictConfig,
    shared_dir: Path,
    *,
    plot_run_suffix: str | None = None,
) -> None:
    """Creates and saves bar charts for each metric in cfg.evaluation.metrics.

    plot_run_suffix:
        Unique per Hydra job (e.g. ``multirun_time__jobidx``) so PNGs are not overwritten
        when repeating the same dataset/generator under a shared multirun directory.
    """
    metrics = cfg.evaluation.metrics
    variants = list(results.keys())
    ds_display = dataset_display_name_from_cfg(cfg.dataset)
    ds_label = safe_filename_part(ds_display)
    gen_label = safe_filename_part(cfg.generator.name)
    suffix_part = safe_filename_part(plot_run_suffix) if plot_run_suffix else ""

    # One folder per dataset: shared_dir/<dataset_name_safe>/...
    dataset_dir = shared_dir / ds_label
    dataset_dir.mkdir(parents=True, exist_ok=True)

    minority_frac = cfg.get("minority_fraction", "null")

    for metric in metrics:
        means = []
        stds = []

        for var in variants:
            means.append(results[var].get(f"{metric}_mean", 0.0))
            stds.append(results[var].get(f"{metric}_std", 0.0))

        fig_w = 10.0 + max(0, len(variants) - 5) * 0.85
        plt.figure(figsize=(fig_w, 6))
        x_pos = np.arange(len(variants))

        bars = plt.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=8,
            alpha=0.8,
            color="#4C72B0",
            edgecolor="black",
        )

        plt.xticks(x_pos, variants, rotation=45, ha="right")
        plt.ylabel(metric.replace("_", " ").title())

        title_mf = minority_frac if minority_frac is not None else "None (50%)"
        plt.title(
            f"Dataset: {ds_display} ({dataset_label_from_cfg(cfg.dataset)}) | "
            f"Generator: {cfg.generator.name}\nMinority Fraction: {title_mf} | Metric: {metric}"
        )

        for bar, mean_val, std_val in zip(bars, means, stds):
            yval = bar.get_height()
            offset = std_val if std_val > 0 else 0
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + offset + 0.005,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        if len(variants) > 5:
            plt.subplots_adjust(bottom=0.22)
        else:
            plt.tight_layout()

        if suffix_part:
            plot_path = (
                dataset_dir
                / f"{metric}_run_{suffix_part}_ds_{ds_label}_gen_{gen_label}_mf_{minority_frac}.png"
            )
        else:
            plot_path = dataset_dir / f"{metric}_ds_{ds_label}_gen_{gen_label}_mf_{minority_frac}.png"
        plt.savefig(str(plot_path), dpi=300)
        plt.close()
        print(f"      [Plot Saved] {plot_path}")


def csv_row_to_results_and_cfg(
    row: Any,
    metrics: list[str],
) -> tuple[dict, DictConfig]:
    """Rebuild results dict + minimal cfg from one flattened CSV row (Series or dict)."""
    variants_order = [
        "C0_No_FT",
        "C1_FT_Imbal",
        "C2_FT_SynBal",
        "C3_FT_RandBal",
        "C4_FT_DownBal",
    ]
    results: dict[str, dict] = {v: {} for v in variants_order}
    r = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    for v in variants_order:
        for m in metrics:
            k_mean = f"{v}_{m}_mean"
            k_std = f"{v}_{m}_std"
            if k_mean in r and not pd.isna(r[k_mean]):
                results[v][f"{m}_mean"] = float(r[k_mean])
            if k_std in r and not pd.isna(r[k_std]):
                results[v][f"{m}_std"] = float(r[k_std])

    ds_name = str(r.get("dataset", "openml"))
    params: dict[str, Any] = {}
    did = r.get("dataset_id", "")
    if did is not None and str(did).strip() != "" and str(did).lower() != "nan":
        try:
            params["dataset_id"] = int(float(did))
        except (TypeError, ValueError):
            pass
    mf_raw = r.get("minority_fraction")
    if mf_raw is None or (isinstance(mf_raw, float) and np.isnan(mf_raw)) or str(mf_raw).strip() in ("", "None", "nan"):
        mf_out = None
    else:
        mf_out = mf_raw

    cfg = OmegaConf.create(
        {
            "dataset": {"name": ds_name, "params": params},
            "generator": {"name": str(r.get("generator", "unknown"))},
            "evaluation": {"metrics": metrics},
            "minority_fraction": mf_out,
        }
    )

    return results, cfg


def wide_csv_row_to_results_and_cfg(
    row: Any,
    metrics: list[str],
) -> tuple[dict, DictConfig]:
    """One wide CSV row: C0, C1, C2_ctgan, C2_tvae, C2_tabddpm, C3, C4 (see build_wide_dataset_csv)."""
    results: dict[str, dict] = {v: {} for v in VARIANTS_WIDE_C2}
    r = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    for v in VARIANTS_WIDE_C2:
        for m in metrics:
            km = f"{v}_{m}_mean"
            ks = f"{v}_{m}_std"
            if km in r and not pd.isna(r[km]):
                results[v][f"{m}_mean"] = float(r[km])
            if ks in r and not pd.isna(r[ks]):
                results[v][f"{m}_std"] = float(r[ks])

    ds_name = str(r.get("dataset", "openml"))
    params: dict[str, Any] = {}
    did = r.get("dataset_id", "")
    if did is not None and str(did).strip() != "" and str(did).lower() != "nan":
        try:
            params["dataset_id"] = int(float(did))
        except (TypeError, ValueError):
            pass

    mf_raw = r.get("minority_fraction")
    if mf_raw is None or (isinstance(mf_raw, float) and np.isnan(mf_raw)) or str(mf_raw).strip() in ("", "None", "nan"):
        mf_out = None
    else:
        mf_out = mf_raw

    cfg = OmegaConf.create(
        {
            "dataset": {"name": ds_name, "params": params},
            "generator": {"name": "c2_syn_by_generator"},
            "evaluation": {"metrics": metrics},
            "minority_fraction": mf_out,
        }
    )

    return results, cfg
