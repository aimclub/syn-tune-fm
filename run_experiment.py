import logging
import os
import sys
from pathlib import Path

from src.utils.tabpfn_env import ensure_tabpfn_token

ensure_tabpfn_token()

# TabPFN logs use emoji; cp1251 consoles choke on them — strip non-ASCII from tabpfn* messages only.
_orig_logger_log = logging.Logger._log


def _patched_logger_log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
    if self.name.startswith("tabpfn") and isinstance(msg, str):
        msg = msg.encode("ascii", errors="replace").decode("ascii")
    return _orig_logger_log(
        self,
        level,
        msg,
        args,
        exc_info=exc_info,
        extra=extra,
        stack_info=stack_info,
        stacklevel=stacklevel,
    )


logging.Logger._log = _patched_logger_log

# Optional: UTF-8 for our own prints (Hydra / tqdm).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
import pandas as pd

from src.pipeline.runner import ExperimentRunner
from src.utils.experiment_labels import dataset_display_name_from_cfg, dataset_label_from_cfg
from src.utils.plot_experiment import plot_experiment_results


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def main(cfg: DictConfig):
    # 1. Start experiment
    runner = ExperimentRunner(cfg)
    results = runner.run()

    # --- DEFINE SHARED FOLDER FOR CURRENT RUN (WITH TIMESTAMP) ---
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    
    if hydra_cfg.mode == hydra.types.RunMode.MULTIRUN:
        shared_exp_dir = output_dir.parent
    else:
        shared_exp_dir = output_dir

    # Unique suffix per Hydra child / single run folder (avoids clobbering PNGs in shared multirun dir).
    plot_run_suffix = f"{output_dir.parent.name}__{output_dir.name}"

    # 2. Generate plots in the shared folder
    print("\n[5/5] Generating Visualizations...")
    plot_experiment_results(results, cfg, shared_exp_dir, plot_run_suffix=plot_run_suffix)

    # 3. Collect parameters of the current run
    p = OmegaConf.to_container(cfg.dataset.params, resolve=True) if cfg.dataset.get("params") else {}
    dataset_id = p.get("dataset_id") if isinstance(p, dict) else None
    choices = getattr(hydra_cfg.runtime, "choices", None) or {}
    dataset_choice = choices.get("dataset", "") if isinstance(choices, dict) else ""
    generator_choice = choices.get("generator", "") if isinstance(choices, dict) else ""
    # hydra.job.num exists for multirun jobs, but may be missing in a single run.
    # Use a safe lookup to avoid crashing after successful training/evaluation.
    job_num = OmegaConf.select(hydra_cfg, "job.num", default=None)
    row = {
        "hydra_job_num": job_num if job_num is not None else "",
        "hydra_output_dir": str(output_dir),
        "hydra_group_dir": str(shared_exp_dir),
        "hydra_plot_suffix": plot_run_suffix,
        "dataset": cfg.dataset.name,
        "dataset_config": dataset_choice,
        "dataset_label": dataset_label_from_cfg(cfg.dataset),
        "dataset_display_name": dataset_display_name_from_cfg(cfg.dataset),
        "dataset_id": dataset_id if dataset_id is not None else "",
        "generator": cfg.generator.name,
        "generator_config": generator_choice,
        "model": cfg.model.name,
        "minority_fraction": cfg.get("minority_fraction", "None"),
    }
    
    # 4. Unpack results by variants
    if isinstance(results, dict):
        for variant, metrics in results.items(): 
            for metric_name, val in metrics.items():
                row[f"{variant}_{metric_name}"] = round(val, 4)

    df = pd.DataFrame([row])

    # 5. Save local CSV specifically for this run (in folder with timestamp)
    local_csv = shared_exp_dir / "run_results.csv"
    df.to_csv(str(local_csv), mode='a', header=not local_csv.exists(), index=False)

    # 5b. One-row CSV inside this job's Hydra output dir (never shared / never appended across jobs).
    per_job_csv = output_dir / "run_metrics_row.csv"
    df.to_csv(str(per_job_csv), index=False)

    # 6. Save global CSV for history (in project root)
    orig_cwd = Path(get_original_cwd())
    global_csv = orig_cwd / "all_experiments_results.csv"
    df.to_csv(str(global_csv), mode='a', header=not global_csv.exists(), index=False)
    
    print("\n" + "="*80)
    print(f"EXPERIMENTS SUMMARY TABLE")
    print(f" -> Local record (append): {local_csv}")
    print(f" -> Per-job row (overwrite only this job file): {per_job_csv}")
    print(f" -> Global record (append): {global_csv}")
    print("="*80)
    print(df.T.to_string(header=False))

if __name__ == "__main__":
    main()
