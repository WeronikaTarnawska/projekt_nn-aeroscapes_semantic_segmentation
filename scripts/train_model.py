from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import fiddle as fdl
import lightning as L
from fiddle.codegen import codegen
from fiddle.printing import as_dict_flattened

import wandb
from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.utils.config import get_wandb_config, parse_fiddle_config

_LOG_CKPT_SUFFIX = ".ckpt"


def _find_newest_ckpt_under_logs(logs_root: Path) -> Path | None:
    """Newest ``*.ckpt`` under ``logs/`` by modification time (recursive)."""
    if not logs_root.is_dir():
        return None
    best: Path | None = None
    best_mtime = -1.0
    for path in logs_root.rglob(f"*{_LOG_CKPT_SUFFIX}"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best = path
    return best


if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger

    from src.config.schemas import ExperimentConfig


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=False,
)
@click.option("--resume_run_name", type=str, default=None)
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=str),
    default=None,
    help="Resume from this Lightning checkpoint (.ckpt). New checkpoints stay in the same directory.",
)
@click.option(
    "--resume-latest",
    is_flag=True,
    default=False,
    help=f"Resume from the newest *{_LOG_CKPT_SUFFIX} under ./logs (by mtime). Requires CONFIG_PATH.",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Total max_epochs for the Trainer (overrides the value from the Fiddle config). Use to extend training when resuming.",
)
@click.option("--no_wandb", is_flag=True, default=False)
@click.option("--seed", type=int, default=42)
def main(
    config_path,
    resume_run_name,
    ckpt,
    resume_latest,
    epochs,
    no_wandb,
    seed,
):
    L.seed_everything(seed)

    if resume_latest and ckpt is not None:
        raise click.UsageError("Use either --resume-latest or --ckpt, not both.")
    if resume_latest and resume_run_name is not None:
        raise click.UsageError(
            "--resume-latest cannot be used with --resume_run_name (that run already fixes the log directory)."
        )
    if resume_latest and not config_path:
        raise click.UsageError(
            "--resume-latest requires CONFIG_PATH (same experiment config as before)."
        )

    ckpt_path: str | None = ckpt

    if resume_run_name is not None:
        # Fetch the config from the original run stored in W&B — ensures the resumed run
        # uses the exact same config. config_path is ignored.
        cfg: fdl.Config[ExperimentConfig] = get_wandb_config(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/{resume_run_name}"
        )
        run_name = resume_run_name
        log_dir = f"logs/{run_name}"

        if ckpt_path is None:
            # Find the latest checkpoint to resume training from, preferring last.ckpt —
            # Lightning's dedicated file that always points to the most recent epoch.
            last_ckpt = Path(log_dir) / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
            else:
                ckpts = sorted(
                    Path(log_dir).glob(f"*{_LOG_CKPT_SUFFIX}"),
                    key=lambda p: p.stat().st_mtime,
                )
                if ckpts:
                    ckpt_path = str(ckpts[-1])
            if ckpt_path is None:
                warnings.warn(f"No checkpoint found in {log_dir}")
    elif resume_latest:
        found = _find_newest_ckpt_under_logs(Path("logs"))
        if found is None:
            raise click.UsageError("No .ckpt files found under ./logs")
        ckpt_path = str(found)
        log_dir = str(found.parent)
        run_name = found.parent.name
        cfg = parse_fiddle_config(config_path)
    elif ckpt_path is not None:
        if not config_path:
            raise click.UsageError(
                "--ckpt requires CONFIG_PATH (same experiment config as before)."
            )
        # Local resume: keep writing checkpoints next to the loaded file.
        p = Path(ckpt_path)
        log_dir = str(p.parent)
        run_name = p.parent.name
        cfg = parse_fiddle_config(config_path)
    else:
        if not config_path:
            raise click.UsageError(
                "CONFIG_PATH is required unless --resume_run_name is set."
            )
        # Config files are Python modules that must be imported dynamically —
        # parse_fiddle_config handles this.
        cfg = parse_fiddle_config(config_path)
        # Suffix with timestamp to ensure the run name is unique across runs of the same config.
        run_name = cfg.name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = f"logs/{run_name}"

    built_cfg: ExperimentConfig = fdl.build(cfg)
    max_epochs = built_cfg.training_cfg.max_epochs if epochs is None else epochs
    # model is a LightningModule — it encapsulates the architecture, loss function,
    # optimizer, and train/val/test step logic, keeping this script model-agnostic.
    model: L.LightningModule = built_cfg.model

    # wandb_logger and checkpoint_callback are stored as partials in the config because
    # they require run_name and log_dir, which are only known at runtime
    partial_checkpoint_callback = built_cfg.training_cfg.checkpoint_callback
    checkpoint_callback = (
        partial_checkpoint_callback(dirpath=log_dir)
        if partial_checkpoint_callback is not None
        else None
    )
    callbacks = built_cfg.training_cfg.callbacks + (
        [checkpoint_callback] if checkpoint_callback is not None else []
    )

    logger: list[Logger] = []
    partial_wandb_logger = built_cfg.training_cfg.wandb_logger
    if partial_wandb_logger is not None and not no_wandb:
        # id ties this run to its W&B entry — required for resumption.
        # resume="allow" means W&B will start a fresh run if no run with this id exists yet.
        wandb_logger = partial_wandb_logger(
            id=run_name,
            name=run_name,
            resume="allow",
            tags=[f"seed: {seed}"],
        )
        logger.append(wandb_logger)

        if epochs is not None:
            wandb_logger.experiment.config.update(
                {"max_epochs": max_epochs}, allow_val_change=True
            )

        # Config is only uploaded on new runs — it was already uploaded during the original run.
        if config_path is not None and resume_run_name is None and ckpt_path is None:
            # Upload config values as W&B run config — populates the hyperparameter columns
            # in the runs table, enabling filtering and side-by-side comparison.
            wandb_logger.experiment.config.update(as_dict_flattened(cfg))
            # fiddle.codegen serializes the fully resolved config — all values are inlined,
            # including those inherited from base configs. This ensures the uploaded file is
            # self-contained and accurately reflects what was actually used in this run.
            # Uploading the file at config_path directly would only capture the overrides,
            # losing any values inherited from a base config (e.g. if config_path points to
            # src/config/lr_comparison/1e-2.py, all values from convnet_utkface_classifier.py
            # would be lost).
            generated = codegen.codegen_dot_syntax(cfg)
            code_str = "\n".join(generated.lines())
            artifact = wandb.Artifact(name=f"{run_name}_config", type="config")
            with artifact.new_file("config.py", mode="w") as file:
                file.write(code_str)

            wandb_logger.experiment.log_artifact(artifact)

    # data_module is a LightningDataModule — it encapsulates the dataset, train/val/test
    # splits, transforms, and dataloaders, keeping data preparation self-contained.
    data_module: L.LightningDataModule = built_cfg.data_module

    trainer = L.Trainer(
        log_every_n_steps=50,
        logger=logger,
        max_epochs=max_epochs,
        default_root_dir=log_dir,
        callbacks=callbacks,
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
