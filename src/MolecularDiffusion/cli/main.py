"""MolCraft CLI - Unified command-line interface for MolecularDiffusion.

Usage:
    molcraft train config.yaml [overrides...]
    molcraft generate config.yaml [overrides...]
    molcraft predict config.yaml [overrides...]
"""

import os
import logging
import platform

import click

# Printed on every CLI run (incl. --help/--version), not on library import.
_BANNER = r"""
░█▄█░█▀█░█░░░█▀▀░█▀▄░█▀█░█▀▀░▀█▀░█▀▄░▀█▀░█▀▀░█▀▀
░█░█░█░█░█░░░█░░░█▀▄░█▀█░█▀▀░░█░░█░█░░█░░█▀▀░█▀▀
░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░▀░░░░▀░░▀▀░░▀▀▀░▀░░░▀░░"""

print(_BANNER)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_system_info():
    """Log basic system information."""
    import psutil
    
    logger.info("=" * 60)
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"CPU: {platform.processor()}, Cores: {os.cpu_count()}")
    
    ram = psutil.virtual_memory()
    logger.info(f"RAM: Total {ram.total / (1024**3):.2f} GB, Available {ram.available / (1024**3):.2f} GB")
    logger.info(f"Python: {platform.python_version()}")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}, GPUs: {torch.cuda.device_count()}")
        elif torch.backends.mps.is_available():
            logger.info("Apple MPS: available")
    except ImportError:
        pass
    
    logger.info("=" * 60)


# Enable -h as alias for --help
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(package_name="molcraftdiffusion")
def cli():
    """MolCraft - Molecular Diffusion CLI.
    
    A unified command-line interface for training, generation, and prediction
    with molecular diffusion models.
    
    \b
    Examples:
        molcraft train configs/my_train_config.yaml
        molcraft generate configs/my_gen_config.yaml 
        molcraft predict configs/my_pred_config.yaml
    """
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config", type=str)
@click.argument("overrides", nargs=-1)
def train(config: str, overrides: tuple):
    """Train a molecular diffusion model.
    
    \b
    Arguments:
        CONFIG     Config file path (e.g., configs/train.yaml)
        OVERRIDES  Hydra-style config overrides (e.g., trainer.num_epochs=100)
    
    \b
    Examples:
        molcraft train configs/train_tabasco_geom.yaml
        molcraft train configs/my_config.yaml trainer.num_epochs=50 seed=42
    """
    log_system_info()
    logger.info(f"Starting training with config: {config}")
    
    from MolecularDiffusion.cli._hydra import run_hydra_app
    from MolecularDiffusion.cli.train import train_main
    
    run_hydra_app(
        config_name=config,
        task_function=train_main,
        config_dir=None,
        overrides=list(overrides),
    )


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config", type=str)
@click.argument("overrides", nargs=-1)
def generate(config: str, overrides: tuple):
    """Generate molecules using a trained model.
    
    \b
    Arguments:
        CONFIG     Config file path (e.g., configs/generate.yaml)
        OVERRIDES  Hydra-style config overrides
    
    \b
    Examples:
        molcraft generate configs/gen_config.yaml
        molcraft generate configs/gen_config.yaml interference.n_samples=1000
    """
    log_system_info()
    logger.info(f"Starting generation with config: {config}")
    
    from MolecularDiffusion.cli._hydra import run_hydra_app
    from MolecularDiffusion.cli.generate import generate_main
    
    run_hydra_app(
        config_name=config,
        task_function=generate_main,
        config_dir=None,
        overrides=list(overrides),
    )


@cli.command("generate-sweep", context_settings=CONTEXT_SETTINGS)
@click.argument("sweep_config", type=str)
@click.option("--dry-run", is_flag=True, help="Print commands without executing them.")
@click.option("--skip-gen", is_flag=True, help="Skip generation for each run.")
@click.option("--skip-eval", is_flag=True, help="Skip evaluation for each run.")
@click.option("--max-runs", type=int, default=None, metavar="N",
              help="Run at most N new combinations.")
@click.option("--retry-failed", is_flag=True,
              help="Retry configurations recorded as failed or interrupted.")
@click.option("--early-fail-batches", type=int, default=None, metavar="N",
              help="Stop generation after N consecutive failed batches before any success.")
def generate_sweep(
    sweep_config: str,
    dry_run: bool,
    skip_gen: bool,
    skip_eval: bool,
    max_runs: int | None,
    retry_failed: bool,
    early_fail_batches: int | None,
):
    """Run a generation parameter sweep from a sweep YAML config."""
    from MolecularDiffusion.runmodes.generate.sweep import main as sweep_main

    argv = [sweep_config]
    if dry_run:
        argv.append("--dry-run")
    if skip_gen:
        argv.append("--skip-gen")
    if skip_eval:
        argv.append("--skip-eval")
    if max_runs is not None:
        argv.extend(["--max-runs", str(max_runs)])
    if retry_failed:
        argv.append("--retry-failed")
    if early_fail_batches is not None:
        argv.extend(["--early-fail-batches", str(early_fail_batches)])
    sweep_main(argv)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("config", type=str)
@click.argument("overrides", nargs=-1)
def predict(config: str, overrides: tuple):
    """Run property prediction on molecules.
    
    \b
    Arguments:
        CONFIG     Config file path (e.g., configs/predict.yaml)
        OVERRIDES  Hydra-style config overrides
    
    \b
    Examples:
        molcraft predict configs/predict.yaml
        molcraft predict configs/my_pred.yaml xyz_directory=/path/to/xyz
    """
    log_system_info()
    logger.info(f"Starting prediction with config: {config}")
    
    from MolecularDiffusion.cli._hydra import run_hydra_app
    from MolecularDiffusion.cli.predict import predict_main
    
    run_hydra_app(
        config_name=config,
        task_function=predict_main,
        config_dir=None,
        overrides=list(overrides),
    )


@cli.command("eval-predict", context_settings=CONTEXT_SETTINGS)
@click.argument("config", type=str)
@click.argument("overrides", nargs=-1)
def eval_predict(config: str, overrides: tuple):
    """Evaluate model predictions on validation/test sets.
    
    \b
    Arguments:
        CONFIG     Config file path (e.g., configs/eval_predict.yaml)
        OVERRIDES  Hydra-style config overrides
    
    \b
    Examples:
        molcraft eval-predict configs/eval_predict.yaml
    """
    log_system_info()
    logger.info(f"Starting eval-predict with config: {config}")
    
    from MolecularDiffusion.cli._hydra import run_hydra_app
    from MolecularDiffusion.cli.eval_predict import eval_predict_main
    
    run_hydra_app(
        config_name=config,
        task_function=eval_predict_main,
        config_dir=None,
        overrides=list(overrides),
    )


# Register analyze subcommand group
from MolecularDiffusion.cli.analyze import analyze
cli.add_command(analyze)

# Register data subcommand group
from MolecularDiffusion.cli.data import data
cli.add_command(data)

# Register zoo subcommand group
from MolecularDiffusion.cli.zoo import zoo
cli.add_command(zoo)


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
