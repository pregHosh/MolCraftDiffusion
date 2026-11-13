"""
Generative evaluation callback for diffusion models.

Performs molecule generation and validity checking during validation.
"""

from typing import Any
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from MolecularDiffusion.runmodes.train.eval import analyze_and_save

logger = logging.getLogger(__name__)


class GenerativeEvalCallback(Callback):
    """
    Generative evaluation callback for diffusion models.

    Generates molecules during validation and computes validity metrics.
    This callback runs the generative analysis from eval.py.

    Parameters:
        n_samples (int): Number of molecules to generate
        batch_size (int): Batch size for generation
        metric (str): Primary metric to monitor
        output_dir (str): Directory to save generated molecules
        use_posebuster (bool): Whether to use posebuster for validation
        posebuster_timeout (int): Timeout for posebuster validation
        use_ema (bool): Generate from the EMA copy of the weights when the run
            keeps one (what the original engine always does and what the
            validation loss uses). Default False keeps the raw weights.
    """

    def __init__(
        self,
        n_samples: int = 100,
        batch_size: int = 100,
        metric: str = "Validity Relax and connected",
        output_dir: str = "generated_molecules",
        use_posebuster: bool = False,
        posebuster_timeout: int = 60,
        monitor_metric: Any = None,
        use_ema: bool = False,
    ):
        super().__init__()
        self.use_ema = use_ema
        self._warned_raw_weights = False
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.metric = metric
        self.output_dir = output_dir
        self.use_posebuster = use_posebuster
        self.posebuster_timeout = posebuster_timeout
        self.monitor_metric = monitor_metric

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Run generative evaluation at the end of validation epoch.

        Only runs if this is a validation epoch (respects check_val_every_n_epoch).
        """
        # Create output directory if it doesn't exist
        if trainer.is_global_zero:
            os.makedirs(self.output_dir, exist_ok=True)

        # Only rank 0 performs generation to avoid duplicates
        if not trainer.is_global_zero:
            return

        logger.info(f"Running generative evaluation: {self.n_samples} samples")

        try:
            ema = getattr(pl_module, "ema_model", None)
            if self.use_ema and ema is not None:
                model = ema
            else:
                model = pl_module.task
                if ema is not None and not self._warned_raw_weights:
                    self._warned_raw_weights = True
                    logger.info(
                        "Generative evaluation uses the raw weights although the run keeps an "
                        "EMA copy (validation loss uses the EMA). Set generative_eval_use_ema: "
                        "true to generate from the EMA weights, as the original engine does."
                    )
            stats = analyze_and_save(
                model=model,
                epoch=trainer.current_epoch,
                n_samples=self.n_samples,
                batch_size=self.batch_size,
                logger="logging",  # Use logging instead of wandb for compatibility
                path_save=os.path.join(self.output_dir, f"gen_xyz_{trainer.current_epoch}"),
                use_posebuster=self.use_posebuster,
                postbuster_timeout=self.posebuster_timeout,
            )

            # Log statistics to trainer
            for key, value in stats.items():
                # Filter progress bar if monitor_metric is set
                show_on_bar = True
                if self.monitor_metric is not None:
                     if isinstance(self.monitor_metric, str):
                         targets = [self.monitor_metric]
                     else:
                         targets = self.monitor_metric

                     metric_full_name = f"gen/{key}"
                     show_on_bar = (metric_full_name in targets)

                pl_module.log(f"gen/{key}", value, on_epoch=True, rank_zero_only=True, prog_bar=show_on_bar)

            logger.info(f"Generative evaluation completed: {stats}")

        except Exception as e:
            logger.error(f"Generative evaluation failed: {e}", exc_info=True)
