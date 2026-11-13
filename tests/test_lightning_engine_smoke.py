"""A real ``pl.Trainer`` driving ``EngineLightning`` on CPU with a tiny task."""

import importlib.util

import pytest
import torch
from torch import nn

pl = pytest.importorskip("pytorch_lightning")

from MolecularDiffusion.cli.train import resolve_lightning_precision  # noqa: E402
from MolecularDiffusion.core.engine_lightning import EngineLightning  # noqa: E402


class _Task(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.backbone = nn.Linear(2, 2)
        self.lora_A = nn.Linear(2, 2, bias=False)
        self.lora_B = nn.Linear(2, 2, bias=False)

    def forward(self, batch):
        out = self.lora_B(self.lora_A(self.backbone(batch)))
        loss = out.pow(2).mean()
        return loss, {"loss": loss.detach()}


def _fit(tmp_path, opt_cfg, sched=None, precision=32, epochs=2):
    engine = EngineLightning(
        optimizer_config={"optimizer_choice": "adamw", "weight_decay": 0.0, **opt_cfg},
        task=_Task(),
        scheduler_config=sched or {},
    )
    before = {k: v.clone() for k, v in engine.task.state_dict().items()}
    loader = torch.utils.data.DataLoader(torch.randn(4, 2), batch_size=2)
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path), max_epochs=epochs, accelerator="cpu", logger=False,
        enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False,
        precision=precision, num_sanity_val_steps=0,
    )
    trainer.fit(engine, loader, loader)
    return trainer, engine, before


def _changed(engine, before, prefix):
    sd = engine.task.state_dict()
    return any(not torch.equal(sd[k], before[k]) for k in sd if k.startswith(prefix))


def test_default_config_trains(tmp_path) -> None:
    _, engine, before = _fit(tmp_path, {"lr": 1e-2})
    assert _changed(engine, before, "backbone") and _changed(engine, before, "lora_A")


@pytest.mark.skipif(
    importlib.util.find_spec("MolecularDiffusion.core.optim_utils") is None,
    reason="LoRA/adapter param groups are not in this checkout",
)
def test_lora_lr_trains_the_lora_group_while_the_base_group_stays_put(tmp_path) -> None:
    # base lr 0: only a group with its own learning rate can move
    _, engine, before = _fit(tmp_path, {"lr": 0.0, "lora_lr": 5e-2})
    assert _changed(engine, before, "lora_A") and _changed(engine, before, "lora_B")
    assert not _changed(engine, before, "backbone")


def test_caws_scheduler_steps_during_training(tmp_path) -> None:
    trainer, _, _ = _fit(
        tmp_path, {"lr": 1e-2},
        sched={"scheduler": "caws", "scheduler_kwargs": {"T_0": 1}}, epochs=3,
    )
    sched = trainer.lr_scheduler_configs[0].scheduler
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
    assert sched.last_epoch == 3


def test_lightning_native_precision_name_reaches_the_trainer(tmp_path) -> None:
    trainer, _, _ = _fit(tmp_path, {"lr": 1e-2}, precision=resolve_lightning_precision("bf16-mixed"), epochs=1)
    assert trainer.precision == "bf16-mixed"
