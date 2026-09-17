"""Root vs tasks ``diffusion_steps`` reconciliation in ``cli/generate.generate``.

Rule: a root value > 0 always wins; only 0 / None / absent falls back to
``tasks.diffusion_steps``. A former ``== 900`` "unset" sentinel made a root 900
silently lose to e.g. ``diffusion_painn``'s tasks default of 400.
"""

import pytest
from omegaconf import OmegaConf

import MolecularDiffusion.cli.main  # noqa: F401  (avoids a circular import)
import MolecularDiffusion.cli.generate as g


class _Captured(Exception):
    pass


def _fake_load_model(chkpt_directory, task_config=None, atom_vocab=None,
                     total_step=0, base_chkpt_path=None):
    raise _Captured(total_step)


@pytest.mark.parametrize(
    ("root", "tasks", "expected"),
    [
        (900, 400, 900),   # the bug: used to return 400
        (1000, 400, 1000),
        (500, 400, 500),
        (0, 400, 400),
        (None, 400, 400),
        ("absent", 400, 400),
        (900, 900, 900),
        ("absent", "absent", 0),
    ],
)
def test_root_diffusion_steps_wins_unless_unset(monkeypatch, root, tasks, expected):
    cfg = {"interference": {}, "chkpt_directory": "unused", "atom_vocab": ["H"],
           "tasks": {}}
    if root != "absent":
        cfg["diffusion_steps"] = root
    if tasks != "absent":
        cfg["tasks"]["diffusion_steps"] = tasks
    monkeypatch.setattr(g, "load_model", _fake_load_model)
    with pytest.raises(_Captured) as got:
        g.generate(OmegaConf.create(cfg))
    assert got.value.args[0] == expected
