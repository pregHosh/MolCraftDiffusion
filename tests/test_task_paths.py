"""Old dotted paths of task classes keep resolving.

``task_class_paths.lst`` lists ``module:Class`` for every top-level class in the
flat ``modules/tasks`` modules, recorded before they were grouped into family
packages. Checkpoints pickle some of these (node-distribution objects) and
saved configs name the factories by module path, so each must still resolve.
"""

import importlib
from pathlib import Path

import pytest

# Imported first on purpose: the task modules hit a known circular import otherwise.
import MolecularDiffusion.data.dataset  # noqa: F401

PATHS = Path(__file__).with_name("task_class_paths.lst").read_text().split()


def test_recorded_paths_cover_the_tasks() -> None:
    assert len(PATHS) > 100


@pytest.mark.parametrize("path", PATHS)
def test_task_class_path_resolves(path: str) -> None:
    module, cls = path.split(":")
    assert hasattr(importlib.import_module(module), cls)


def test_tasks_package_all_resolves() -> None:
    import MolecularDiffusion.modules.tasks as tasks

    assert all(hasattr(tasks, n) for n in tasks.__all__)
