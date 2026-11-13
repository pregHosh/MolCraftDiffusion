"""Deeper offline tests for generation execution control flow."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.generation


class FixedNodeDist:
    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.calls = []

    def sample(self, n_samples):
        self.calls.append(n_samples)
        return torch.tensor(self.sizes[:n_samples], dtype=torch.long)


class FakeTargetFactory:
    def __init__(self):
        self.atom_vocab = None
        self.norm_factor = None
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self


class FakeSchedulerFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self


class FakeGenerationTask:
    atom_vocab = ["H", "C"]
    prop_dist_model = None
    normalize_condition = None
    predictive_model = None
    device = torch.device("cpu")

    def __init__(self):
        self.model = SimpleNamespace(T=10, norm_values=[1, 1, 1])
        self.n_node_dist = {3: 1, 4: 1, 8: 1}
        self.node_dist_model = FixedNodeDist([8, 4, 3])
        self.calls = []

    def _payload(self, nodesxsample, n_frames=0):
        batch_size = int(len(nodesxsample))
        n_nodes = int(torch.as_tensor(nodesxsample).max().item())
        n_atom_types = len(self.atom_vocab)
        if n_frames:
            one_hot = torch.zeros(n_frames, batch_size, n_nodes, n_atom_types)
            one_hot[..., 0] = 1
            charges = torch.ones(n_frames, batch_size, n_nodes, 1)
            coords = torch.zeros(n_frames, batch_size, n_nodes, 3)
            node_mask = torch.ones(batch_size, n_nodes)
        else:
            one_hot = torch.zeros(batch_size, n_nodes, n_atom_types)
            one_hot[..., 0] = 1
            charges = torch.ones(batch_size, n_nodes, 1)
            coords = torch.zeros(batch_size, n_nodes, 3)
            node_mask = torch.ones(batch_size, n_nodes)
        return one_hot, charges, coords, node_mask

    def sample(self, nodesxsample, **kwargs):
        self.calls.append(("sample", nodesxsample.clone(), kwargs))
        return self._payload(nodesxsample, n_frames=kwargs.get("n_frames", 0))

    def sample_conditonal(self, nodesxsample, target_value, **kwargs):
        self.calls.append(
            ("sample_conditonal", nodesxsample.clone(), target_value, kwargs)
        )
        return self._payload(nodesxsample, n_frames=kwargs.get("n_frames", 0))

    def sample_guidance_conitional(self, target_value, nodesxsample, **kwargs):
        self.calls.append(
            (
                "sample_guidance_conitional",
                nodesxsample.clone(),
                target_value,
                kwargs,
            )
        )
        return self._payload(nodesxsample, n_frames=kwargs.get("n_frames", 0))

    def sample_guidance(self, target_function, nodesxsample, **kwargs):
        self.calls.append(("sample_guidance", nodesxsample.clone(), target_function, kwargs))
        return self._payload(nodesxsample)

    def sample_hybrid_guidance(self, target_function, target_value, nodesxsample, **kwargs):
        self.calls.append(
            (
                "sample_hybrid_guidance",
                nodesxsample.clone(),
                target_function,
                target_value,
                kwargs,
            )
        )
        return self._payload(nodesxsample, n_frames=kwargs.get("n_frames", 0))


@pytest.fixture
def fake_xyz_io(monkeypatch):
    import MolecularDiffusion.runmodes.generate.tasks_generate as gen_tasks

    writes = []
    moves = []

    def write_xyz(output_path, one_hot, coords, atom_decoder=None, **kwargs):
        del one_hot, coords, atom_decoder, kwargs
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "molecule_000.xyz"), "w", encoding="utf-8") as f:
            f.write("1\nfake\nH 0 0 0\n")
        writes.append(("one_hot", output_path))

    def write_xyz_atomic(output_path, coords, charges, **kwargs):
        del coords, charges, kwargs
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "molecule_000.xyz"), "w", encoding="utf-8") as f:
            f.write("1\nfake\nH 0 0 0\n")
        writes.append(("atomic", output_path))

    def move_xyz(self, src_path, mol_idx, trajectory_dir=None):
        dest = os.path.join(self.output_path, f"molecule_{mol_idx:04d}.xyz")
        os.makedirs(self.output_path, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(f"1\nfinal-{mol_idx}\nH {mol_idx}.0 0.0 0.0\n")
        if os.path.exists(src_path):
            os.unlink(src_path)
        moves.append((mol_idx, trajectory_dir))
        return dest

    monkeypatch.setattr(gen_tasks, "save_xyz_file", write_xyz)
    monkeypatch.setattr(gen_tasks, "save_xyz_file_atomic_numbers", write_xyz_atomic)
    monkeypatch.setattr(gen_tasks.GenerativeFactory, "_move_xyz", move_xyz)
    return SimpleNamespace(writes=writes, moves=moves)


def test_unconditional_generation_splits_batches_clamps_node_counts_and_writes_files(
    tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type="unconditional",
        num_generate=3,
        batch_size=2,
        mol_size=[0, 0],
        max_mol_size=5,
        output_path=os.fspath(tmp_path),
    )

    factory.unconditional_generation()

    sample_calls = [call for call in task.calls if call[0] == "sample"]
    assert [call[1].tolist() for call in sample_calls] == [[5, 4], [5]]
    assert task.node_dist_model.calls == [2, 1]
    assert [move[0] for move in fake_xyz_io.moves] == [0, 1, 2]
    final_files = sorted(tmp_path.glob("molecule_*.xyz"))
    assert [path.name for path in final_files] == [
        "molecule_0000.xyz",
        "molecule_0001.xyz",
        "molecule_0002.xyz",
    ]
    for idx, path in enumerate(final_files):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "1"
        assert lines[1] == f"final-{idx}"
        assert lines[2] == f"H {idx}.0 0.0 0.0"


def test_conditional_generation_passes_target_values_and_cfg_parameters(
    tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type="cfg",
        num_generate=2,
        batch_size=2,
        mol_size=[4],
        target_values=[3.0],
        property_names=["energy"],
        negative_target_values=[-1.0],
        condition_configs={"cfg_scale": 2.5, "cfg_scale_schedule": "linear"},
        output_path=os.fspath(tmp_path),
    )

    factory.conditional_generation()

    method, nodesxsample, target_value, kwargs = task.calls[0]
    assert method == "sample_guidance_conitional"
    assert nodesxsample.tolist() == [4, 4]
    assert target_value == [3.0]
    assert kwargs["negative_target_value"] == [-1.0]
    assert kwargs["cfg_scale"] == 2.5
    assert kwargs["cfg_scale_schedule"] == "linear"
    assert kwargs["guidance_ver"] == "cfg"


def test_property_guidance_builds_target_and_scheduler_then_calls_sampler(
    tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    target_factory = FakeTargetFactory()
    scheduler_factory = FakeSchedulerFactory()
    factory = GenerativeFactory(
        task=task,
        task_type="gg",
        num_generate=1,
        batch_size=1,
        mol_size=[3],
        condition_configs={
            "target_function": target_factory,
            "scheduler": scheduler_factory,
            "gg_scale": 0.25,
            "max_norm": 0.5,
            "guidance_ver": 2,
            "guidance_at": 0.8,
            "guidance_stop": 0.1,
            "n_backwards": 4,
        },
        output_path=os.fspath(tmp_path),
    )

    factory.property_guidance()

    method, nodesxsample, target_function, kwargs = task.calls[0]
    assert method == "sample_guidance"
    assert nodesxsample.tolist() == [3]
    assert target_function is target_factory
    assert target_factory.calls == 1
    assert target_factory.atom_vocab == task.atom_vocab
    assert target_factory.norm_factor == task.model.norm_values
    assert scheduler_factory.calls == 1
    assert kwargs["scale"] == 0.25
    assert kwargs["max_norm"] == 0.5
    assert kwargs["guidance_ver"] == 2
    assert kwargs["n_backwards"] == 4


def _outpaint_factory(tmp_path, mol_size, monkeypatch, ref_natoms=4):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type="outpaint",
        num_generate=1,
        batch_size=1,
        mol_size=mol_size,
        condition_configs={
            "n_retrys": 0,
            "condition_component": "xh",
            "outpaint_cfgs": {"t_start": 0.75, "connector_dicts": {0: [1]}},
        },
        output_path=os.fspath(tmp_path),
    )
    monkeypatch.setattr(
        factory,
        "preprocess_ref_structure",
        lambda device: torch.zeros(1, ref_natoms, 6, device=device),
    )
    return task, factory


def test_structural_guidance_rejects_outpaint_size_not_exceeding_scaffold(
    monkeypatch, tmp_path, fake_xyz_io
):
    """Outpaint needs at least one atom to grow, so a target <= the scaffold is
    a config error -- it used to be silently clamped up to the scaffold size."""
    _, factory = _outpaint_factory(tmp_path, [2], monkeypatch)

    with pytest.raises(ValueError, match="nothing to grow"):
        factory.structural_guidance()


def test_structural_guidance_passes_outpaint_config(
    monkeypatch, tmp_path, fake_xyz_io
):
    task, factory = _outpaint_factory(tmp_path, [6], monkeypatch)

    factory.structural_guidance()

    method, nodesxsample, kwargs = task.calls[0]
    assert method == "sample"
    assert nodesxsample.tolist() == [6]
    assert kwargs["condition_mode"] == "outpaint_xh"
    assert kwargs["condition_tensor"].shape == (1, 4, 6)
    assert kwargs["outpaint_cfgs"] == {"t_start": 0.75, "connector_dicts": {0: [1]}}


def test_structural_guidance_passes_silvr_config(monkeypatch, tmp_path, fake_xyz_io):
    """SILVR routes through structural_guidance and carries silvr_cfgs.

    Also pins the size guardrail: a target below the reference is snapped up
    to it (the driver's `total_atoms = max(total_atoms, n_ref)`), and no
    outpaint-style "nothing to grow" error fires.
    """
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    silvr_cfgs = {"silvr_rate": 0.02, "shift_centre": True}
    factory = GenerativeFactory(
        task=task,
        task_type="silvr",
        num_generate=1,
        batch_size=1,
        mol_size=[2],
        condition_configs={"n_retrys": 3, "silvr_cfgs": silvr_cfgs},
        output_path=os.fspath(tmp_path),
    )
    monkeypatch.setattr(
        factory,
        "preprocess_ref_structure",
        lambda device: torch.zeros(1, 4, 6, device=device),
    )

    factory.run()

    method, nodesxsample, kwargs = task.calls[0]
    assert method == "sample"
    assert nodesxsample.tolist() == [4]
    assert kwargs["condition_mode"] == "silvr_xh"
    assert kwargs["condition_tensor"].shape == (1, 4, 6)
    assert kwargs["silvr_cfgs"] == silvr_cfgs
    assert kwargs["n_retrys"] == 0


def test_structural_guidance_rejects_ddim_for_silvr(monkeypatch, tmp_path):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    factory = GenerativeFactory(
        task=FakeGenerationTask(),
        task_type="silvr",
        sampling_mode="ddim",
        num_generate=1,
        batch_size=1,
        mol_size=[6],
        condition_configs={"silvr_cfgs": {}},
        output_path=os.fspath(tmp_path),
    )
    with pytest.raises(ValueError, match="ddpm"):
        factory.structural_guidance()


@pytest.mark.parametrize("task_type", ["inpaint", "outpaint", "outpaintft", "silvr"])
def test_structural_guidance_always_disables_retries(
    task_type, monkeypatch, tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type=task_type,
        num_generate=2,
        batch_size=2,
        mol_size=[5],
        condition_configs={"n_retrys": 3},
        output_path=os.fspath(tmp_path),
    )
    monkeypatch.setattr(
        factory,
        "preprocess_ref_structure",
        lambda device: torch.zeros(1, 4, 6, device=device),
    )

    factory.structural_guidance()

    _, _, kwargs = task.calls[0]
    assert kwargs["n_retrys"] == 0
    assert factory.batch_size == 2


def test_diffusion_retry_with_frames_returns_original_chain(monkeypatch):
    import MolecularDiffusion.modules.models.en_diffusion as en_diffusion

    class MinimalRetryDiffusion:
        T = 2
        COV_R = None
        condition_tensor = None
        device = torch.device("cpu")
        extra_norm_values = []
        n_dims = 3
        ndim_extra = 0
        norm_values = [1.0, 1.0, 1.0]
        num_classes = 2

        def sample_combined_position_feature_noise(
            self, n_samples, n_nodes, node_mask
        ):
            del node_mask
            return torch.zeros(n_samples, n_nodes, 6)

        def _build_outpaint_extras(
            self,
            condition_tensor,
            connector_indices,
            natom_extra,
            init_method,
            skeleton_type,
            seed_dist,
            min_dist,
            spread,
            n_bq_atom,
            bond_len,
        ):
            del (
                connector_indices,
                init_method,
                skeleton_type,
                seed_dist,
                min_dist,
                spread,
                n_bq_atom,
                bond_len,
            )
            return torch.zeros(condition_tensor.size(0), natom_extra, 6)

        def sample_p_zs_given_zt_op(
            self, s_array, t_array, z, node_mask, edge_mask, context, *args, **kwargs
        ):
            del s_array, t_array, node_mask, edge_mask, context, args, kwargs
            return z

        def unnormalize_z(self, z, node_mask):
            del node_mask
            return z

        def sample_p_xh_given_z0(
            self, z, node_mask, edge_mask, context, **kwargs
        ):
            del node_mask, edge_mask, context, kwargs
            x = z[:, :, :3]
            h = {
                "categorical": z[:, :, 3:5],
                "integer": z[:, :, 5:6].long(),
            }
            return x, h

    quality_results = iter(
        [
            (False, 2, [0, 0]),
            (True, 1, [0, 0]),
        ]
    )
    # `sample` lives in the sampling mixin; patch the name where it is looked up
    from MolecularDiffusion.modules.models.edm import en_diffusion_sampling

    monkeypatch.setattr(en_diffusion_sampling, "check_quality", lambda *args: next(quality_results))

    node_mask = torch.ones(1, 2, 1)
    edge_mask = torch.ones(4, 1)
    x, h, chain = en_diffusion.EnVariationalDiffusion.sample(
        MinimalRetryDiffusion(),
        n_samples=1,
        n_nodes=2,
        node_mask=node_mask,
        edge_mask=edge_mask,
        context=None,
        condition_tensor=torch.zeros(1, 1, 6),
        condition_mode="outpaint_xh",
        outpaint_cfgs={
            "connector_dicts": {0: [0]},
            "init_method": "seed",
        },
        n_frames=2,
        t_retry=1,
        n_retrys=1,
    )

    assert x.shape == (2, 2, 3)
    assert h["categorical"].shape == (2, 2, 2)
    assert chain.shape == (2, 1, 2, 6)
    # both scripted quality results were used: the patch really drove the retry
    assert next(quality_results, None) is None


def test_random_walk_spread_controls_skeleton_geometry():
    from MolecularDiffusion.utils.geom_constraint import build_extra_node_template

    scaffold = torch.zeros(1, 1, 6)
    connector_indices = torch.tensor([0])

    torch.manual_seed(11)
    straight = build_extra_node_template(
        scaffold,
        connector_indices,
        n_extra=4,
        skeleton_type="random_walk",
        min_dist=0,
        spread=0,
    )
    torch.manual_seed(11)
    dispersed = build_extra_node_template(
        scaffold,
        connector_indices,
        n_extra=4,
        skeleton_type="random_walk",
        min_dist=0,
        spread=1,
    )

    straight_steps = torch.diff(straight[0, :, :3], dim=0)
    assert torch.allclose(straight_steps, straight_steps[:1].expand_as(straight_steps))
    assert not torch.allclose(straight[:, :, :3], dispersed[:, :, :3])


def test_jitter_scale_must_not_fall_back_to_spread():
    from MolecularDiffusion.modules.models.en_diffusion import _resolve_jitter_scale

    with pytest.raises(ValueError, match="jitter_scale must be set explicitly"):
        _resolve_jitter_scale(
            {"spread": 9.0}, init_method="skeleton", forward_noise="jitter"
        )

    assert (
        _resolve_jitter_scale(
            {"spread": 9.0, "jitter_scale": 0.4},
            init_method="skeleton",
            forward_noise="jitter",
        )
        == 0.4
    )


def test_outpaintft_accepts_connector_indices_list_and_rejects_empty():
    import MolecularDiffusion.modules.models.en_diffusion as en_diffusion

    seen = {}

    class MinimalOutpaintFtDiffusion:
        T = 1
        COV_R = None
        condition_tensor = None
        device = torch.device("cpu")
        extra_norm_values = []
        n_dims = 3
        ndim_extra = 0
        norm_values = [1.0, 1.0, 1.0]
        num_classes = 2

        def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
            del node_mask
            return torch.zeros(n_samples, n_nodes, 6)

        def _build_outpaint_extras(
            self, condition_tensor, connector_indices, natom_extra, *args, **kwargs
        ):
            del args, kwargs
            seen["connector_indices"] = connector_indices.tolist()
            return torch.zeros(condition_tensor.size(0), natom_extra, 6)

        def sample_p_zs_given_zt_op_ft(self, s_array, t_array, z, *args, **kwargs):
            del s_array, t_array, args
            seen["t_critical"] = kwargs["t_critical"]
            return z

        def sample_p_xh_given_z0(self, z, node_mask, edge_mask, context, **kwargs):
            del node_mask, edge_mask, context, kwargs
            return z[:, :, :3], {"categorical": z[:, :, 3:5], "integer": z[:, :, 5:6].long()}

    def run(outpaint_cfgs):
        return en_diffusion.EnVariationalDiffusion.sample(
            MinimalOutpaintFtDiffusion(),
            n_samples=1,
            n_nodes=2,
            node_mask=torch.ones(1, 2, 1),
            edge_mask=torch.ones(4, 1),
            context=None,
            condition_tensor=torch.zeros(1, 1, 6),
            condition_mode="outpaintft_xh",
            outpaint_cfgs=outpaint_cfgs,
        )

    # A plain list stands in for connector_dicts on the ft path (degrees unused).
    run({"connector_indices": [0], "init_method": "seed"})
    assert seen["connector_indices"] == [0]
    assert seen["t_critical"] == 0.05

    run({"connector_indices": [0], "init_method": "seed", "t_critical": 0.0})
    assert seen["t_critical"] == 0.0

    # ...but the placement builders still need at least one connector.
    with pytest.raises(ValueError, match="connector"):
        run({"init_method": "seed"})


def test_hybrid_cfg_filters_geometric_constraint_keys_before_sampling(
    monkeypatch, tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    factory = GenerativeFactory(
        task=task,
        task_type="outpaint_cfg",
        num_generate=1,
        batch_size=1,
        mol_size=[5],
        target_values=[1.0],
        property_names=["energy"],
        condition_configs={
            "cfg_scale": 3.0,
            "condition_component": "xh",
            "outpaint_cfgs": {
                "t_start": 0.4,
                "connectors": {0: [1]},
                "constraint_strength": 2.0,
                "scale_factor": 1.2,
            },
        },
        output_path=os.fspath(tmp_path),
    )
    monkeypatch.setattr(
        factory,
        "preprocess_ref_structure",
        lambda device: torch.zeros(1, 3, 6, device=device),
    )

    factory.hybrid_guidance()

    method, nodesxsample, target_function, target_value, kwargs = task.calls[0]
    assert method == "sample_hybrid_guidance"
    assert nodesxsample.tolist() == [5]
    assert target_function is None
    assert target_value == [1.0]
    assert kwargs["guidance_ver"] == "cfg"
    assert kwargs["cfg_scale"] == 3.0
    assert kwargs["condition_mode"] == "outpaint_xh"
    # constraint_strength / scale_factor are stripped: the guided sampler applies no
    # geometric constraint. `connectors` survives the strip because it also carries WHICH
    # atoms to grow from, which this path does need — stripping it left the sampler with
    # no connectors at all and it raised on every batch.
    assert kwargs["outpaint_cfgs"] == {"t_start": 0.4, "connectors": {0: [1]}}


def test_trajectory_generation_computes_frame_schedule_and_moves_each_frame_batch(
    tmp_path, fake_xyz_io
):
    from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

    task = FakeGenerationTask()
    task.model.T = 12
    factory = GenerativeFactory(
        task=task,
        task_type="unconditional",
        num_generate=2,
        batch_size=2,
        mol_size=[3],
        n_frames=4,
        condition_configs={"denoising_strength": 0.5},
        output_path=os.fspath(tmp_path),
    )

    factory.unconditional_generation()

    assert factory.visualize_trajectory is True
    assert factory.s_saves.tolist() == [6, 4, 2, 0]
    assert [move[0] for move in fake_xyz_io.moves] == [0, 1]
    assert all(move[1] is not None for move in fake_xyz_io.moves)
