

from MolecularDiffusion.modules.tasks.task import Task
from MolecularDiffusion.modules.tasks.pharmacophore import PharmacophoreGenerative  # noqa: F401 — registers class in Registry
from MolecularDiffusion.utils.geom_utils import (
    remove_mean_with_mask,
    save_xyz_file,
    save_xyz_file_atomic_numbers,
)

import logging
import glob
import os
import re
import shutil
import tempfile
from typing import List
from tqdm import tqdm
import pandas as pd
import torch
from MolecularDiffusion.device import get_device
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph
from MolecularDiffusion.data.component.pointcloud import PointCloud_Mol
from MolecularDiffusion.data.component.feature import onehot
from ase.data import atomic_masses, atomic_numbers

logger = logging.getLogger(__name__)

# Geometric-constraint knobs the guided (*_cfg / *_gg / *_cfggg) samplers do not apply.
# The connector spec is deliberately NOT here: it also carries WHICH atoms to grow from,
# which the guided path does need. Stripping it used to leave that path with no connectors
# at all, so it raised on every batch.
_GEOM_CONSTRAINT_CFG_KEYS = {
    "constraint_strength",
    "scale_factor",
}


def _without_geometric_constraint_cfgs(cfgs):
    if not cfgs:
        return {}
    return {
        key: value
        for key, value in cfgs.items()
        if key not in _GEOM_CONSTRAINT_CFG_KEYS
    }


_XYZRENDER_API = None
_XYZRENDER_UNAVAILABLE = False
_XYZRENDER_UNAVAILABLE_WARNED = False


def _get_xyzrender_api():
    global _XYZRENDER_API, _XYZRENDER_UNAVAILABLE, _XYZRENDER_UNAVAILABLE_WARNED

    if _XYZRENDER_UNAVAILABLE:
        return None

    if _XYZRENDER_API is None:
        try:
            from xyzrender import load, render, render_gif
            _XYZRENDER_API = (load, render, render_gif)
        except ImportError:
            _XYZRENDER_UNAVAILABLE = True
            if not _XYZRENDER_UNAVAILABLE_WARNED:
                logger.warning(
                    "save_xyzrender_figures=True, but the 'xyzrender' Python package "
                    "is not installed. Skipping XYZ figure rendering."
                )
                _XYZRENDER_UNAVAILABLE_WARNED = True
            return None

    return _XYZRENDER_API


def _set_xyzrender_warning_level():
    xyzrender_logger = logging.getLogger("xyzrender")
    previous_level = xyzrender_logger.level
    xyzrender_logger.setLevel(logging.WARNING)
    return xyzrender_logger, previous_level


def _restore_logger_level(logger_obj, previous_level: int) -> None:
    logger_obj.setLevel(previous_level)


def _sanitize_xyz_text_for_render(xyz_text: str) -> str:
    """Replace unsupported XYZ element symbols with carbon for rendering only."""
    lines = xyz_text.splitlines()
    if len(lines) <= 2:
        return xyz_text

    sanitized = lines[:2]
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            symbol = parts[0]
            if symbol not in atomic_numbers or atomic_numbers[symbol] <= 0:
                parts[0] = "C"
                line = " ".join(parts)
        sanitized.append(line)
    return "\n".join(sanitized) + "\n"


def _renderable_xyz_path(xyz_path: str) -> str:
    with open(xyz_path, "r", encoding="utf-8") as f:
        xyz_text = f.read()
    sanitized_text = _sanitize_xyz_text_for_render(xyz_text)
    if sanitized_text == xyz_text:
        return xyz_path

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xyz",
        prefix="xyzrender_",
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        tmp.write(sanitized_text)
    return tmp.name


def _render_xyz_figure(xyz_path: str) -> None:
    """Render one XYZ file to SVG with the optional xyzrender Python API."""
    api = _get_xyzrender_api()
    if api is None:
        return

    load, render, _ = api
    svg_path = os.path.splitext(xyz_path)[0] + ".svg"
    xyzrender_logger, previous_level = _set_xyzrender_warning_level()
    render_input_path = None
    try:
        render_input_path = _renderable_xyz_path(xyz_path)
        mol = load(render_input_path)
        render(mol, output=svg_path)
    except Exception as exc:
        logger.warning(f"Failed to render {xyz_path} with xyzrender: {exc}")
    finally:
        _restore_logger_level(xyzrender_logger, previous_level)
        if render_input_path is not None and render_input_path != xyz_path:
            try:
                os.unlink(render_input_path)
            except OSError:
                pass


def _render_xyz_trajectory_gif(trajectory_xyz_path: str, gif_path: str) -> None:
    """Render a multi-frame XYZ trajectory to GIF with xyzrender."""
    api = _get_xyzrender_api()
    if api is None:
        return

    _, _, render_gif = api
    xyzrender_logger, previous_level = _set_xyzrender_warning_level()
    try:
        render_gif(trajectory_xyz_path, gif_trj=True, output=gif_path)
    except Exception as exc:
        logger.warning(f"Failed to render denoising GIF {gif_path} with xyzrender: {exc}")
    finally:
        _restore_logger_level(xyzrender_logger, previous_level)


class GenerativeFactory:
    def __init__(self,
                 task: Task,
                 task_type: str = "unconditional",
                 sampling_mode: str = "ddpm",
                 num_generate: int = 100,
                 mol_size: List[int] = [0,0],
                 target_values: List[float] = [],
                 property_names: List[str] = [],
                 negative_target_values: List[float] = [],
                 batch_size: int = 1,
                 seed: int = 86,
                 n_frames: int = 0,
                 output_path: str = "generated_mol",
                 save_xyzrender_figures: bool = False,
                 condition_configs={},
                 max_mol_size: int = 0,
                 sdf_per_molecule: bool = False,
                 **kwargs,
    ):

        self.task = task
        # Opt-in: write one molecule_XXXX.sdf beside each molecule_XXXX.xyz.
        # Independent of the task's combined sdf_output_path sidecar -- both,
        # either, or neither may be enabled. No-op for models that do not
        # generate bonds (they never set task.last_bond_types).
        self.sdf_per_molecule = sdf_per_molecule
        self.task_type = task_type
        self.num_generate = num_generate
        self.max_mol_size = max_mol_size
        try:
            self.max_atom = max(task.n_node_dist.keys())
        except (ValueError, AttributeError):
            logger.warning("Node distribution model is not available, set max_atom to 86")
            self.max_atom = 86

            
        self.mol_size = mol_size

        if self.mol_size is not None and len(self.mol_size) > 0:
            if self.mol_size[-1] > self.max_atom:
                logger.info(
                    "Specified molecular size is larger than the largest molecules in the training data, reset...")
                self.mol_size[-1] = self.max_atom
            if self.mol_size[0] > self.max_atom:
                logger.info(
                    "Specified molecular size is larger than the largest molecules in the training data, reset...")
                self.mol_size[0] = self.max_atom
            if len(self.mol_size) > 1 and self.mol_size[0] > self.mol_size[-1]:
                raise ValueError(
                    f"Invalid mol_size range after clamping to the model's "
                    f"training distribution (max={self.max_atom}): "
                    f"mol_size[0]={self.mol_size[0]} > mol_size[-1]="
                    f"{self.mol_size[-1]}. The model's n_node_dist may "
                    f"represent a smaller quantity than a full molecule "
                    f"(e.g. a linker-size histogram) — configure mol_size "
                    f"within that model's actual range."
                )

        self.target_values = target_values
        self.property_names = property_names
        self.negative_target_values = negative_target_values

        if len(self.target_values) != len(self.property_names):
            logger.warning("Number of target values must match with number of property names")
            self.property_names = ["a"]*len(self.target_values)

        # Validate number of target_values against model's condition_names (from checkpoint)
        if len(self.target_values) > 0:
            model_conditions = getattr(task, "condition", None)
            if model_conditions is not None and len(model_conditions) > 0:
                n_model = len(model_conditions)
                n_requested = len(self.target_values)
                if n_requested != n_model:
                    raise ValueError(
                        f"Property count mismatch: {n_requested} target value(s) specified "
                        f"but the model was trained with {n_model} condition(s) "
                        f"({list(model_conditions)}). "
                        f"Update 'target_values' in your config to match."
                    )
        
        self.batch_size = batch_size
        self.seed = seed
        self.n_frames = n_frames

        # Batch outcome tracking. Every sampling loop swallows its own exceptions so one
        # bad batch cannot kill a long run; these let _assert_produced() tell "some batches
        # failed" from "nothing worked at all". Set here, not in run(), because the
        # generation methods are also called directly.
        self._num_round = 0
        self._fail_count = 0
        self._first_exc = None

        if n_frames > 0:
            self.visualize_trajectory = True
            
            # Use T for DDPM/LDM, fm_num_timesteps for FM
            total_steps = getattr(self.task.model, 'T', getattr(self.task.model, 'fm_num_timesteps', 100))
            
            if condition_configs.get("denoising_strength", 0) > 0:
                t_start = int(total_steps * condition_configs.get("denoising_strength", 0.8))
            elif condition_configs.get("t_start", 1) < 1:
                t_start = int(total_steps * condition_configs.get("t_start", 1))
            else:
                t_start = int(total_steps)
                
            s_saves = torch.linspace(0, t_start, 
                                            steps=self.n_frames).long()
            self.s_saves = s_saves.flip(0)
            logger.warning(f"Frames will be saved at timesteps: {self.s_saves.tolist()}")
        else:
            self.visualize_trajectory = False
            
        self.output_path = output_path
        self.save_xyzrender_figures = save_xyzrender_figures
        
        self.sampling_mode = sampling_mode # ddim not available for CFG and GG

        self.condition_configs = condition_configs
        if "use_noised_conditioning" in kwargs:
            self.condition_configs["use_noised_conditioning"] = kwargs["use_noised_conditioning"]

        if self.task.node_dist_model is None:
            logger.warning("Number of atoms distribution is not available, specify the size of molecules to generate")
            import random
            if len(self.mol_size) == 2:
                if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                    self.mol_size = [random.randint(14, 100)]

    def _write_molecule_sdf(self, mol_idx: int, j: int, one_hot, charges, x, node_mask) -> None:
        """Write ``molecule_<mol_idx>.sdf`` next to the matching ``.xyz``.

        Bond-generating tasks stash their dense (B,N,N) bond matrix on
        ``task.last_bond_types`` (MiDi, FlowMol, LoQI, JODO). The ``.xyz``
        writer has no bond channel, so without this the 2D half of those
        models is dropped at write time. Pairing is by construction: the
        caller passes the same ``mol_idx`` it just gave ``_move_xyz``, and
        ``j`` is that molecule's row in the current batch -- so the numbering
        cannot drift the way a task-side counter would when a batch fails or
        comes up short.

        Silent no-op when the task generates no bonds. Chemistry failures are
        warned and skipped: one unsanitizable sample must not kill a run.
        """
        bonds = getattr(self.task, "last_bond_types", None)
        if bonds is None:
            return
        try:
            from rdkit import Chem

            from MolecularDiffusion.data.component.graph3d_dataset import (
                build_rdkit_mol,
            )
            from ase.data import atomic_numbers as _atomic_numbers
        except ImportError:
            logger.warning("RDKit unavailable, skipping per-molecule .sdf")
            return

        vocab = getattr(self.task, "atom_vocab", None)
        if vocab is None:
            return
        z_of_vocab = [_atomic_numbers[sym] for sym in vocab]

        mask_j = node_mask[j].reshape(-1).cpu()
        n = int(mask_j.sum())
        if n == 0:
            return
        atom_idx = one_hot[j].argmax(dim=-1).cpu()
        sub = bonds[j, :n, :n].cpu()
        rows, cols = torch.triu_indices(n, n, offset=1)
        keep = sub[rows, cols] > 0
        bond_index = torch.stack((rows[keep], cols[keep])).numpy()
        bond_type = sub[rows, cols][keep].numpy()
        zs = [z_of_vocab[int(i)] for i in atom_idx[:n]]

        path = os.path.join(self.output_path, f"molecule_{str(mol_idx).zfill(4)}.sdf")
        try:
            mol = build_rdkit_mol(
                zs,
                bond_index,
                bond_type,
                formal_charge=charges[j, :n].reshape(-1).cpu().numpy(),
                coords=x[j, :n].cpu().numpy(),
            )
            writer = Chem.SDWriter(path)
            writer.write(mol)
            writer.close()
        except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
            logger.warning("Skipping unsanitizable sample %d: %s", mol_idx, exc)

    def _warn_if_bonds_dropped(self) -> None:
        """Warn once when a bond-generating task is writing no bonds anywhere.

        `.xyz` has no bond channel, so a task that stashed `last_bond_types`
        with neither `tasks.sdf_output_path` nor `interference.sdf_per_molecule`
        set is discarding half its output. That used to happen silently when a
        bundled example config was run from outside the repo. One warning, only
        for tasks that actually generate bonds -- bondless models never hit it.
        """
        if getattr(self, "_warned_bonds_dropped", False):
            return
        if getattr(self.task, "last_bond_types", None) is None:
            return
        if self.sdf_per_molecule or getattr(self.task, "sdf_output_path", None):
            return
        self._warned_bonds_dropped = True
        logger.warning(
            "%s generated bond orders but neither tasks.sdf_output_path nor "
            "interference.sdf_per_molecule is set -- only .xyz is being "
            "written, so the bonds are discarded. Set one of them to keep "
            "them.",
            type(self.task).__name__,
        )

    def _move_xyz(self, src_path: str, mol_idx: int, trajectory_dir: str = None) -> str:
        self._warn_if_bonds_dropped()
        dest_path = os.path.join(self.output_path, f"molecule_{str(mol_idx).zfill(4)}.xyz")
        shutil.move(src_path, dest_path)
        if self.save_xyzrender_figures:
            _render_xyz_figure(dest_path)
            if trajectory_dir is not None:
                self._render_denoising_gif(trajectory_dir, dest_path)
        return dest_path

    def _render_denoising_gif(self, trajectory_dir: str, final_xyz_path: str) -> None:
        frame_paths = sorted(
            (
                path
                for path in glob.glob(os.path.join(trajectory_dir, "molecule_*.xyz"))
                if os.path.abspath(path) != os.path.abspath(final_xyz_path)
            ),
            key=self._frame_sort_key,
            reverse=True,
        )
        frame_paths.append(final_xyz_path)
        if len(frame_paths) < 2:
            logger.warning(f"Not enough denoising frames in {trajectory_dir} to render a GIF.")
            return

        atom_counts = []
        for frame_path in frame_paths:
            try:
                with open(frame_path, "r", encoding="utf-8") as f:
                    atom_counts.append(int(f.readline().strip()))
            except (OSError, ValueError) as exc:
                logger.warning(f"Skipping denoising GIF for {trajectory_dir}: invalid XYZ frame {frame_path}: {exc}")
                return

        if len(set(atom_counts)) != 1:
            logger.warning(
                f"Skipping denoising GIF for {trajectory_dir}: xyzrender requires all frames "
                f"to have the same atom count, got {sorted(set(atom_counts))}."
            )
            return

        trajectory_xyz_path = os.path.join(trajectory_dir, "denoising_trajectory.xyz")
        with open(trajectory_xyz_path, "w", encoding="utf-8") as out:
            for frame_path in frame_paths:
                with open(frame_path, "r", encoding="utf-8") as f:
                    out.write(_sanitize_xyz_text_for_render(f.read()).rstrip())
                    out.write("\n")

        _render_xyz_trajectory_gif(
            trajectory_xyz_path,
            os.path.join(trajectory_dir, "denoising.gif"),
        )

    @staticmethod
    def _frame_sort_key(path: str) -> int:
        match = re.search(r"molecule_(\d+)\.xyz$", os.path.basename(path))
        return int(match.group(1)) if match else -1
        
    def _assert_produced(self):
        """Raise if every sampling batch failed.

        Without this the run writes zero molecules and still exits 0, which no caller can
        tell from success. A config error (bad connector shape, wrong mol_size, missing
        reference file) fails identically on every batch, so all-failed is the signature.
        """
        if self._num_round and self._fail_count >= self._num_round:
            raise RuntimeError(
                f"All {self._num_round} sampling batches failed; no molecules were written. "
                f"First failure was {type(self._first_exc).__name__}: {self._first_exc}"
            ) from self._first_exc

    def run(self):

        if self.task_type == "unconditional":
            self.unconditional_generation()
        elif self.task_type in ("conditional", "cfg"):
            self.conditional_generation()
        elif self.task_type in ("gradient_guidance", "gg", "cfggg"):
            self.property_guidance()
        elif self.task_type in ("inpaint", "outpaint", "outpaintft", "silvr"):
            self.structural_guidance()
        elif self.task_type in {"inpaint_cfg", "inpaint_gg", "inpaint_cfggg", "outpaint_cfg", "outpaint_gg", "outpaint_cfggg"}:
            self.hybrid_guidance()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self._assert_produced()

    def unconditional_generation(self):
            
        fail_count = 0
        
        num_round = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size != 0:
            num_round += 1
        current_batch_size = self.batch_size
        
        progress_bar = tqdm(range(num_round), desc="Sampling molecules", leave=True)
        self._num_round = num_round
        
        for i in progress_bar:
            if i == num_round-1 and self.num_generate % self.batch_size != 0:
                current_batch_size = self.num_generate % self.batch_size
            else:
                current_batch_size = self.batch_size
                
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                    nodesxsample = nodesxsample.repeat(current_batch_size) 
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(current_batch_size)
                        if self.max_mol_size > 0:
                            nodesxsample = torch.clamp(nodesxsample, max=self.max_mol_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])
                        nodesxsample = nodesxsample.repeat(current_batch_size) 
                if self.task.prop_dist_model and len(self.target_values) == 0:
                    size = nodesxsample[0].item()
                    target_value = self.task.prop_dist_model.sample(size)
                    one_hot, charges, x, node_mask = self.task.sample_conditonal(
                        nodesxsample=nodesxsample, 
                        target_value=target_value,
                        mode=self.sampling_mode,
                        n_frames=self.n_frames
                    )
                else:
                    one_hot, charges, x, node_mask = self.task.sample(
                        nodesxsample=nodesxsample,
                        mode=self.sampling_mode,
                        n_frames=self.n_frames
                    )
            
                if self.visualize_trajectory:
                    for j in range(current_batch_size):
                        mol_idx = i * self.batch_size + j
                        output_path_frame = os.path.join(self.output_path, f"mol_{mol_idx}")
                        if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                            save_xyz_file_atomic_numbers(
                                output_path_frame,
                                x[:, j],
                                charges[:, j].squeeze(-1),
                                idxs=self.s_saves.tolist(),
                                node_mask=node_mask[j].unsqueeze(0).expand(one_hot[:, j].size(0), -1, -1) if node_mask is not None else None,
                            )
                        else:
                            save_xyz_file(
                                output_path_frame,
                                one_hot[:, j],
                                x[:, j],
                                atom_decoder=self.task.atom_vocab,
                                idxs=self.s_saves.tolist(),
                                atomic_numbers=charges[:, j].squeeze(-1) if hasattr(self.task.model, 'use_unknown_fallback') else None,
                                use_unknown_fallback=getattr(self.task.model, 'use_unknown_fallback', False),
                                node_mask=node_mask[j].unsqueeze(0).expand(one_hot[:, j].size(0), -1, -1) if node_mask is not None else None,
                            )   
                        path_xyz = os.path.join(output_path_frame, f"molecule_000.xyz")
                        idx = i * self.batch_size + j
                        self._move_xyz(path_xyz, idx, trajectory_dir=output_path_frame)
                        if self.sdf_per_molecule:
                            self._write_molecule_sdf(idx, j, one_hot, charges, x, node_mask)
                else:
                    if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                        save_xyz_file_atomic_numbers(
                            self.output_path,
                            x,
                            charges.squeeze(-1),
                            node_mask=node_mask,
                        )
                    else:
                        save_xyz_file(
                            self.output_path,
                            one_hot,
                            x,
                            atom_decoder=self.task.atom_vocab,
                            atomic_numbers=charges.squeeze(-1) if hasattr(self.task.model, 'use_unknown_fallback') else None,
                            use_unknown_fallback=getattr(self.task.model, 'use_unknown_fallback', False),
                            node_mask=node_mask,
                        )

                    for j in range(current_batch_size):
                        
                        path_xyz = os.path.join(self.output_path, f"molecule_{str(j).zfill(3)}.xyz")
                        idx = i * self.batch_size + j
                        self._move_xyz(path_xyz, idx)
                        if self.sdf_per_molecule:
                            self._write_molecule_sdf(idx, j, one_hot, charges, x, node_mask)
            except Exception as e:
                fail_count += 1
                self._fail_count += 1
                if self._first_exc is None:
                    self._first_exc = e
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })
    
    def conditional_generation(self):
        assert len(self.target_values) > 0, "Target values must be provided for conditional generation"
        
        if hasattr(self.task, 'predictive_model'):
            property_eval = True
            df_dict = {
                "filename": [],
            }
            for prop_name in self.property_names:
                df_dict[prop_name] = []
            df_dict["size"] = []           
        else:
            logger.warning("Property model is not available, skip evaluation.")
            property_eval = False        
        
        fail_count = 0
        num_round = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size != 0:
            num_round += 1
        current_batch_size = self.batch_size
        progress_bar = tqdm(range(num_round), desc="Sampling molecules", leave=True)
        self._num_round = num_round
        
        for i in progress_bar:

            if i == num_round-1 and self.num_generate % self.batch_size != 0:
                current_batch_size = self.num_generate % self.batch_size
            else:
                current_batch_size = self.batch_size
                
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                    nodesxsample = nodesxsample.repeat(current_batch_size) 
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(current_batch_size)
                        if self.max_mol_size > 0:
                            nodesxsample = torch.clamp(nodesxsample, max=self.max_mol_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])
                        nodesxsample = nodesxsample.repeat(current_batch_size) 
              
                if self.task_type == "conditional":
                    one_hot, charges, x, node_mask = self.task.sample_conditonal(
                            nodesxsample=nodesxsample, 
                            target_value=self.target_values,
                            n_frames=self.n_frames,
                            mode=self.sampling_mode
                        )
                elif self.task_type == "cfg":
                    one_hot, charges, x, node_mask = self.task.sample_guidance_conitional(
                            target_function=None,
                            target_value=self.target_values,
                            negative_target_value=self.negative_target_values,
                            nodesxsample=nodesxsample, 
                            cfg_scale=self.condition_configs.get("cfg_scale",1),
                            cfg_scale_schedule=self.condition_configs.get("cfg_scale_schedule", None),
                            guidance_ver="cfg",
                            n_frames=self.n_frames
                        )

                if self.visualize_trajectory:
                    for j in range(current_batch_size):
                        mol_idx = i * self.batch_size + j
                        output_path_frame = os.path.join(self.output_path, f"mol_{mol_idx}")
                        if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                            save_xyz_file_atomic_numbers(
                                output_path_frame,
                                x[:, j],
                                charges[:, j].squeeze(-1) if charges is not None else None,
                                idxs=self.s_saves.tolist(),
                                node_mask=node_mask[j].unsqueeze(0).expand(one_hot[:, j].size(0), -1, -1) if node_mask is not None else None,
                            )
                        else:
                            save_xyz_file(
                                output_path_frame,
                                one_hot[:, j],
                                x[:, j],
                                atom_decoder=self.task.atom_vocab,
                                idxs=self.s_saves.tolist(),
                                atomic_numbers=charges[:, j].squeeze(-1) if charges is not None and hasattr(self.task.model, 'use_unknown_fallback') else None,
                                use_unknown_fallback=getattr(self.task.model, 'use_unknown_fallback', False),
                                node_mask=node_mask[j].unsqueeze(0).expand(one_hot[:, j].size(0), -1, -1) if node_mask is not None else None,
                            )     
                        path_xyz = os.path.join(output_path_frame, f"molecule_000.xyz")
                        idx = i * self.batch_size + j
                        self._move_xyz(path_xyz, idx, trajectory_dir=output_path_frame)
                    x = x[:, -1]
                    one_hot = one_hot[:, -1]   
                else:
                    if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                        save_xyz_file_atomic_numbers(
                            self.output_path,
                            x,
                            charges.squeeze(-1) if charges is not None else None,
                            node_mask=node_mask,
                        )
                    else:
                        save_xyz_file(
                            self.output_path,
                            one_hot,
                            x,
                            atom_decoder=self.task.atom_vocab,
                            atomic_numbers=charges.squeeze(-1) if charges is not None and hasattr(self.task.model, 'use_unknown_fallback') else None,
                            use_unknown_fallback=getattr(self.task.model, 'use_unknown_fallback', False),
                            node_mask=node_mask,
                        )

                    for j in range(current_batch_size):
                        
                        path_xyz = os.path.join(self.output_path, f"molecule_{str(j).zfill(3)}.xyz")
                        idx = i * self.batch_size + j
                        self._move_xyz(path_xyz, idx)
                
                #TODO to adapt this to work with batch of mols
                if property_eval:
                    xh = torch.cat([
                        x,
                        one_hot,
                        charges
                    ])
                    preds = self.property_prediction(xh, t=0)
                    for prop_name in self.property_names:
                        logger.info(f"{prop_name}: {preds[prop_name]}")
                        df_dict[prop_name].append(preds[prop_name])
                    df_dict["filename"].append(f"molecule_{str(i+1).zfill(4)}.xyz")
                    df_dict["size"].append(nodesxsample.item())
            except Exception as e:
                fail_count += 1
                self._fail_count += 1
                if self._first_exc is None:
                    self._first_exc = e
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })
        if property_eval:    
            self.df = pd.DataFrame(df_dict)
    
    
    def property_guidance(self):
        
        target_function=self.condition_configs.get("target_function", None)
        target_function.atom_vocab = self.task.atom_vocab  

        target_function.norm_factor = self.task.model.norm_values
        target_function = target_function()
        scheduler = self.condition_configs.get("scheduler", None)
        if scheduler is not None:
            scheduler = scheduler()
            
        fail_count = 0
        
        # Batch generation setup (matching unconditional_generation pattern)
        num_round = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size != 0:
            num_round += 1
        current_batch_size = self.batch_size
        
        progress_bar = tqdm(range(num_round), desc="Sampling molecules", leave=True)
        self._num_round = num_round
        
        if hasattr(self.task, 'predictive_model'):
            property_eval = True
            df_dict = {
                "filename": [],
            }
            for prop_name in self.property_names:
                df_dict[prop_name] = []
            df_dict["size"] = []           
        else:
            logger.warning("Property model is not available, skip evaluation.")
            property_eval = False
             
        for i in progress_bar:
            # Adjust batch size for last incomplete batch
            if i == num_round - 1 and self.num_generate % self.batch_size != 0:
                current_batch_size = self.num_generate % self.batch_size
            else:
                current_batch_size = self.batch_size
                
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                    nodesxsample = nodesxsample.repeat(current_batch_size)
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(current_batch_size)
                        if self.max_mol_size > 0:
                            nodesxsample = torch.clamp(nodesxsample, max=self.max_mol_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])
                        nodesxsample = nodesxsample.repeat(current_batch_size)
                
                if len(self.target_values) == 0:
                    one_hot, charges, x, node_mask  = self.task.sample_guidance(
                        target_function=target_function,
                        nodesxsample=nodesxsample,
                        scale=self.condition_configs.get("gg_scale",1),
                        max_norm=self.condition_configs.get("max_norm",1),
                        std=1,
                        scheduler=scheduler,
                        guidance_ver=self.condition_configs.get("guidance_ver",1),
                        guidance_at=self.condition_configs.get("guidance_at",1),
                        guidance_stop=self.condition_configs.get("guidance_stop",0),
                        n_backwards=self.condition_configs.get("n_backwards",1)
                    )              
                else:
                    
                    one_hot, charges, x, node_mask  = self.task.sample_guidance_conitional(
                        target_function=target_function,
                        target_value=self.target_values,
                        negative_target_value=self.negative_target_values,
                        nodesxsample=nodesxsample,
                        gg_scale=self.condition_configs.get("gg_scale",1),
                        cfg_scale=self.condition_configs.get("cfg_scale",1),
                        max_norm=self.condition_configs.get("max_norm",1),
                        std=1,
                        scheduler=scheduler,
                        guidance_ver=self.condition_configs.get("guidance_ver",1),
                        guidance_at=self.condition_configs.get("guidance_at",1),
                        guidance_stop=self.condition_configs.get("guidance_stop",0),
                        n_backwards=self.condition_configs.get("n_backwards",1)
                    )   
                    
                if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                    save_xyz_file_atomic_numbers(
                        self.output_path,
                        x,
                        charges.squeeze(-1) if charges is not None else None,
                        node_mask=node_mask,
                    )
                else:
                    save_xyz_file(
                        self.output_path,
                        one_hot,
                        x,
                        atom_decoder=self.task.atom_vocab,
                        atomic_numbers=charges.squeeze(-1) if charges is not None and hasattr(self.task.model, 'use_unknown_fallback') else None,
                        use_unknown_fallback=getattr(self.task.model, 'use_unknown_fallback', False),
                        node_mask=node_mask,
                    )

                # Rename files for batch: molecule_000.xyz, molecule_001.xyz, ... -> molecule_XXXX.xyz
                for j in range(current_batch_size):
                    path_xyz = os.path.join(self.output_path, f"molecule_{str(j).zfill(3)}.xyz")
                    idx = i * self.batch_size + j
                    self._move_xyz(path_xyz, idx)
                
                # Property evaluation (TODO: adapt for batch)
                if property_eval:
                    for j in range(current_batch_size):
                        idx = i * self.batch_size + j
                        df_dict["filename"].append(f"molecule_{str(idx).zfill(4)}.xyz")
                        df_dict["size"].append(nodesxsample[j].item() if nodesxsample.dim() > 0 else nodesxsample.item())
                        # TODO: add per-molecule property prediction
                        for prop_name in self.property_names:
                            df_dict[prop_name].append(None)  # Placeholder
                            
            except Exception as e:
                fail_count += 1
                self._fail_count += 1
                if self._first_exc is None:
                    self._first_exc = e
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": (i + 1) * self.batch_size,
                "failed": fail_count,
                "success": ((i + 1) * self.batch_size - fail_count * self.batch_size),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })
            
        if property_eval:    
            self.df = pd.DataFrame(df_dict)
    

    # Max node-distribution redraws before giving up on an outpaint [0,0] batch.
    _SIZE_RESAMPLE_TRIES = 20

    def _warn_size_once(self, msg):
        """Emit a size-guardrail warning once per run, visibly (tqdm.write goes to
        stdout so it surfaces in the MCP job log, not just the logger)."""
        if not getattr(self, "_size_warned", False):
            self._size_warned = True
            logger.warning(msg)
            tqdm.write(msg)

    def _validate_size_config(self, ref_natoms):
        """Fail fast, before the sampling loop, when outpaint can never fit the
        scaffold. Inpaint is never fatal here — it recovers by clamping per-sample.

        Outpaint needs a target strictly larger than the scaffold (>=1 atom to
        grow), so a target <= scaffold is meaningless:
          - explicit mol_size whose maximum is <= scaffold -> config error, raise;
          - mol_size=[0,0] (randomized) -> only fatal if the node distribution's
            largest molecule still can't exceed the scaffold (otherwise low draws
            are resampled per-batch by _enforce_scaffold_size).
        """
        if "outpaint" not in self.task_type:
            return
        floor = ref_natoms + 1
        ms = self.mol_size
        randomized = len(ms) == 2 and ms[0] == 0 and ms[1] == 0
        if randomized:
            if self.max_atom < floor:
                raise ValueError(
                    f"Outpaint with mol_size=[0,0]: the model's node-size "
                    f"distribution tops out at {self.max_atom} atoms, which cannot "
                    f"exceed the {ref_natoms}-atom scaffold — nothing can be grown. "
                    f"Set an explicit mol_size:[lo,hi] with lo > {ref_natoms}, or "
                    f"use a smaller scaffold."
                )
            return
        if ms and ms[-1] < floor:
            raise ValueError(
                f"Outpaint: mol_size {list(ms)} is <= the scaffold "
                f"({ref_natoms} atoms), so there is nothing to grow. Set mol_size "
                f"with a maximum > {ref_natoms}."
            )

    def _enforce_scaffold_size(self, nodesxsample, ref_natoms):
        """Per-sample size guardrail for structure-guided generation.

        - inpaint / silvr: a target below the reference is snapped up to the
          reference size (both keep every reference atom present -- inpaint
          regenerates masked atoms in place, silvr guides them -- so the size is
          derived, not requested), with a warning.
        - outpaint: the target must exceed the scaffold. Randomized [0,0] draws that
          land <= scaffold are resampled from the node distribution (up to
          _SIZE_RESAMPLE_TRIES) rather than killing the run or clamping to a
          degenerate 1-atom growth; explicit too-small sizes are already killed
          upfront by _validate_size_config.
        """
        if "outpaint" not in self.task_type:
            floor = ref_natoms
            n_below = int((nodesxsample < floor).sum())
            if n_below:
                nodesxsample = torch.clamp(nodesxsample, min=floor)
                self._warn_size_once(
                    f"[size guardrail] {n_below}/{nodesxsample.numel()} "
                    f"{self.task_type} size(s) were below the scaffold "
                    f"({floor} atoms) and were "
                    f"snapped up to the scaffold size."
                )
            return nodesxsample

        floor = ref_natoms + 1
        bad = nodesxsample < floor
        if not bool(bad.any()):
            return nodesxsample

        randomized = len(self.mol_size) == 2 and self.mol_size[0] == 0 and self.mol_size[1] == 0
        if not randomized:
            # Should have been caught upfront; belt-and-suspenders.
            raise ValueError(
                f"Outpaint: target size {nodesxsample.tolist()} is <= the scaffold "
                f"({ref_natoms} atoms), so there is nothing to grow. Set mol_size "
                f"with a maximum > {ref_natoms}."
            )

        n_below = int(bad.sum())
        for _ in range(self._SIZE_RESAMPLE_TRIES):
            redraw = self.task.node_dist_model.sample(int(bad.sum()))
            if self.max_mol_size > 0:
                redraw = torch.clamp(redraw, max=self.max_mol_size)
            nodesxsample[bad] = redraw.to(dtype=nodesxsample.dtype, device=nodesxsample.device)
            bad = nodesxsample < floor
            if not bool(bad.any()):
                break
        if bool(bad.any()):
            raise ValueError(
                f"Outpaint: could not draw a size above the {ref_natoms}-atom "
                f"scaffold in {self._SIZE_RESAMPLE_TRIES} tries "
                f"(max_mol_size={self.max_mol_size}). Set an explicit "
                f"mol_size:[lo,hi] with lo > {ref_natoms}."
            )
        self._warn_size_once(
            f"[size guardrail] {n_below} randomized outpaint size(s) landed <= the "
            f"{ref_natoms}-atom scaffold and were resampled to grow the scaffold."
        )
        return nodesxsample

    def structural_guidance(self):

        # SILVR modifies the DDPM reverse loop only; sample_ddim is a separate
        # loop with no SILVR step, so fail loudly instead of silently ignoring
        # the requested mode.
        if self.task_type == "silvr" and self.sampling_mode != "ddpm":
            raise ValueError(
                f"silvr supports sampling_mode 'ddpm' only, got "
                f"'{self.sampling_mode}'."
            )

        # get condition structure
        xh_ref = self.preprocess_ref_structure(self.task.device)
        self._validate_size_config(xh_ref.shape[1])

        requested_n_retrys = self.condition_configs.get("n_retrys", 0) or 0
        n_retrys = 0
        if requested_n_retrys != 0:
            logger.warning(
                "Structure-guided retries are disabled; forcing n_retrys=0 "
                f"(requested {requested_n_retrys})."
            )

        # process condition values
        if len(self.target_values) > 0 and self.task.prop_dist_model is not None:

            context = []
            for i, key in enumerate(self.task.prop_dist_model.distributions):

                if self.task.normalize_condition == "mad":
                    mean, mad = (
                        self.task.prop_dist_model.normalizer[key]["mean"],
                        self.task.prop_dist_model.normalizer[key]["mad"],
                    )
                    val = (self.target_values[i] - mean) / (mad)
                elif self.task.normalize_condition == "maxmin":
                    mean, min, max = (
                        self.task.prop_dist_model.normalizer[key]["mean"],
                        self.task.prop_dist_model.normalizer[key]["min"],
                        self.task.prop_dist_model.normalizer[key]["max"],
                    )
                    val = 2 * (self.target_values[i] - min) / (max - min) - 1
                else:
                    val = self.target_values[i]
                context_row = torch.tensor(
                    [val]
                ).unsqueeze(1)
                context.append(context_row)
            context = torch.cat(context, dim=1).float().to(self.task.device)

        else:
            context = None

        fail_count = 0
        num_round = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size != 0:
            num_round += 1
        current_batch_size = self.batch_size

        progress_bar = tqdm(range(num_round), desc="Sampling molecules", leave=True)
        self._num_round = num_round

        condition_mode = self.task_type + "_" + self.condition_configs.get("condition_component", "xh")

        for i in progress_bar:
            if i == num_round - 1 and self.num_generate % self.batch_size != 0:
                current_batch_size = self.num_generate % self.batch_size
            else:
                current_batch_size = self.batch_size

            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                    nodesxsample = nodesxsample.repeat(current_batch_size)
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(current_batch_size)
                        if self.max_mol_size > 0:
                            nodesxsample = torch.clamp(nodesxsample, max=self.max_mol_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])
                        nodesxsample = nodesxsample.repeat(current_batch_size)

                nodesxsample = self._enforce_scaffold_size(nodesxsample, xh_ref.shape[1])

                xh_tensor = xh_ref.repeat(current_batch_size, 1, 1)

                if self.task_type == "inpaint":
                    one_hot, charges, x, node_mask = self.task.sample(
                        nodesxsample,
                        condition_tensor=xh_tensor,
                        condition_mode=condition_mode,
                        inpaint_cfgs=self.condition_configs.get("inpaint_cfgs", {}),
                        use_noised_conditioning=self.condition_configs.get("use_noised_conditioning", False),
                        n_frames=self.n_frames,
                        n_retrys=n_retrys,
                        t_retry=self.condition_configs.get("t_retry"),
                        context=context,
                    )

                elif self.task_type == "outpaint":
                    one_hot, charges, x, node_mask = self.task.sample(
                        nodesxsample,
                        condition_tensor=xh_tensor,
                        condition_mode=condition_mode,
                        outpaint_cfgs=self.condition_configs.get("outpaint_cfgs", {}),
                        use_noised_conditioning=self.condition_configs.get("use_noised_conditioning", False),
                        n_frames=self.n_frames,
                        n_retrys=n_retrys,
                        t_retry=self.condition_configs.get("t_retry"),
                        context=context,
                    )
                elif self.task_type == "outpaintft":
                    one_hot, charges, x, node_mask = self.task.sample(
                        nodesxsample,
                        condition_tensor=xh_tensor,
                        condition_mode=condition_mode,
                        outpaint_cfgs=self.condition_configs.get("outpaint_cfgs", {}),
                        use_noised_conditioning=self.condition_configs.get("use_noised_conditioning", False),
                        n_frames=self.n_frames,
                        n_retrys=n_retrys,
                        t_retry=self.condition_configs.get("t_retry"),
                        context=context,
                    )
                elif self.task_type == "silvr":
                    # No post-hoc reference-frame restore here: the model's
                    # reverse loop already adds total_gravity_center_shift back
                    # (gated by silvr_cfgs.shift_centre). Doing it here too
                    # would double-shift.
                    one_hot, charges, x, node_mask = self.task.sample(
                        nodesxsample,
                        condition_tensor=xh_tensor,
                        condition_mode=condition_mode,
                        silvr_cfgs=self.condition_configs.get("silvr_cfgs", {}),
                        n_frames=self.n_frames,
                        n_retrys=n_retrys,
                        t_retry=self.condition_configs.get("t_retry"),
                        context=context,
                    )

                if self.visualize_trajectory:
                    for j in range(current_batch_size):
                        mol_idx = i * self.batch_size + j
                        output_path_frame = os.path.join(self.output_path, f"mol_{mol_idx}")
                        if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                            save_xyz_file_atomic_numbers(
                                output_path_frame,
                                x[:, j],
                                charges[:, j].squeeze(-1) if charges is not None else None,
                                idxs=self.s_saves.tolist(),
                                node_mask=node_mask[j].unsqueeze(0).expand(one_hot[:, j].size(0), -1, -1) if node_mask is not None else None,
                            )
                        else:
                            save_xyz_file(
                                output_path_frame,
                                one_hot[:, j],
                                x[:, j],
                                atom_decoder=self.task.atom_vocab,
                                idxs=self.s_saves.tolist()
                            )
                        path_xyz = os.path.join(output_path_frame, f"molecule_000.xyz")
                        self._move_xyz(path_xyz, mol_idx, trajectory_dir=output_path_frame)
                else:
                    if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                        save_xyz_file_atomic_numbers(
                            self.output_path,
                            x,
                            charges.squeeze(-1) if charges is not None else None,
                            node_mask=node_mask,
                        )
                    else:
                        save_xyz_file(
                            self.output_path,
                            one_hot,
                            x,
                            atom_decoder=self.task.atom_vocab,
                        )
                    for j in range(current_batch_size):
                        path_xyz = os.path.join(self.output_path, f"molecule_{str(j).zfill(3)}.xyz")
                        idx = i * self.batch_size + j
                        self._move_xyz(path_xyz, idx)

            except Exception as e:
                fail_count += 1
                self._fail_count += 1
                if self._first_exc is None:
                    self._first_exc = e
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })

    def hybrid_guidance(self):

        xh_ref = self.preprocess_ref_structure(self.task.device)
        self._validate_size_config(xh_ref.shape[1])
        condition_mode = self.task_type.split("_")[0] + "_" + self.condition_configs.get("condition_component",  "xh")
        
        # Map task_type suffix to guidance_ver if not explicitly set in config
        if self.task_type.endswith("_cfggg"):
            guidance_ver = self.condition_configs.get("guidance_ver", "cfg_gg")
        elif self.task_type.endswith("_gg"):
            guidance_ver = self.condition_configs.get("guidance_ver", 2)
        else:
            guidance_ver = self.condition_configs.get("guidance_ver", "cfg")
        
        if guidance_ver in [0, 1, 2, "cfg_gg"]:

            target_function=self.condition_configs.get("target_function", None)            
            if target_function is None:
                raise ValueError("Target function must be provided for gradient guidance")
            else:
                target_function.atom_vocab = self.task.atom_vocab      
                target_function.norm_factor = self.task.model.norm_values
                target_function = target_function()
        else:
            target_function = None
                
        scheduler = self.condition_configs.get("scheduler", None)
        if scheduler is not None:
            scheduler = scheduler()
            
        fail_count = 0
        num_round = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size != 0:
            num_round += 1
        current_batch_size = self.batch_size
        progress_bar = tqdm(range(num_round), desc="Sampling molecules", leave=True)
        self._num_round = num_round
        
        
        if hasattr(self.task, 'predictive_model'):
            property_eval = True
            df_dict = {
                "filename": [],
            }
            for prop_name in self.property_names:
                df_dict[prop_name] = []
            df_dict["size"] = []           
        else:
            logger.warning("Property model is not available, skip evaluation.")
            property_eval = False
             
        for i in progress_bar:
            if i == num_round-1 and self.num_generate % self.batch_size != 0:
                current_batch_size = self.num_generate % self.batch_size
            else:
                current_batch_size = self.batch_size
            try:
                if len(self.mol_size) == 1:
                    nodesxsample = torch.tensor(self.mol_size, dtype=torch.long)
                    nodesxsample = nodesxsample.repeat(current_batch_size) 
                elif len(self.mol_size) == 2:
                    if self.mol_size[0] == 0 and self.mol_size[1] == 0:
                        nodesxsample = self.task.node_dist_model.sample(current_batch_size)
                        if self.max_mol_size > 0:
                            nodesxsample = torch.clamp(nodesxsample, max=self.max_mol_size)
                    else:
                        mean = (self.mol_size[0] + self.mol_size[1]) / 2
                        std = (self.mol_size[1] - self.mol_size[0]) / 4
                        nodesxsample = torch.normal(mean=mean, std=std, size=(1,)).long()
                        nodesxsample = torch.clamp(nodesxsample, min=self.mol_size[0], max=self.mol_size[1])
                        nodesxsample = nodesxsample.repeat(current_batch_size) 

                if "inpaint" in self.task_type or "outpaint" in self.task_type:
                    nodesxsample = self._enforce_scaffold_size(nodesxsample, xh_ref.shape[1])

                xh_tensor = xh_ref.repeat(current_batch_size, 1, 1)
                inpaint_cfgs = self.condition_configs.get("inpaint_cfgs", {})
                outpaint_cfgs = self.condition_configs.get("outpaint_cfgs", {})
                if self.task_type in ("inpaint_cfg", "outpaint_cfg"):
                    inpaint_cfgs = _without_geometric_constraint_cfgs(inpaint_cfgs)
                    outpaint_cfgs = _without_geometric_constraint_cfgs(outpaint_cfgs)

                one_hot, charges, x, node_mask  = self.task.sample_hybrid_guidance(
                    target_function=target_function,
                    target_value=self.target_values,
                    negative_target_value=self.negative_target_values,
                    nodesxsample=nodesxsample,
                    gg_scale=self.condition_configs.get("gg_scale",1),
                    cfg_scale=self.condition_configs.get("cfg_scale",1),
                    cfg_scale_schedule=self.condition_configs.get("cfg_scale_schedule", None),
                    max_norm=self.condition_configs.get("max_norm",1),
                    std=1,
                    scheduler=scheduler,
                    guidance_ver=guidance_ver,
                    guidance_at=self.condition_configs.get("guidance_at",1),
                    guidance_stop=self.condition_configs.get("guidance_stop",0),
                    n_backwards=self.condition_configs.get("n_backwards",1),
                    condition_tensor=xh_tensor,
                    condition_mode=condition_mode,
                    inpaint_cfgs=inpaint_cfgs,
                    outpaint_cfgs=outpaint_cfgs,
                    use_noised_conditioning=self.condition_configs.get("use_noised_conditioning", False),
                    n_frames=self.n_frames,
                )   
                
                if self.visualize_trajectory:
                    for j in range(current_batch_size):
                        mol_idx = i * self.batch_size + j
                        output_path_frame = os.path.join(self.output_path, f"mol_{mol_idx}")
                        if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                            save_xyz_file_atomic_numbers(
                                output_path_frame,
                                x[:, j],
                                charges[:, j].squeeze(-1) if charges is not None else None,
                                idxs=self.s_saves.tolist(),
                                node_mask=node_mask[j].unsqueeze(0).expand(one_hot[:, j].size(0), -1, -1) if node_mask is not None else None,
                            )
                        else:
                            save_xyz_file(
                                output_path_frame,
                                one_hot[:, j],
                                x[:, j],
                                atom_decoder=self.task.atom_vocab,
                                idxs=self.s_saves.tolist(),
                                atomic_numbers=charges[:, j].squeeze(-1) if charges is not None and hasattr(self.task.model, 'use_unknown_fallback') else None,
                                use_unknown_fallback=getattr(self.task.model, 'use_unknown_fallback', False),
                                node_mask=node_mask[j].unsqueeze(0).expand(one_hot[:, j].size(0), -1, -1) if node_mask is not None else None,
                            )     
                        path_xyz = os.path.join(output_path_frame, f"molecule_000.xyz")
                        idx = i * self.batch_size + j
                        self._move_xyz(path_xyz, idx, trajectory_dir=output_path_frame)
            
                    x = x[:, -1]
                    one_hot = one_hot[:, -1]   
                else:     
                    if torch.all(one_hot == 0) or getattr(self.task, "atom_vocab", None) is None or one_hot.shape[-1] != len(self.task.atom_vocab):
                        save_xyz_file_atomic_numbers(
                            self.output_path,
                            x,
                            charges.squeeze(-1) if charges is not None else None,
                            node_mask=node_mask,
                        )
                    else:
                        save_xyz_file(
                            self.output_path,
                            one_hot,
                            x,
                            atom_decoder=self.task.atom_vocab,
                            atomic_numbers=charges.squeeze(-1) if charges is not None and hasattr(self.task.model, 'use_unknown_fallback') else None,
                            use_unknown_fallback=getattr(self.task.model, 'use_unknown_fallback', False),
                            node_mask=node_mask,
                        )

                    for j in range(current_batch_size):
                        
                        path_xyz = os.path.join(self.output_path, f"molecule_{str(j).zfill(3)}.xyz")
                        idx = i * self.batch_size + j
                        self._move_xyz(path_xyz, idx)
            
                if property_eval:
                    xh = torch.cat([
                        x,
                        one_hot,
                        charges
                    ])
                    preds = self.property_prediction(xh, t=0)
                    for prop_name in self.property_names:
                        logger.info(f"{prop_name}: {preds[prop_name]}")
                        df_dict[prop_name].append(preds[prop_name])
                    df_dict["filename"].append(f"molecule_{str(i+1).zfill(4)}.xyz")
                    df_dict["size"].append(nodesxsample.item())    
            except Exception as e:
                fail_count += 1
                self._fail_count += 1
                if self._first_exc is None:
                    self._first_exc = e
                tqdm.write(f"[Batch {i}] Sampling failed: {e}")

            progress_bar.set_postfix({
                "completed": i + 1,
                "failed": fail_count,
                "success": (i + 1 - fail_count),
                "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
            })
            
        if property_eval:    
            self.df = pd.DataFrame(df_dict)
    
                   
    def property_prediction(self, 
                            xh: torch.Tensor, # pos, node_feature
                            t: int):
        DIM = 3; RADIUS = 4
        bs, n_nodes, _ = xh.shape
        
        mol = {}
        device = get_device()
        mol = {}
        coords = xh[:, :, :DIM].view(n_nodes*bs, DIM).to(device)
        h = xh[:, :, DIM:-1].view(n_nodes*bs, -1).to(device)
        charge = xh[:, :, -1].view(n_nodes*bs).to(device)

        edge_index = radius_graph(coords, r=RADIUS)
        tags = torch.zeros(n_nodes, dtype=torch.long, device=device)
        
        times = torch.zeros(n_nodes, dtype=torch.float32, device=device) + t.item()
        times = times.view(n_nodes, 1)

        graph_data = Data(
                            x=h,
                            pos=coords,
                            atomic_numbers=charge,
                            natoms=n_nodes,
                            smiles=None,
                            xyz=None,
                            edge_index=edge_index,
                            tags=tags,
                            times=times,
                        )
        graph_data = Batch.from_data_list([graph_data])
        mol["graph"] = graph_data
        preds = self.task.predictive_model.predict(mol, evaluate=True)[0]
        return preds
    

    def _reference_from_feature_stats(self, device):
        stats = getattr(self.task, "reference_feature_stats", None)
        if stats is not None:
            node_feature = stats["node_feature"].to(device)
            atomic_numbers = stats["atomic_numbers"].to(device)
            coords = torch.zeros(
                node_feature.size(0),
                node_feature.size(1),
                3,
                dtype=node_feature.dtype,
                device=device,
            )
            return torch.cat([coords, node_feature, atomic_numbers], dim=-1)

        scaffold = getattr(self.task, "reference_scaffold", None)
        if scaffold is not None:
            scaffold = scaffold.to(device)
            coords = torch.zeros_like(scaffold[:, :, :3])
            return torch.cat([coords, scaffold[:, :, 3:]], dim=-1)

        return None

    def _center_saved_scaffold_by_com(self, scaffold: torch.Tensor) -> torch.Tensor:
        """Center embedded scaffold coordinates by center of mass (COM)."""
        if scaffold.ndim != 3 or scaffold.size(-1) < 4:
            logger.warning(
                "Cannot center saved scaffold: expected shape (B, N, C>=4), got %s",
                tuple(scaffold.shape),
            )
            return scaffold

        norm_values = getattr(self.task.model, "norm_values", None)
        if norm_values is None or len(norm_values) < 3:
            logger.warning(
                "Cannot center saved scaffold: model.norm_values is missing or invalid."
            )
            return scaffold

        norm_coords = torch.as_tensor(
            norm_values[0], dtype=scaffold.dtype, device=scaffold.device
        )
        norm_charges = torch.as_tensor(
            norm_values[2], dtype=scaffold.dtype, device=scaffold.device
        )

        if torch.any(norm_coords == 0) or torch.any(norm_charges == 0):
            logger.warning(
                "Cannot center saved scaffold: normalization values contain zeros."
            )
            return scaffold

        centered = scaffold.clone()
        coords_norm = centered[:, :, :3]
        coords_phys = coords_norm * norm_coords

        # Atomic number is stored in the final (normalized) charge channel.
        atomic_number = torch.round(centered[:, :, -1] * norm_charges).long()

        masses_table = torch.as_tensor(
            atomic_masses, dtype=coords_phys.dtype, device=coords_phys.device
        )
        masses = torch.zeros_like(coords_phys[:, :, 0])
        valid_z = (atomic_number >= 0) & (atomic_number < masses_table.numel())
        masses[valid_z] = masses_table[atomic_number[valid_z]]
        masses = torch.where(torch.isfinite(masses) & (masses > 0), masses, 0.0)

        total_mass = masses.sum(dim=1, keepdim=True)
        weighted_sum = (coords_phys * masses.unsqueeze(-1)).sum(dim=1)
        denom = total_mass.clamp_min(torch.finfo(coords_phys.dtype).eps)
        com = weighted_sum / denom

        invalid_mass_batches = (total_mass.squeeze(-1) <= 0)
        if torch.any(invalid_mass_batches):
            logger.warning(
                "Falling back to arithmetic-mean centering for %d scaffold batch(es) "
                "with invalid total mass.",
                int(invalid_mass_batches.sum().item()),
            )
            node_mask = torch.ones_like(coords_phys[:, :, :1])
            mean_centered = remove_mean_with_mask(coords_phys, node_mask)
            fallback_com = coords_phys - mean_centered
            fallback_com = fallback_com[:, :1, :].squeeze(1)
            com[invalid_mass_batches] = fallback_com[invalid_mass_batches]

        centered[:, :, :3] = (coords_phys - com.unsqueeze(1)) / norm_coords
        return centered

    def preprocess_ref_structure(self, device):
        """
        Load and preprocess a reference molecular structure from an XYZ file.
        
        This function reads an XYZ file, encodes atomic features, normalizes
        coordinates and features, and returns a tensor combining positions
        and processed features.  When the model uses extra node features
        (ndim_extra > 0), they are either computed from the reference
        structure or zero-padded.

        Returns:
            torch.Tensor: Tensor of shape (1, n_atoms, 3 + n_features + ndim_extra + 1)
                        containing [normalized_coords | normalized_onehot | normalized_extra | normalized_charges].
        
        Raises:
            FileNotFoundError: If the reference structure file is not found.
            ValueError: If the processed reference structure is empty.
        """
        file_path = self.condition_configs.get("reference_structure_path", None)
        if file_path and not os.path.exists(file_path):
            # A path that was given but does not resolve is a mistake, not a request for
            # the embedded scaffold. Falling back here silently builds every molecule
            # around a different reference than the caller named.
            raise FileNotFoundError(
                f"reference_structure_path was set to {file_path!r} but that file does "
                "not exist. Omit the key entirely to use the scaffold embedded in the "
                "checkpoint."
            )
        if not file_path:
            reference_freeze_mode = getattr(
                self.task, "reference_freeze_mode", "all"
            )
            if reference_freeze_mode == "features_only":
                feature_reference = self._reference_from_feature_stats(device)
                if feature_reference is not None:
                    logger.info(
                        "No reference_structure_path provided; using frozen "
                        "reference feature statistics embedded in checkpoint."
                    )
                    return feature_reference
            if (
                hasattr(self.task, "reference_scaffold")
                and self.task.reference_scaffold is not None
            ):
                logger.info(
                    "No reference_structure_path provided; using representative "
                    "medoid scaffold embedded in checkpoint."
                )
                scaffold = self.task.reference_scaffold.to(device)
                if self.condition_configs.get("center_saved_scaffold", False):
                    scaffold = self._center_saved_scaffold_by_com(scaffold)
                return scaffold
            raise FileNotFoundError(
                "No reference_structure_path was given and this checkpoint embeds no "
                "reference scaffold or feature statistics. Structure-guided generation "
                "needs one of the two."
            )

        # Load molecule with hydrogen atoms
        mol = PointCloud_Mol.from_xyz(
            file_path, with_hydrogen=True, forbidden_atoms=[]
        )
        
        # Extract atomic coordinates and number of atoms
        coords = mol.get_coord()
        n_atoms = len(mol.atoms)

        # One-hot encode atomic types
        atom_vocab = self.task.atom_vocab
        node_features = [
            onehot(atom.element, atom_vocab, allow_unknown=False)
            for atom in mol.atoms
        ]

        # Atomic numbers (or model-specific charge encoding)
        charges = [atomic_numbers[atom.element]
                    for atom in mol.atoms
                    if atom.element in atomic_numbers]
        
        # Normalization
        normalize_coords, normalize_feats, normalize_charges = self.task.model.norm_values
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32).view(1, n_atoms, 3) / normalize_coords
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32).view(1, n_atoms, -1) / normalize_feats
        charges_tensor = torch.tensor(charges, dtype=torch.float32).view(1, n_atoms, 1) / normalize_charges

        # Handle extra node features
        ndim_extra = getattr(self.task.model, 'ndim_extra', 0)
        if ndim_extra > 0:
            extra_features_path = self.condition_configs.get("extra_features_path", None)
            if extra_features_path is None:
                raise ValueError(
                    f"Model uses {ndim_extra} extra node feature dimensions, but "
                    "'extra_features_path' is not set in condition_configs. "
                    "Provide a .npy file of shape (N_atoms, ndim_extra) containing "
                    "the raw (unnormalized) extra features for the reference structure."
                )
            extra_features = self._load_extra_features(
                extra_features_path, n_atoms, ndim_extra
            )
            # extra_features: (1, n_atoms, ndim_extra), already normalized
            features = torch.cat([node_features_tensor, extra_features, charges_tensor], dim=-1)
        else:
            features = torch.cat([node_features_tensor, charges_tensor], dim=-1)

        xh_ref = torch.cat([coords_tensor, features], dim=-1).to(device)

        if xh_ref.nelement() == 0:
            raise ValueError("Reference structure is empty or could not be processed.")

        return xh_ref

    def _load_extra_features(self, path, n_atoms, ndim_extra):
        """
        Load user-provided extra node features from a .npy file.

        The file must contain a float array of shape (n_atoms, ndim_extra)
        with **raw (unnormalized)** feature values matching the training
        pipeline's feature order. Normalization by extra_norm_values is
        applied automatically.

        Args:
            path: Path to the .npy file.
            n_atoms: Expected number of atoms.
            ndim_extra: Expected number of extra feature dimensions.

        Returns:
            Tensor of shape (1, n_atoms, ndim_extra), normalized.
        """
        import numpy as np

        if not os.path.exists(path):
            raise FileNotFoundError(f"Extra features file not found: {path}")

        feats = np.load(path)

        if feats.ndim == 1:
            # Allow flat (n_atoms * ndim_extra,) → reshape
            if feats.shape[0] == n_atoms * ndim_extra:
                feats = feats.reshape(n_atoms, ndim_extra)
            else:
                raise ValueError(
                    f"Extra features file has shape {feats.shape}, expected "
                    f"({n_atoms}, {ndim_extra}) or ({n_atoms * ndim_extra},)"
                )

        if feats.shape != (n_atoms, ndim_extra):
            raise ValueError(
                f"Extra features shape mismatch: got {feats.shape}, "
                f"expected ({n_atoms}, {ndim_extra})"
            )

        extra_norm = torch.tensor(
            self.task.model.extra_norm_values, dtype=torch.float32
        ).view(1, 1, -1)

        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, N, ndim_extra)
        feats_t = feats_t / extra_norm

        logger.info(
            f"Loaded extra features from {path}: shape {feats.shape}, "
            f"normalized by {self.task.model.extra_norm_values}"
        )

        return feats_t

class PharmacophoreConditionGenerator:
    """
    Entry point for all ShEPhERD pharmacophore generation modes.

    task_type selects the generation mode:
      - unconditional          : joint x1/x2/x3/x4 from pure noise
      - pharmacophore_condition: condition on pharmacophores extracted from reference_mol
      - shape_conditioned      : condition on surface + electrostatics from reference_mol
      - pharmacophore_inpaint  : scaffold inpainting — keep scaffold_smarts atoms, regenerate rest
      - from_intermediate_time : soft scaffold hopping starting from a noisy intermediate

    For all conditional modes, provide:
      reference_mol : path to .pkl (list of (molblock, charges) tuples) or .sdf file
      mol_idx       : which molecule to use from the file (default 0)

    For pharmacophore_inpaint:
      scaffold_smarts: SMARTS pattern selecting atoms to keep (e.g. 'c1ccccc1')
                       If null, keeps all ring atoms and regenerates chains.
    """

    def __init__(
        self,
        task,
        task_type: str = "pharmacophore_condition",
        num_generate: int = 10,
        batch_size: int = 1,
        N_x1: list = [20],
        N_x4: int = 5,
        N_x1_sampling: str = "uniform",
        distributions_path: str = None,
        distributions_key: str = "gdb",
        num_steps: int = 400,
        prior_noise_scale: float = 1.0,
        denoising_noise_scale: float = 1.0,
        output_path: str = "generated_pharmacophore",
        save_xyzrender_figures: bool = False,
        seed: int = 42,
        verbose: bool = True,
        # --- reference molecule (all conditional modes) ---
        reference_mol: str = None,
        mol_idx: int = 0,
        # --- pharmacophore_inpaint scaffold selection ---
        scaffold_smarts: str = None,
        inpaint_x1_pos: bool = True,
        inpaint_x1_x: bool = True,
        inpaint_x1_bonds: bool = True,
        stop_inpainting_at_time_x1_pos: float = 0.0,
        stop_inpainting_at_time_x1_x: float = 0.0,
        stop_inpainting_at_time_x1_bonds: float = 0.0,
        stop_inpainting_at_time_x4: float = 0.0,
        save_modalities: bool = False,
        # --- multi-profile generation toggles ---
        compute_x1: bool = True,
        compute_x2: bool = True,
        compute_x3: bool = True,
        compute_x4: bool = True,
        # --- from_intermediate_time ---
        start_time: float = 0.5,
        use_noised_conditioning: bool = False,
        condition_configs: dict = {},
    ):
        self.task = task
        self.task_type = task_type
        self.num_generate = num_generate
        self.batch_size = batch_size
        # Normalise N_x1: accept int for back-compat, convert to list
        if isinstance(N_x1, int):
            N_x1 = [N_x1]
        self.N_x1 = N_x1
        if N_x1_sampling not in ("uniform", "normal"):
            raise ValueError(f"N_x1_sampling must be 'uniform' or 'normal', got '{N_x1_sampling}'")
        self.N_x1_sampling = N_x1_sampling
        self.N_x4 = N_x4
        self.num_steps = num_steps

        # Load empirical P(N_x4 | N_x1) distribution if N_x4 == 0
        self._nx4_dist = None
        if N_x4 == 0:
            if distributions_path is None:
                raise ValueError("distributions_path must be set when N_x4=0")
            import numpy as np
            data = np.load(distributions_path)
            if distributions_key not in data:
                raise ValueError(f"distributions_key '{distributions_key}' not found in {distributions_path}. Available: {list(data.keys())}")
            self._nx4_dist = data[distributions_key]
            logger.info(f"N_x4 will be sampled from '{distributions_key}' distribution in {distributions_path}")
        self.prior_noise_scale = prior_noise_scale
        self.denoising_noise_scale = denoising_noise_scale
        self.output_path = output_path
        self.save_xyzrender_figures = save_xyzrender_figures
        self.seed = seed
        self.verbose = verbose
        self.reference_mol = reference_mol
        self.mol_idx = mol_idx
        self.scaffold_smarts = scaffold_smarts
        self.inpaint_x1_pos = inpaint_x1_pos
        self.inpaint_x1_x = inpaint_x1_x
        self.inpaint_x1_bonds = inpaint_x1_bonds
        self.stop_inpainting_at_time_x1_pos = stop_inpainting_at_time_x1_pos
        self.stop_inpainting_at_time_x1_x = stop_inpainting_at_time_x1_x
        self.stop_inpainting_at_time_x1_bonds = stop_inpainting_at_time_x1_bonds
        self.stop_inpainting_at_time_x4 = stop_inpainting_at_time_x4
        self.start_time = start_time
        self.save_modalities = save_modalities
        self.compute_x1 = compute_x1
        self.compute_x2 = compute_x2
        self.compute_x3 = compute_x3
        self.compute_x4 = compute_x4
        self.condition_configs = condition_configs
        self.use_noised_conditioning = self.condition_configs.get("use_noised_conditioning", use_noised_conditioning)

        # Load and cache conditioning data from reference molecule
        self._ref = None
        if task_type != "unconditional":
            if reference_mol is None:
                raise ValueError(f"reference_mol is required for task_type='{task_type}'")
            self._ref = self._load_reference(reference_mol, mol_idx)

    def _render_batch_xyz_figures(self, idx_offset: int, batch_size: int) -> None:
        if not self.save_xyzrender_figures:
            return
        for idx in range(idx_offset, idx_offset + batch_size):
            xyz_path = os.path.join(self.output_path, f"mol_{idx:04d}.xyz")
            if os.path.exists(xyz_path):
                _render_xyz_figure(xyz_path)

    def _load_reference(self, path: str, mol_idx: int) -> dict:
        """
        Load a reference molecule from a .pkl or .sdf file and extract all
        conditioning arrays needed by the sampler.

        Returns a dict with keys:
          mol, charges, surface, electrostatics, center_of_mass,
          pharm_types, pharm_pos, pharm_direction
        """
        import numpy as np
        from rdkit import Chem

        path = str(path)

        # --- Load mol + charges ---
        if path.endswith('.pkl'):
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            entry = data[mol_idx]
            molblock, charges = entry[0], np.array(entry[1], dtype=np.float32)
            mol = Chem.MolFromMolBlock(molblock, removeHs=False)
            if mol is None:
                raise ValueError(f"Failed to parse molblock at index {mol_idx} in {path}")
        elif path.endswith('.sdf'):
            supplier = Chem.SDMolSupplier(path, removeHs=False)
            mol = supplier[mol_idx]
            if mol is None:
                raise ValueError(f"Failed to parse molecule at index {mol_idx} in {path}")
            from rdkit.Chem import AllChem
            AllChem.ComputeGasteigerCharges(mol)
            charges = np.array(
                [float(mol.GetAtomWithIdx(i).GetPropsAsDict().get('_GasteigerCharge', 0.0))
                 for i in range(mol.GetNumAtoms())],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported reference_mol format (expected .pkl or .sdf): {path}")

        # Center molecule at origin (required by ShEPhERD)
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        com = coords.mean(axis=0)
        new_conf = Chem.Conformer(mol.GetNumAtoms())
        for i, pos in enumerate(coords - com):
            new_conf.SetAtomPosition(i, pos.tolist())
        mol.RemoveAllConformers()
        mol.AddConformer(new_conf, assignId=True)

        centers = mol.GetConformer().GetPositions().astype(np.float32)
        num_surf = self.task.model.dynamics.params.get('dataset', {}).get('x2', {}).get('num_points', 75)

        from MolecularDiffusion.utils.shepherd_utils import (
            get_molecular_surface, get_atomic_vdw_radii, get_electrostatics_given_point_charges,
            get_pharmacophores
        )

        # --- Surface (x2) + electrostatics (x3) ---
        radii = get_atomic_vdw_radii(mol)
        surface = get_molecular_surface(centers, radii, num_points=num_surf, probe_radius=0.6, num_samples_per_atom=20)
        electrostatics = get_electrostatics_given_point_charges(charges, centers, surface)

        # --- Pharmacophores (x4) ---
        # Note: check_accessibility=False in original tasks_generate.py
        pharm_types, pharm_pos, pharm_direction = get_pharmacophores(mol, multi_vector=False)

        logger.info(
            f"Loaded reference mol [{mol_idx}] from {path}: "
            f"{mol.GetNumAtoms()} atoms, {len(pharm_types)} pharmacophores"
        )

        return dict(
            mol=mol,
            charges=charges,
            surface=surface,
            electrostatics=electrostatics,
            center_of_mass=np.zeros(3, dtype=np.float32),
            pharm_types=pharm_types,
            pharm_pos=pharm_pos,
            pharm_direction=pharm_direction,
        )

    def _get_scaffold_inds(self, mol):
        """
        Return (atom_inds_to_inpaint, exit_vector_atom_inds) for inpainting.

        Uses scaffold_smarts if provided; otherwise keeps all ring atoms.
        """
        from rdkit import Chem
        if self.scaffold_smarts:
            pattern = Chem.MolFromSmarts(self.scaffold_smarts)
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                raise ValueError(f"scaffold_smarts '{self.scaffold_smarts}' had no match in reference mol")
            scaffold_set = set(matches[0])
        else:
            # Default: keep all ring atoms
            scaffold_set = set(idx for ring in mol.GetRingInfo().AtomRings() for idx in ring)
            if not scaffold_set:
                raise ValueError("No ring atoms found and scaffold_smarts not provided")

        atom_inds_to_inpaint = sorted(scaffold_set)
        exit_vector_atom_inds = [
            i for i in atom_inds_to_inpaint
            if any(n.GetIdx() not in scaffold_set for n in mol.GetAtomWithIdx(i).GetNeighbors())
        ]
        # If every scaffold atom is an exit vector there are no interior scaffold atoms,
        # which causes an indexing error in the sampler bond inpainting code.
        # In that case treat them all as interior (pass no exit vectors).
        if set(exit_vector_atom_inds) == set(atom_inds_to_inpaint):
            exit_vector_atom_inds = []
        return atom_inds_to_inpaint, exit_vector_atom_inds

    def run(self):
        from MolecularDiffusion.utils.geom_utils import save_shepherd_outputs
        torch.manual_seed(self.seed)
        os.makedirs(self.output_path, exist_ok=True)

        num_rounds = self.num_generate // self.batch_size
        if self.num_generate % self.batch_size:
            num_rounds += 1

        idx_offset = 0
        for i in tqdm(range(num_rounds), desc=f"PharmacophoreConditionGenerator [{self.task_type}]"):
            bs = self.batch_size
            if i == num_rounds - 1 and self.num_generate % self.batch_size:
                bs = self.num_generate % self.batch_size
            try:
                structures = self._generate_batch(bs)
                save_shepherd_outputs(self.output_path, structures, idx_offset=idx_offset, save_modalities=self.save_modalities)
                self._render_batch_xyz_figures(idx_offset, len(structures))
                idx_offset += len(structures)
            except Exception as e:
                logger.warning(f"[Batch {i}] Generation failed: {e}")

        logger.info(f"Generated {idx_offset} structures → {self.output_path}")

    def _sample_N_x1(self) -> int:
        """Return N_x1 for one batch.
        [N]       -> always N
        [N1, N2]  -> sample in [N1, N2] via uniform or normal distribution
        """
        import numpy as np
        if len(self.N_x1) == 1:
            return self.N_x1[0]
        n1, n2 = self.N_x1[0], self.N_x1[1]
        if self.N_x1_sampling == "uniform":
            return int(np.random.randint(n1, n2 + 1))
        else:  # normal
            mean = (n1 + n2) / 2
            std = (n2 - n1) / 4
            val = int(round(np.random.normal(mean, std)))
            return int(np.clip(val, n1, n2))

    def _sample_N_x4(self, N_x1: int) -> int:
        """Sample N_x4 from P(N_x4 | N_x1) when N_x4==0 was specified."""
        import numpy as np
        row = self._nx4_dist[N_x1, :]
        if row.sum() > 0:
            prob = row / row.sum()
        else:
            prob = self._nx4_dist.sum(axis=0)
            prob = prob / prob.sum()
        return int(np.random.multinomial(1, pvals=prob).argmax())

    def _generate_batch(self, batch_size: int) -> list:
        from MolecularDiffusion.modules.models.shepherd_arch.inference import (
            generate, generate_from_intermediate_time,
        )
        r = self._ref  # shorthand; None for unconditional

        # Filter the generated profiles based on toggles and the model's capabilities
        if hasattr(self.task, 'model') and hasattr(self.task.model, 'dynamics'):
            p = self.task.model.dynamics.params
            if not hasattr(self, '_original_vars'):
                self._original_vars = list(p.get('explicit_diffusion_variables', ['x1', 'x2', 'x3', 'x4']))
            
            new_vars = []
            if self.compute_x1 and 'x1' in self._original_vars: new_vars.append('x1')
            if self.compute_x2 and 'x2' in self._original_vars: new_vars.append('x2')
            if self.compute_x3 and 'x3' in self._original_vars: new_vars.append('x3')
            if self.compute_x4 and 'x4' in self._original_vars: new_vars.append('x4')
            p['explicit_diffusion_variables'] = new_vars

        N_x1 = self._sample_N_x1()

        if self.task_type == "unconditional":
            N_x4 = self._sample_N_x4(N_x1) if self._nx4_dist is not None else self.N_x4
            return generate(
                model_pl=self.task.model,
                batch_size=batch_size,
                N_x1=N_x1,
                N_x4=N_x4,
                unconditional=True,
                prior_noise_scale=self.prior_noise_scale,
                denoising_noise_scale=self.denoising_noise_scale,
                num_steps=self.num_steps,
                verbose=self.verbose,
            )

        elif self.task_type == "pharmacophore_condition":
            return generate(
                model_pl=self.task.model,
                batch_size=batch_size,
                N_x1=N_x1,
                N_x4=len(r['pharm_types']),
                unconditional=False,
                inpaint_x4_pos=True,
                inpaint_x4_direction=True,
                inpaint_x4_type=True,
                pharm_types=r['pharm_types'],
                pharm_pos=r['pharm_pos'],
                pharm_direction=r['pharm_direction'],
                surface=r['surface'],
                electrostatics=r['electrostatics'],
                center_of_mass=r['center_of_mass'],
                prior_noise_scale=self.prior_noise_scale,
                denoising_noise_scale=self.denoising_noise_scale,
                num_steps=self.num_steps,
                verbose=self.verbose,
            )

        elif self.task_type == "shape_conditioned":
            return generate(
                model_pl=self.task.model,
                batch_size=batch_size,
                N_x1=N_x1,
                N_x4=self.N_x4,
                unconditional=False,
                inpaint_x2_pos=True,
                inpaint_x3_pos=True,
                inpaint_x3_x=True,
                surface=r['surface'],
                electrostatics=r['electrostatics'],
                center_of_mass=r['center_of_mass'],
                prior_noise_scale=self.prior_noise_scale,
                denoising_noise_scale=self.denoising_noise_scale,
                num_steps=self.num_steps,
                verbose=self.verbose,
            )

        elif self.task_type == "pharmacophore_inpaint":
            atom_inds, exit_inds = self._get_scaffold_inds(r['mol'])
            return generate(
                model_pl=self.task.model,
                batch_size=batch_size,
                N_x1=N_x1,
                N_x4=len(r['pharm_types']),
                unconditional=False,
                inpaint_x1_pos=self.inpaint_x1_pos,
                inpaint_x1_x=self.inpaint_x1_x,
                inpaint_x1_bonds=self.inpaint_x1_bonds,
                inpaint_x4_pos=True,
                inpaint_x4_direction=True,
                inpaint_x4_type=True,
                stop_inpainting_at_time_x1_pos=self.stop_inpainting_at_time_x1_pos,
                stop_inpainting_at_time_x1_x=self.stop_inpainting_at_time_x1_x,
                stop_inpainting_at_time_x1_bonds=self.stop_inpainting_at_time_x1_bonds,
                stop_inpainting_at_time_x4=self.stop_inpainting_at_time_x4,
                mol=r['mol'],
                atom_inds_to_inpaint=atom_inds,
                exit_vector_atom_inds=exit_inds,
                pharm_types=r['pharm_types'],
                pharm_pos=r['pharm_pos'],
                pharm_direction=r['pharm_direction'],
                surface=r['surface'],
                electrostatics=r['electrostatics'],
                center_of_mass=r['center_of_mass'],
                prior_noise_scale=self.prior_noise_scale,
                denoising_noise_scale=self.denoising_noise_scale,
                num_steps=self.num_steps,
                verbose=self.verbose,
            )

        elif self.task_type == "from_intermediate_time":
            atom_inds, exit_inds = self._get_scaffold_inds(r['mol'])
            return generate_from_intermediate_time(
                model_pl=self.task.model,
                batch_size=batch_size,
                start_time=self.start_time,
                N_x1=max(N_x1, r['mol'].GetNumAtoms()),
                N_x4=len(r['pharm_types']),
                mol=r['mol'],
                atom_inds_to_inpaint=atom_inds,
                exit_vector_atom_inds=exit_inds,
                inpaint_x1_bonds=self.inpaint_x1_bonds,
                pharm_types=r['pharm_types'],
                pharm_pos=r['pharm_pos'],
                pharm_direction=r['pharm_direction'],
                surface=r['surface'],
                electrostatics=r['electrostatics'],
                center_of_mass=r['center_of_mass'],
                stop_inpainting_at_time_x1_pos=self.stop_inpainting_at_time_x1_pos,
                stop_inpainting_at_time_x1_x=self.stop_inpainting_at_time_x1_x,
                stop_inpainting_at_time_x1_bonds=self.stop_inpainting_at_time_x1_bonds,
                denoising_noise_scale=self.denoising_noise_scale,
                num_steps=self.num_steps,
                verbose=self.verbose,
            )

        else:
            raise ValueError(f"Unknown task_type for PharmacophoreConditionGenerator: {self.task_type}")
