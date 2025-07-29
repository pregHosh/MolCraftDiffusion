

import torch
from torch import nn

from MolecularDiffusion import data # TODO
import os
import numpy as np
import random

MIN_SEED_VALUE = 0
MAX_SEED_VALUE = 2**32 - 1


def recursive_module_to_device(module: nn.Module, device: torch.device):
    """
    Recursively assigns a given device to all submodules of a torch.nn.Module.

    Args:
        module (nn.Module): The main module to which the device needs to be assigned.
        device (torch.device): The target device (e.g., torch.device('cuda') or torch.device('cpu')).
    """
    for child_name, child_module in module.named_children():
        # If the child module has its own children, recurse
        if len(list(child_module.children())) > 0:
            recursive_module_to_device(child_module, device)
        # Move the current child module to the specified device
        child_module.to(device)
        child_module.device = device

    # Finally, move the top-level module itself to the device
    module.to(device)
    module.device = device
    
    
    
def cpu(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CPU.
    """
    if hasattr(obj, "cpu"):
        return obj.cpu(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cpu(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cpu(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def detach(obj):
    """
    Detach tensors in any nested conatiner.
    """
    if hasattr(obj, "detach"):
        return obj.detach()
    elif isinstance(obj, dict):
        return type(obj)({k: detach(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(detach(x) for x in obj)

    raise TypeError("Can't perform detach over object type `%s`" % type(obj))


def clone(obj, *args, **kwargs):
    """
    Clone tensors in any nested conatiner.
    """
    if hasattr(obj, "clone"):
        return obj.clone(*args, **kwargs)
    elif isinstance(obj, dict):
        return type(obj)({k: clone(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(clone(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't perform detach over object type `%s`" % type(obj))


def mean(obj, *args, **kwargs):
    """
    Compute mean of tensors in any nested container.
    """
    if hasattr(obj, "mean"):
        return obj.mean(*args, **kwargs)
    elif isinstance(obj, dict):
        return type(obj)({k: mean(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(mean(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't perform mean over object type `%s`" % type(obj))


def cat(objs, *args, **kwargs):
    """
    Concatenate a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.cat(objs, *args, **kwargs)
    elif isinstance(obj, data.PackedGraph):
        return data.cat(objs)
    elif isinstance(obj, dict):
        return {k: cat([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cat(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform concatenation over object type `%s`" % type(obj))


def stack(objs, *args, **kwargs):
    """
    Stack a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.stack(objs, *args, **kwargs)
    elif isinstance(obj, dict):
        return {k: stack([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(stack(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform stack over object type `%s`" % type(obj))


def seed_everything(seed: int = None, workers: bool = False, verbose: bool = True) -> int:
    """Sets seed for reproducibility across torch, numpy, and random modules.

    Args:
        seed (int, optional): The seed to use. If None, it checks 'PL_GLOBAL_SEED' in env or defaults to 0.
        workers (bool): Whether to set the 'PL_SEED_WORKERS' env variable.
        verbose (bool): If True, logs the chosen seed.
    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED", None)
        try:
            seed = int(env_seed) if env_seed is not None else 0
        except ValueError:
            seed = 0
            if verbose:
                print(f"Invalid env seed {repr(env_seed)}; defaulting to 0")
    else:
        seed = int(seed)

    if not (MIN_SEED_VALUE <= seed <= MAX_SEED_VALUE):
        if verbose:
            print(f"Seed {seed} out of bounds; must be between {MIN_SEED_VALUE} and {MAX_SEED_VALUE}")
        seed = 0

    if verbose:
        print(f"Setting seed to {seed}")

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    return seed

def get_vram_size():
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024**3)  # Convert bytes to gigabytes
        return vram_gb
    else:
        return None