from collections import deque
from collections.abc import Mapping, Sequence

import torch
import torch_geometric


def graph_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            # out = elem.new(storage)
            out = elem.new(storage).resize_(len(batch), *elem.size())
        try:
            stacked = torch.stack(batch, 0, out=out)
        except RuntimeError:
            stacked = torch.cat(batch, 0, out=out)
        return stacked
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, torch_geometric.data.Data):
        return torch_geometric.data.Batch.from_data_list(batch)
    elif isinstance(elem, Mapping):
        return {key: graph_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("Each element in list of batch should be of equal size")
        return [graph_collate(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))


def pointcloud_collate_v0(batch, to_keep=None):
    """
    Convert any list of same nested container into a container of tensors.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *elem.size())
        try:
            stacked = torch.stack(batch, 0, out=out)
        except RuntimeError:
            stacked = torch.cat(batch, 0, out=out)

        norigin = stacked.size()[1]
        if to_keep is not None:
            stacked = stacked[:, to_keep]

            if stacked.dim() == 3 and stacked.size()[2] == norigin:
                stacked = stacked[:, :, to_keep]

        return stacked
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, Mapping):
        charges_collated = pointcloud_collate_v0([d["charges"] for d in batch])
        to_keep = charges_collated.sum(0) > 0
        to_keep = to_keep.squeeze()
        return {
            key: pointcloud_collate_v0([d[key] for d in batch], to_keep) for key in elem
        }
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("Each element in list of batch should be of equal size")
        return [pointcloud_collate_v0(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))


def pointcloud_collate(vram_size=40):
    def _pointcloud_collate(batch):
        """
        Convert any list of same nested container into a container of tensors.
        
        Parameters:
            batch (list): list of samples with the same nested container
        """
        elem = batch[0]

        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *elem.size())
            try:
                stacked = torch.stack(batch, 0, out=out)
            except RuntimeError:
                stacked = torch.cat(batch, 0, out=out)

            return stacked
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, (str, bytes)):
            return batch
        elif isinstance(elem, Mapping):

            natoms = _pointcloud_collate([d["natoms"] for d in batch])

            max_atom = natoms.max()
            eff_batch_size = estimate_batch_size(max_atom, vram_size)
            batch_size = len(batch)

            if batch_size > eff_batch_size:
                idcs_choose = torch.randint(0, batch_size, (eff_batch_size,))
                print(
                    "Batch size too large, randomly choose %d samples" % eff_batch_size
                )
            else:
                idcs_choose = torch.arange(batch_size)

            eff_batch = []
            point_cloud_keys = [
                "coords",
                "node_feature",
                "charges",
                "edge_mask",
                "node_mask",
                "natoms",
            ]
            for i in idcs_choose:

                data_obj = {}
                n_nodes = natoms[i]
                node_mask = torch.zeros(max_atom)
                coords_full = torch.zeros(max_atom, 3)
                charges_mask = torch.zeros(max_atom)
                coords_full[:n_nodes] = batch[i]["coords"]
                node_mask[:n_nodes] = 1
                node_features = batch[i]["node_feature"]
                node_feat_full = torch.zeros(max_atom, node_features.size(1))
                node_feat_full[:n_nodes] = node_features
                charges_mask[:n_nodes] = batch[i]["charges"]
                edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                diag_mask = ~torch.eye(max_atom, dtype=torch.bool)
                edge_mask *= diag_mask

                data_obj["node_mask"] = node_mask
                data_obj["coords"] = coords_full
                data_obj["node_feature"] = node_feat_full
                data_obj["charges"] = charges_mask
                data_obj["edge_mask"] = edge_mask
                data_obj["natoms"] = n_nodes
                for key in elem:
                    if key not in point_cloud_keys:
                        data_obj[key] = batch[i][key]
                eff_batch.append(data_obj)

            return {
                key: _pointcloud_collate([d[key] for d in eff_batch]) for key in elem
            }

        elif isinstance(elem, Sequence):
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    "Each element in list of batch should be of equal size"
                )
            return [_pointcloud_collate(samples) for samples in zip(*batch)]

        raise TypeError("Can't collate data with type `%s`" % type(elem))

    return _pointcloud_collate


def estimate_batch_size(nmax, vram_size=40):

    eff_batch_size = 32 * vram_size / (0.0022 * nmax**2 + 0.0208 * nmax - 0.2333)

    return int(eff_batch_size)


class DataLoader(torch.utils.data.DataLoader):
    """
    Extended data loader for batching graph structured data.

    See `torch.utils.data.DataLoader`_ for more details.

    .. _torch.utils.data.DataLoader:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    Parameters:
        dataset (Dataset): dataset from which to load the data
        batch_size (int, optional): how many samples per batch to load
        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch
        sampler (Sampler, optional): sampler that draws single sample from the dataset
        batch_sampler (Sampler, optional): sampler that draws a mini-batch of data from the dataset
        num_workers (int, optional): how many subprocesses to use for data loading
        collate_fn (callable, optional): merge a list of samples into a mini-batch
        kwargs: keyword arguments for `torch.utils.data.DataLoader`_
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=graph_collate,
        **kwargs
    ):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            **kwargs
        )


class DataQueue(torch.utils.data.Dataset):

    def __init__(self):
        self.queue = deque()

    def append(self, item):
        self.queue.append(item)

    def pop(self):
        self.queue.popleft()

    def __getitem__(self, index):
        return self.queue[index]

    def __len__(self):
        return len(self.deque)

