import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from MolecularDiffusion.modules.layers import common, functional
from MolecularDiffusion import core

from .task import Task, _get_criterion_name, _get_metric_name
from .metrics import pearsonr, accuracy, matthews_corrcoef, area_under_prc, area_under_roc, r2, spearmanr


@core.Registry.register("ProperyPrediction")
class ProperyPrediction(Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(
        self,
        model,
        task=(),
        include_charge=False,
        criterion="mse",
        metric=("mae", "rmse"),
        loss_param=None,
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        mlp_batch_norm=False,
        condition_time=False,
        readout="mean",
        mlp_dropout=0,
        std_mean=None,
        load_mlps_layer=0,
        verbose=0,
    ):

        super(ProperyPrediction, self).__init__()
        self.model = model
        self.task = task
        self.include_charge = include_charge
        self.criterion = criterion
        self.metric = metric
        self.loss_param = loss_param
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.verbose = verbose
        self.std_mean = std_mean
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.readout = readout

        self.condition_time = condition_time

        if self.model.__class__.__name__ in [
            "GraphTransformer",
            "GraphDiffTransformer",
        ]:
            self.architecture = "egt"
        elif self.model.__class__.__name__ in ["EGNN"]:
            self.architecture = "egcn"
        elif self.model.__class__.__name__ in ["PaiNN", "GemNetOC"]:
            self.architecture = "egnn_extra"

        self.THESHOLD = 4.5
        self.mlp = None
        self.mlp_final = None
        if std_mean:
            self.std = std_mean[0]
            self.mean = std_mean[1]

        if load_mlps_layer > 0:
            hidden_dims = [self.model.hidden_nf] * (self.num_mlp_layer - 1)
            self.mlp = common.MLP(
                self.model.hidden_nf,
                hidden_dims,
                batch_norm=self.mlp_batch_norm,
                dropout=self.mlp_dropout,
            )
        self.load_mlps_layer = load_mlps_layer

        if self.num_class:
            if load_mlps_layer > 0:
                n_layer_final = self.num_mlp_layer - load_mlps_layer - 1
                self.mlp_final = common.MLP(
                    hidden_dims[n_layer_final:-1],
                    sum(self.num_class),
                    batch_norm=self.mlp_batch_norm,
                    dropout=self.mlp_dropout,
                )
            else:
                hidden_dims = [self.model.hidden_nf] * (self.num_mlp_layer - 1)
                self.mlp = common.MLP(
                    self.model.hidden_nf,
                    hidden_dims + [sum(self.num_class)],
                    batch_norm=self.mlp_batch_norm,
                    dropout=self.mlp_dropout,
                )

    def preprocess(self, train_set, valid_set=None, test_set=None):
        """
        Compute the mean and derivation for each task on the training set.
        """
        if len(train_set) == 0:
            raise ValueError("Training set is empty. check the data path and format.")
        values = defaultdict(list)
        if train_set is not None:
            for sample in train_set:
                if not sample.get("labeled", True):
                    continue
                for task in self.task:
                    if not math.isnan(sample[task]):
                        values[task].append(sample[task])
            mean = []
            std = []
            weight = []
            num_class = []
            for task, w in self.task.items():
                value = torch.tensor(values[task])
                mean.append(value.float().mean())
                std.append(value.float().std())
                weight.append(w)
                if value.ndim > 1:
                    num_class.append(value.shape[1])
                elif value.dtype == torch.long:
                    task_class = value.max().item()
                    if task_class == 1 and "bce" in self.criterion:
                        num_class.append(1)
                    else:
                        num_class.append(task_class + 1)
                else:
                    num_class.append(1)
            if not hasattr(self, "mean"):
                print("mean and std not found, registering buffer")
                self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))

            if not hasattr(self, "std"):
                self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
            self.num_class = self.num_class or num_class

            hidden_dims = [self.model.hidden_nf] * (self.num_mlp_layer - 1)

            if self.mlp is None:
                self.mlp = common.MLP(
                    self.model.hidden_nf,
                    hidden_dims + [sum(self.num_class)],
                    batch_norm=self.mlp_batch_norm,
                    dropout=self.mlp_dropout,
                )
            if self.load_mlps_layer > 0:
                n_layer_final = self.num_mlp_layer - self.load_mlps_layer - 1
                self.mlp_final = common.MLP(
                    hidden_dims[n_layer_final:-1],
                    sum(self.num_class),
                    batch_norm=self.mlp_batch_norm,
                    dropout=self.mlp_dropout,
                )
        else:
            pass

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0
        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss(
                        (pred - self.mean) / self.std,
                        (target - self.mean) / self.std,
                        reduction="none",
                    )
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "huber":
                if self.normalization:
                    loss = F.huber_loss(
                        (pred - self.mean) / self.std,
                        (target - self.mean) / self.std,
                        reduction="none",
                        delta=self.loss_param,
                    )
                else:
                    loss = F.huber_loss(
                        pred,
                        target,
                        reduction="none",
                        delta=self.loss_param,
                    )
            elif criterion == "smooth_l1":
                if self.normalization:
                    loss = F.smooth_l1_loss(
                        (pred - self.mean) / self.std,
                        (target - self.mean) / self.std,
                        reduction="none",
                        beta=self.loss_param,
                    )
                else:
                    loss = F.smooth_l1_loss(
                        pred,
                        target,
                        reduction="none",
                        beta=self.loss_param,
                    )
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none"
                )
            elif criterion == "ce":
                loss = F.cross_entropy(
                    pred, target.long().squeeze(-1), reduction="none"
                ).unsqueeze(-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = _get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get(
            "labeled", torch.ones(len(target), dtype=torch.bool, device=target.device)
        )
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class : num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class : num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = matthews_corrcoef(
                        _pred[_labeled], _target[_labeled].long()
                    )
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(
                    pred.t(), target.long().t(), labeled.t()
                ):
                    _score = area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(
                    pred.t(), target.long().t(), labeled.t()
                ):
                    _score = area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = _get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric

    def predict(self, batch, all_loss=None, metric=None, evaluate=False):
        
        if self.architecture == "egcn":
            h = batch["graph"].x
            charges = batch["graph"].atomic_numbers
            
            if self.include_charge:
                h = torch.cat([h, charges.unsqueeze(-1)], dim=1)
            if self.condition_time:
                t_zeropad = torch.zeros(h.shape[0], 1).to(self.device)
                h = torch.cat([h, t_zeropad], dim=1)
            
            x = batch["graph"].pos
            edge_index = batch["graph"].edge_index
            edges = [edge_index[0], edge_index[1]]
            node_mask = None
            edge_mask = None
            h_final, x_final = self.model(
                h, x, edges, node_mask=node_mask, edge_mask=edge_mask, use_embed=True
            )
        elif self.architecture == "egt":

            h = batch["graph"].x
            _, d = h.shape
            charge = batch["graph"].atomic_numbers

            if self.include_charge:
                h = torch.cat([h, charge.unsqueeze(-1)], dim=1)
                d+=1
            if self.condition_time:
                t_zeropad = torch.zeros(h.shape[0], 1).to(self.device)
                h = torch.cat([h, t_zeropad], dim=1)
                d+=1
            x = batch["graph"].pos
            edge_index = batch["graph"].edge_index
            edges = [edge_index[0], edge_index[1]]
                    
            h = self.pad_data(h, batch, d) 
            x = self.pad_data(x, batch, 3)
            bs, n_nodes, _ = x.size()
            
            node_mask = torch.ones(bs, n_nodes).bool().to(self.device) #TODO wrong!               
    
            edge_mask = torch.ones(bs, n_nodes, n_nodes, 1).to(self.device) #TODO wrong!
            y = torch.ones(bs, 1).to(self.device)
            h_final, _, _, _ = self.model(
                h, E=edge_mask, y=y, pos=x, node_mask=node_mask
            )
        
        elif self.architecture == "egnn_extra":
            if self.include_charge:
                h = torch.cat([batch["graph"].x, batch["graph"].atomic_numbers.unsqueeze(-1)], dim=1)
                batch["graph"].x = h
            if self.condition_time:
                t_zeropad = torch.zeros(h.shape[0], 1).to(self.device)
                h = torch.cat([h, t_zeropad], dim=1)
                batch["graph"].x = h
            h_final, _ = self.model(batch["graph"], use_embed=True)    
            

        if self.architecture == "egt":
            h_final = h_final.view(bs, n_nodes, self.model.hidden_nf)
        else:
            h_final = self.pad_data(h_final, batch, self.model.hidden_nf)   
        output = {
            "graph_feature": self.readout_f(h_final),
            "node_feature": h_final,
        }

        if self.load_mlps_layer > 0:
            x = self.mlp(output["graph_feature"])
            pred = self.mlp_final(x)
        else:
            pred = self.mlp(output["graph_feature"])
    
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def get_adj_matrix(self, _edges_dict, n_nodes, batch_size):
        if n_nodes in _edges_dict:
            edges_dic_b = _edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(self.device),
                    torch.LongTensor(cols).to(self.device),
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            _edges_dict[n_nodes] = {}
            return self.get_adj_matrix(_edges_dict, n_nodes, batch_size)

    def pad_data(self, array, batch, dim):
        """"
        array: torch.Tensor of shape (n_atoms, n_features)
        batch: pytorch_geometric.data.Batch
        """
        bs = batch["graph"].batch.max().item() + 1
        natoms = batch["graph"].natoms   
        n_nodes = natoms.max().item()
        array_paddded = torch.zeros(bs, n_nodes, array.shape[1]).to(self.device)
        natom_cum = 0
        for i, natom in enumerate(natoms):
            array_mol = array[natom_cum:natom_cum+natom]
            array_mol = torch.cat([array_mol, torch.zeros(n_nodes-natom, array.shape[1]).to(self.device)], dim=0)    
            array_paddded[i] = array_mol
            natom_cum += natom

        array = array_paddded.view(bs, n_nodes, dim)
        
        return array

    def readout_f(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Perform readout operation over nodes in each molecule.

        Parameters:
        - embeddings (torch.Tensor): Tensor of size (x, y, z) where x is the batch size, y is the number of nodes, and z is the feature size.

        Returns:
        torch.Tensor: Aggregated tensor of size (x, z).
        """
        if self.readout == "sum":
            return embeddings.sum(dim=1)
        elif self.readout == "mean":
            return embeddings.mean(dim=1)
        else:
            raise ValueError("Unsupported method. Choose either 'sum' or 'mean'.")

