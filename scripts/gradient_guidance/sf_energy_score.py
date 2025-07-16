import sys
from math import sqrt
from typing import List

import numpy as np
import torch

# from navicatGA.timeout import timeout
from numpy import dot
from MolecularDiffusion.core import Engine
from torch_geometric.data import Data
from torch_geometric.nn import  radius_graph
from torch_geometric.data import Batch
#%% sf energy score
def distance_to_line(point, line_start, line_end):
    """Calculate the perpendicular distance from a point to a line segment."""
    return np.linalg.norm(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(
        line_end - line_start
    )


def is_within_triangle(x, y, T1_cutoff, S1_cutoff):
    """Check if the point is within the defined triangle area."""
    return (x >= T1_cutoff) and (y >= (2 * x)) and (y <= S1_cutoff)


def pointTriangleDistance(TRI, P):
    """
    # source: #https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e
    # function [dist,PP0] = pointTriangleDistance(TRI,P)
    # calculate distance between a point and a triangle in 3D
    # SYNTAX
    #   dist = pointTriangleDistance(TRI,P)
    #   [dist,PP0] = pointTriangleDistance(TRI,P)
    #
    # DESCRIPTION
    #   Calculate the distance of a given point P from a triangle TRI.
    #   Point P is a row vector of the form 1x3. The triangle is a matrix
    #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
    #   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
    #   to the triangle TRI.
    #   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
    #   closest point PP0 to P on the triangle TRI.
    #
    # Author: Gwolyn Fischer
    # Release: 1.0
    # Release date: 09/02/02
    # Release: 1.1 Fixed Bug because of normalization
    # Release: 1.2 Fixed Bug because of typo in region 5 20101013
    # Release: 1.3 Fixed Bug because of typo in region 2 20101014

    # Possible extention could be a version tailored not to return the distance
    # and additionally the closest point, but instead return only the closest
    # point. Could lead to a small speed gain.

    # Example:
    # %% The Problem
    # P0 = [0.5 -0.3 0.5]
    #
    # P1 = [0 -1 0]
    # P2 = [1  0 0]
    # P3 = [0  0 0]
    #
    # vertices = [P1; P2; P3]
    # faces = [1 2 3]
    #
    # %% The Engine
    # [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0)
    #
    # %% Visualization
    # [x,y,z] = sphere(20)
    # x = dist*x+P0(1)
    # y = dist*y+P0(2)
    # z = dist*z+P0(3)
    #
    # figure
    # hold all
    # patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8)
    # plot3(P0(1),P0(2),P0(3),'b*')
    # plot3(PP0(1),PP0(2),PP0(3),'*g')
    # surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
    # view(3)

    # The algorithm is based on
    # "David Eberly, 'Distance Between Point and Triangle in 3D',
    # Geometric Tools, LLC, (1999)"
    # http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    #
    #        ^t
    #  \     |
    #   \reg2|
    #    \   |
    #     \  |
    #      \ |
    #       \|
    #        *P2
    #        |\
    #        | \
    #  reg3  |  \ reg1
    #        |   \
    #        |reg0\
    #        |     \
    #        |      \ P1
    # -------*-------*------->s
    #        |P0      \
    #  reg4  | reg5    \ reg6
    # rewrite triangle in normal form
    """
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = dot(E0, E0)
    b = dot(E0, E1)
    c = dot(E1, E1)
    d = dot(E0, D)
    e = dot(E1, D)
    f = dot(D, D)

    # print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                        # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f
                    # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = (
                            s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
                        )

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = (
                            s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
                        )

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    dist = sqrt(sqrdistance)

    PP0 = B + s * E0 + t * E1
    return dist, PP0


def energy_score(x, y, S1_cutoff=3.8, scaling_S1=1 / 3.0, T1_cutoff=1.5):
    """
    Calculate the fitness of a point (x, y) with respect to a triangle in a 2D space.

    Parameters:
    - x (float): T1 in eV
    - y (float): S1 in eV
    - S1_cutoff (float, optional): Cutoff value for the S1 axis.
    - scaling_S1 (float, optional): Scaling factor for the S1 axis.
    - T1_cutoff (float, optional): Cutoff value for the T1 axis.

    Returns:
    float: Fitness value. NaN if input is invalid.
    """

    if np.isnan(x) or np.isnan(y):
        return np.nan

    TRIANGLE_SCALING = (
        1.0 / 0.11871871871871865
    )  # Scaling of the energy score, such that the maximum is approx. 1
    p1, p2 = np.array([T1_cutoff, 1]), np.array([T1_cutoff, 100])
    p0, p4 = np.array([0, 0]), np.array([50, 100])
    pS1_1, pS1_2 = np.array([1, S1_cutoff]), np.array([10, S1_cutoff])
    p3 = np.array([x, y])

    if is_within_triangle(x, y, T1_cutoff, S1_cutoff):
        dist1 = TRIANGLE_SCALING * distance_to_line(p3, p1, p2)
        dist2 = TRIANGLE_SCALING * distance_to_line(p3, p0, p4)
        dist3 = scaling_S1 * TRIANGLE_SCALING * distance_to_line(p3, pS1_1, pS1_2)
        return min(dist1, dist2, dist3)
    else:
        TRI = np.array(
            [
                [T1_cutoff, 2 * T1_cutoff, 0.0],
                [T1_cutoff, S1_cutoff, 0.0],
                [S1_cutoff / 2.0, S1_cutoff, 0.0],
            ]
        )
        P = np.array([x, y, 0])
        dist, _ = pointTriangleDistance(TRI, P)
        return -1.0 * TRIANGLE_SCALING * dist


#%% wrapper


class SFEnergyScore:
    def __init__(
        self,
        chkpt_directory: str,
        atom_vocab: List[str] = None,
        norm_factor: List[int] = [1, 4, 10] # as used in training the guidance model
    ):
        self.chkpt_directory = chkpt_directory
        self.atom_vocab = atom_vocab if atom_vocab is not None else ["H", "C", "N", "O", "F"]
        self.norm_factor = norm_factor
        self.solver = self._load_model(self.chkpt_directory)
        self.norm_factor = self.solver.model.norm_values

        self._initialize_model_attributes()

    def _load_model(self, chkpt_path):
        """
        Loads a pre-trained model from a checkpoint file.

        Args:
            chkpt_path (str): The path to the checkpoint file.

        Returns:
            MolecularDiffusion.core.Engine: The loaded Engine object with the model in evaluation mode.
        """
        engine = Engine(None, None, None, None, None)
        engine = engine.load_from_checkpoint(chkpt_path)
        engine.model.eval()
        return engine

    def _initialize_model_attributes(self):
        """
        Initializes atom_vocab, node_feature, std, weight, and mean attributes
        of the solver's model if they are not already present.
        """
        if not hasattr(self.solver.model, 'std'):
            chkpt = torch.load(self.chkpt_directory)
            self.solver.model.std = chkpt["model"]["std"].to(self.solver.model.device)
            self.solver.model.weight = chkpt["model"]["weight"].to(self.solver.model.device)
            self.solver.model.mean = chkpt["model"]["mean"].to(self.solver.model.device)

        if not hasattr(self.solver.model, 'atom_vocab'):
            self.solver.model.atom_vocab = self.atom_vocab
        if not hasattr(self.solver.model, 'node_feature'):
            self.solver.model.node_feature = self.node_feature

    def __call__(self, xh, t):
        """
        Calculates the negative energy score for a given molecular configuration.
        This method makes the class instance callable.
        """
        bs, n_nodes, dims = xh.shape
        RADIUS = 4
        device = xh.device
        
        mol = {} # This dictionary is used to pass graph data to the model
        coords = xh[:, :, :dims] / self.norm_factor[0]
        h = xh[:, :, dims:-1] / self.norm_factor[1]
        charge = xh[:, :, -1] / self.norm_factor[2]
        
        coords = coords.view(n_nodes*bs, dims).to(device)
        h = h.view(n_nodes*bs, -1).to(device)
        charge = charge.view(n_nodes*bs).to(device)

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
        
        graph_data = graph_data.to(device)
        mol["graph"] = graph_data # Assign the batched graph data to the 'graph' key
        preds = self.solver.model.predict(mol, evaluate=True)[0]
        target = -energy_score(preds[1], preds[0])
    
        return target
