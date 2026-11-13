import logging
import os
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from scipy.spatial import distance, Delaunay

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry.rdGeometry import Point3D
    from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils
    PT = Chem.GetPeriodicTable()
except ImportError:
    Chem = None
    AllChem = None
    Point3D = None
    FeatDirUtils = None
    PT = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

logger = logging.getLogger(__name__)

# =============================================================================
# Constants & SMARTS
# =============================================================================

P_TYPES = ('Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'Halogen', 'Cation', 'Anion', 'ZnBinder', 'Dummy')

COULOMB_SCALING = 1e4 / (4 * 55.263 * np.pi)

SMARTS_FEATURES_FDEF = """
AtomType AromR4 [a;r4,!R1&r3]
DefineFeature Arom4 [{AromR4}]1:[{AromR4}]:[{AromR4}]:[{AromR4}]:1
 Family Aromatic
 Weights 1,1,1,1
EndFeature
AtomType AromR5 [a;r5,!R1&r4,!R1&r3]
DefineFeature Arom5 [{AromR5}]1:[{AromR5}]:[{AromR5}]:[{AromR5}]:[{AromR5}]:1
 Family Aromatic
 Weights 1,1,1,1,1
EndFeature
AtomType AromR6 [a;r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom6 [{AromR6}]1:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:1
 Family Aromatic
 Weights 1,1,1,1,1,1
EndFeature
AtomType AromR7 [a;r7,!R1&r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom7 [{AromR7}]1:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:1
 Family Aromatic
 Weights 1,1,1,1,1,1,1
EndFeature
AtomType AromR8 [a;r8,!R1&r7,!R1&r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom8 [{AromR8}]1:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:1
 Family Aromatic
 Weights 1,1,1,1,1,1,1,1
EndFeature
DefineFeature ZnBinder1 [S;D1]-[#6]
  Family ZnBinder
  Weights 1,0
EndFeature
DefineFeature ZnBinder2 [#6]-C(=O)-C-[S;D1]
  Family ZnBinder
  Weights 0,0,1,0,1
EndFeature
DefineFeature ZnBinder3 [#6]-C(=O)-C-C-[S;D1]
  Family ZnBinder
  Weights 0,0,1,0,0,1
EndFeature
DefineFeature ZnBinder4 [#6]-C(=O)-N-[O;D1]
  Family ZnBinder
  Weights 0,0,1,0,1
EndFeature
DefineFeature ZnBinder5 [#6]-C(=O)-[O;D1]
  Family ZnBinder
  Weights 0,0,1,1
EndFeature
DefineFeature ZnBinder6 [#6]-P(=O)(-O)-[C,O,N]-[C,H]
  Family ZnBinder
  Weights 0,0,1,1,0,0
EndFeature
DefineFeature N_HBD [#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]
  Family Donor
  Weights 1
EndFeature
DefineFeature O_HBD [#8!H0&!$([OH][C,S,P]=O)]
  Family Donor
  Weights 1
EndFeature
DefineFeature S_HBD [#16!H0]
  Family Donor
  Weights 1
EndFeature
DefineFeature ChalcDonor [O,S;H1;+0]
  Family Donor
  Weights 1
EndFeature
AtomType N_HBA [#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]
AtomType O_HBA [$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]
AtomType ChalcAcceptor [O;H0;v2;!$(O=N-*)]
Atomtype ChalcAcceptor [O;-;!$(*-N=O)]
Atomtype ChalcAcceptor [o;+0]
AtomType Hydroxyl [O;H1;v2]
DefineFeature HBA [{Hydroxyl},{ChalcAcceptor},{O_HBA},{N_HBA}]
  Family Acceptor
  Weights 1
EndFeature
DefineFeature N_positive [$([NX3+]([CX4])([CX4,#1])[CX4,#1])&!$([NX3]-*=[!#6])]
  Family Cation
  Weights 1
EndFeature
DefineFeature amidino_positive1 [CX3](=[N+])(-N)[!N]
  Family Cation
  Weights 1,1,1,1
EndFeature
DefineFeature amidino_positive2 [CX3](=N)(-[N+])[!N]
  Family Cation
  Weights 1,1,1,1
EndFeature
DefineFeature guanidino_positive1 [N+]=[CX3](N)-N
  Family Cation
  Weights 1,1,1,1
EndFeature
DefineFeature guanidino_positive2 [N+]=[CX3](N)-N
  Family Cation
  Weights 1,1,1,1
EndFeature
DefineFeature guanidino_positive3 N=[CX3]([N+])-N
  Family Cation
  Weights 1,1,1,1
EndFeature
DefineFeature charged_positive [$([+,+2,+3])&!$(*[-,-2,-3])]
  Family Cation
  Weights 1
EndFeature
DefineFeature tetrazole_negative c1nn[nH1]n1
  Family Anion
  Weights 1,1,1,1,1
EndFeature
DefineFeature SP_v4_negative [$([SX4,PX4](=O)(=O)[O-])](=O)(=O)[O-]
  Family Anion
  Weights 1,1,1,1
EndFeature
DefineFeature CSP_v3_negative [$([CX3,SX3,PX3](=O)[O-])](=O)[O-]
  Family Anion
  Weights 1,1,1
EndFeature
DefineFeature charged_negative [$([-,-2,-3])&!$(*[+,+2,+3])]
  Family Anion
  Weights 1
EndFeature
AtomType Fluorine [F;$(F-[#6]);!$(FC[F,Cl,Br,I])]
AtomType Chlorine [Cl;$(Cl-[#6]);!$(FC[F,Cl,Br,I])]
AtomType Bromine [Br;$(Br-[#6]);!$(FC[F,Cl,Br,I])]
AtomType Iodine [I;$(I-[#6]);!$(FC[F,Cl,Br,I])]
DefineFeature Halogen [{Fluorine},{Chlorine},{Bromine},{Iodine}]
  Family Halogen
  Weights 1
EndFeature
"""

# =============================================================================
# Surface Generation
# =============================================================================

def get_atomic_vdw_radii(mol) -> np.ndarray:
    if PT is None:
        vdw_dict = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.8,
                    16: 1.8, 17: 1.75, 35: 1.85, 53: 1.98}
        return np.array([vdw_dict.get(a.GetAtomicNum(), 1.7) for a in mol.GetAtoms()], dtype=np.float32)
    radii = np.zeros((mol.GetNumAtoms(),), dtype=np.float32)
    for i in range(len(radii)):
        radii[i] = PT.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
    return radii

def _get_points_fibonacci(num_samples: int):
    offset = 2.0 / num_samples
    increment = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(num_samples)
    y = ((i * offset) - 1) + (offset / 2)
    r = np.sqrt(1 - np.square(y))
    phi = np.mod((i + 1), num_samples) * increment
    x = np.cos(phi) * r
    z = np.sin(phi) * r
    return np.column_stack((x, y, z)).astype(np.float32)

def sample_molecular_surface_with_radius_fibonacci(centers: np.ndarray, radii: np.ndarray,
                                                   probe_radius: float = 1.2,
                                                   num_samples_per_atom: int = 20) -> np.ndarray:
    if num_samples_per_atom > 50:
        num_samples_per_atom = 50
    surf_radii = radii + probe_radius
    n_samples = np.ceil((num_samples_per_atom * (radii / 1.7))**2).astype(int)
    
    all_points = []
    spheres = {n: _get_points_fibonacci(n) for n in set(n_samples)}
    
    for i, n in enumerate(n_samples):
        pts = (spheres[n] * surf_radii[i]) + centers[i]
        all_points.append(pts)
    
    return np.concatenate(all_points, axis=0)

def get_molecular_surface(centers: np.ndarray, radii: np.ndarray,
                          num_points: Optional[int] = None,
                          num_samples_per_atom: int = 25,
                          probe_radius: float = 1.2) -> np.ndarray:
    points = sample_molecular_surface_with_radius_fibonacci(centers, radii, probe_radius, num_samples_per_atom)
    
    # Filter points interior to atoms
    dist_matrix = distance.cdist(points, centers)
    mask = np.all(dist_matrix >= (radii + probe_radius - 0.01), axis=1)
    points = points[mask]
    
    if num_points is not None and len(points) > 0:
        if o3d is not None:
             pcd = o3d.geometry.PointCloud()
             pcd.points = o3d.utility.Vector3dVector(points)
             pcd.estimate_normals()
             mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                 pcd, o3d.utility.DoubleVector([1.2]))
             if mesh.has_triangles():
                 pcd_sampled = mesh.sample_points_poisson_disk(num_points)
                 return np.asarray(pcd_sampled.points, dtype=np.float32)

        # Fallback to random sampling if o3d fails or missing
        if len(points) > num_points:
            idx = np.random.choice(len(points), num_points, replace=False)
            points = points[idx]
        elif len(points) < num_points:
            pad = np.tile(centers.mean(0), (num_points - len(points), 1))
            points = np.concatenate([points, pad], axis=0)
            
    return points.astype(np.float32)

# =============================================================================
# ESP Calculation
# =============================================================================

def get_electrostatics_given_point_charges(charges: np.ndarray, positions: np.ndarray, points: np.ndarray) -> np.ndarray:
    distances = distance.cdist(points, positions)
    E_pot = np.dot(1.0 / np.maximum(distances, 1e-6), charges) * COULOMB_SCALING
    E_pot[np.isinf(E_pot)] = 0
    return E_pot.astype(np.float32)

# =============================================================================
# Pharmacophores Detection
# =============================================================================

__hydrophobic_smarts = [
    "a1aaaaa1", "a1aaaa1",
    "[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
    "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    "*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    "[C&r3]1~[C&r3]~[C&r3]1", "[C&r4]1~[C&r4]~[C&r4]~[C&r4]1",
    "[C&r5]1~[C&r5]~[C&r5]~[C&r5]~[C&r5]1", "[C&r6]1~[C&r6]~[C&r6]~[C&r6]~[C&r6]~[C&r6]1",
    "[C&r7]1~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]1", "[C&r8]1~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]1",
    "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
    "[$([CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
    "[$([S]~[#6])&!$(S~[!#6])]"
]

def find_hydrophobes(mol, cluster_hydrophobic: bool = True):
    if Chem is None: return []
    patterns = [Chem.MolFromSmarts(s) for s in __hydrophobic_smarts]
    all_h = []
    conf = mol.GetConformer()
    for pat in patterns:
        for match in mol.GetSubstructMatches(pat):
            pts = [conf.GetAtomPosition(idx) for idx in match]
            all_h.append(np.mean([[p.x, p.y, p.z] for p in pts], axis=0))
    
    if not cluster_hydrophobic or not all_h:
        return all_h
        
    # Regroup within 2.0A
    all_h = np.array(all_h)
    n = len(all_h)
    clusters = list(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(all_h[i] - all_h[j]) <= 2.0:
                clusters[j] = clusters[i]
    
    results = []
    for cid in set(clusters):
        results.append(np.mean(all_h[np.array(clusters) == cid], axis=0))
    return results

# =============================================================================
# PharmVec Ports
# =============================================================================

def GetAromaticFeatVects(conf, featAtoms, featLoc, return_both: bool = False, scale=1.0):
 head = featLoc
 ats = [conf.GetAtomPosition(x) for x in featAtoms]
 v1 = ats[0] - head
 v2 = ats[1] - head
 norm1 = v1.CrossProduct(v2)
 norm1.Normalize()
 norm1 *= scale
 if return_both:
  norm2 = Point3D(0, 0, 0) - norm1
  return [head]*2, [norm1, norm2]
 return [head], [norm1]

def GetDonorFeatVects(conf, featAtoms, scale=1., exclude=[]):
 aid = featAtoms[0]
 mol = conf.GetOwningMol()
 cpt = conf.GetAtomPosition(aid)
 nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()
 hyd_nbrs = []
 vectors = []
 for nbr in nbrs:
  if nbr.GetAtomicNum() == 1:
   if nbr.GetIdx() in exclude: continue
   hyd_nbrs.append(nbr)
   vec = conf.GetAtomPosition(nbr.GetIdx()) - cpt
   vec.Normalize()
   vec *= scale
   vectors.append(vec)
 if hyd_nbrs:
  return [cpt]*len(vectors), vectors, hyd_nbrs
 return None, None, None

def GetAcceptorFeatVects(conf, featAtoms, scale=1.0):
 aid = featAtoms[0]
 mol = conf.GetOwningMol()
 atom = mol.GetAtomWithIdx(aid)
 nbrs = atom.GetNeighbors()
 nlp = 3
 if atom.GetAtomicNum() == 7: nlp = 1
 elif atom.GetAtomicNum() == 8: nlp = 2
 cpt = conf.GetAtomPosition(aid)
 heavy = [n for n in nbrs if n.GetAtomicNum() > 1]
 
 if nlp == 1:
  if len(nbrs) == 1:
   v = conf.GetAtomPosition(nbrs[0].GetIdx()) - cpt
   v.Normalize(); v *= (-1.0 * scale)
   return [cpt], [v]
  elif len(nbrs) == 2:
   v = FeatDirUtils._findAvgVec(conf, cpt, nbrs)
   v *= (-1.0 * scale)
   return [cpt], [v]
  elif len(nbrs) == 3:
   out = FeatDirUtils._GetTetrahedralFeatVect(conf, aid, scale)
   if out:
    v = out[0][1] - cpt; v.Normalize(); v *= scale
    return [cpt], [v]
 elif nlp == 2:
  if len(nbrs) == 1:
   hn = heavy[0]
   hnn = [n for n in hn.GetNeighbors() if n.GetIdx() != aid][0]
   pt1 = conf.GetAtomPosition(hnn.GetIdx())
   v1 = conf.GetAtomPosition(hn.GetIdx())
   pt1 -= v1; v1 -= cpt
   ra = v1.CrossProduct(pt1); ra.Normalize()
   bv1 = FeatDirUtils.ArbAxisRotation(120, ra, v1); bv1.Normalize(); bv1 *= scale
   bv2 = FeatDirUtils.ArbAxisRotation(-120, ra, v1); bv2.Normalize(); bv2 *= scale
   return [cpt]*2, [bv1, bv2]
  if len(nbrs) == 2:
   bv = FeatDirUtils._findAvgVec(conf, cpt, nbrs); bv *= (-1.0 * scale)
   v1 = conf.GetAtomPosition(nbrs[0].GetIdx()) - cpt
   v2 = conf.GetAtomPosition(nbrs[1].GetIdx()) - cpt
   ra = v1 - v2; ra.Normalize()
   bv1 = FeatDirUtils.ArbAxisRotation(54.5, ra, bv); bv2 = FeatDirUtils.ArbAxisRotation(-54.5, ra, bv)
   bv1.Normalize(); bv2.Normalize(); bv1 *= scale; bv2 *= scale
   return [cpt]*2, [bv1, bv2]
 elif nlp == 3:
  v = conf.GetAtomPosition(heavy[0].GetIdx()) - cpt; v.Normalize(); v *= (-1.0 * scale)
  return [cpt], [v]
 return None, None

def GetHalogenFeatVects(conf, featAtoms, scale=1.0):
 aid = featAtoms[0]
 mol = conf.GetOwningMol()
 nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()
 cpt = conf.GetAtomPosition(aid)
 v = conf.GetAtomPosition(nbrs[0].GetIdx()) - cpt; v.Normalize(); v *= (-1.0 * scale)
 return [cpt], [v]

# =============================================================================
# Main Pharmacophore Entry Point
# =============================================================================

def get_pharmacophores(mol, multi_vector: bool = True, exclude: List[int] = [],
                       check_access: bool = False, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if Chem is None: return np.zeros(0), np.zeros((0,3)), np.zeros((0,3))
    
    factory = AllChem.BuildFeatureFactoryFromString(SMARTS_FEATURES_FDEF)
    mol_feats = factory.GetFeaturesForMol(mol)
    keep = ('Aromatic', 'ZnBinder', 'Donor', 'Acceptor', 'Cation', 'Anion', 'Halogen')
    
    X, P, V = [], [], []
    conf = mol.GetConformer()
    
    for feat in mol_feats:
        family = feat.GetFamily()
        if family not in keep: continue
        pos = feat.GetPos()
        
        anchor, vec = None, None
        if family == 'Aromatic':
            anchor, vec = GetAromaticFeatVects(conf, feat.GetAtomIds(), pos, multi_vector, scale)
        elif family == 'Donor':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                anchor, vec, _ = GetDonorFeatVects(conf, aids, scale, exclude)
        elif family == 'Acceptor':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                anchor, vec = GetAcceptorFeatVects(conf, aids, scale)
        elif family == 'Halogen':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                anchor, vec = GetHalogenFeatVects(conf, aids, scale)
        else:
            anchor, vec = [pos], [Point3D(0,0,0)]
            
        if anchor and vec:
            for a, v in zip(anchor, vec):
                X.append(P_TYPES.index(family))
                P.append([a.x, a.y, a.z])
                V.append([v.x, v.y, v.z])
                
    # Hydrophobes
    h_centers = find_hydrophobes(mol, cluster_hydrophobic=True)
    for h in h_centers:
        X.append(P_TYPES.index('Hydrophobe'))
        P.append(h)
        V.append([0, 0, 0])
        
    if not X:
        return np.zeros(0, dtype=np.int64), np.zeros((0,3), dtype=np.float32), np.zeros((0,3), dtype=np.float32)
    return np.array(X, dtype=np.int64), np.array(P, dtype=np.float32), np.array(V, dtype=np.float32)

def update_mol_coordinates(mol, pos):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(pos[i,0]), float(pos[i,1]), float(pos[i,2])))
    return mol
