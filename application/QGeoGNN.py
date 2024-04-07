import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
import argparse
import warnings
import random
from rdkit.Chem.Descriptors import rdMolDescriptors
import pandas as pd
import os
from mordred import Calculator, descriptors, is_missing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import gradio as gr
from utils import *
import pubchempy as pcp
#v1013
DAY_LIGHT_FG_SMARTS_LIST = [
        # C
        "[CX4]",
        "[$([CX2](=C)=C)]",
        "[$([CX3]=[CX3])]",
        "[$([CX2]#C)]",
        # C & O
        "[CX3]=[OX1]",
        "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "[CX3](=[OX1])C",
        "[OX1]=CN",
        "[CX3](=[OX1])O",
        "[CX3](=[OX1])[F,Cl,Br,I]",
        "[CX3H1](=O)[#6]",
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "[NX3][CX3](=[OX1])[#6]",
        "[NX3][CX3]=[NX3+]",
        "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "[NX3][CX3](=[OX1])[OX2H0]",
        "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
        "[CX3](=O)[O-]",
        "[CX3](=[OX1])(O)O",
        "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
        "C[OX2][CX3](=[OX1])[OX2]C",
        "[CX3](=O)[OX2H1]",
        "[CX3](=O)[OX1H0-,OX2H1]",
        "[NX3][CX2]#[NX1]",
        "[#6][CX3](=O)[OX2H0][#6]",
        "[#6][CX3](=O)[#6]",
        "[OD2]([#6])[#6]",
        # H
        "[H]",
        "[!#1]",
        "[H+]",
        "[+H]",
        "[!H]",
        # N
        "[NX3;H2,H1;!$(NC=O)]",
        "[NX3][CX3]=[CX3]",
        "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
        "[NX3][$(C=C),$(cc)]",
        "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
        "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
        "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]",
        "[CH2X4][CX3](=[OX1])[NX3H2]",
        "[CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[CH2X4][SX2H,SX1H0-]",
        "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]",
        "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1",
        "[CHX4]([CH3X4])[CH2X4][CH3X4]",
        "[CH2X4][CHX4]([CH3X4])[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
        "[CH2X4][CH2X4][SX2][CH3X4]",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
        "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH2X4][OX2H]",
        "[NX3][CX3]=[SX1]",
        "[CHX4]([CH3X4])[OX2H]",
        "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1",
        "[CHX4]([CH3X4])[CH3X4]",
        "N[CX4H2][CX3](=[OX1])[O,N]",
        "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
        "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
        "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
        "[#7]",
        "[NX2]=N",
        "[NX2]=[NX2]",
        "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
        "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
        "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "[NX3][NX3]",
        "[NX3][NX2]=[*]",
        "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "[NX3+]=[CX3]",
        "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
        "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]",
        "[NX1]#[CX2]",
        "[CX1-]#[NX2+]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[NX2]=[OX1]",
        "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
        # O
        "[OX2H]",
        "[#6][OX2H]",
        "[OX2H][CX3]=[OX1]",
        "[OX2H]P",
        "[OX2H][#6X3]=[#6]",
        "[OX2H][cX3]:[c]",
        "[OX2H][$(C=C),$(cc)]",
        "[$([OH]-*=[!#6])]",
        "[OX2,OX1-][OX2,OX1-]",
        # P
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        # S
        "[S-][CX3](=S)[#6]",
        "[#6X3](=[SX1])([!N])[!N]",
        "[SX2]",
        "[#16X2H]",
        "[#16!H0]",
        "[#16X2H0]",
        "[#16X2H0][!#16]",
        "[#16X2H0][#16X2H0]",
        "[#16X2H0][!#16].[#16X2H0][!#16]",
        "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
        "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "[SX4](C)(C)(=O)=N",
        "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
        "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
        "[#16X2][OX2H,OX1H0-]",
        "[#16X2][OX2H0]",
        # X
        "[#6][F,Cl,Br,I]",
        "[F,Cl,Br,I]",
        "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
    ]
device = 'cuda'  # args.device
Eluent_smiles=['CCCCCC','CC(OCC)=O','C(Cl)Cl','CO','CCOCC']
def convert_e(e):
    if e=='PE':
        new_e=0
    elif e=='EA':
        new_e=1
    elif e=='DCM':
        new_e=2
    else:
        print(e)
    return new_e
def get_descriptor(smiles,ratio):
    compound_mol = Chem.MolFromSmiles(smiles)
    descriptor=[]
    descriptor.append(Descriptors.ExactMolWt(compound_mol))
    descriptor.append(Chem.rdMolDescriptors.CalcTPSA(compound_mol))
    descriptor.append(Descriptors.NumRotatableBonds(compound_mol))  # Number of rotable bonds
    descriptor.append(Descriptors.NumHDonors(compound_mol))  # Number of H bond donors
    descriptor.append(Descriptors.NumHAcceptors(compound_mol)) # Number of H bond acceptors
    descriptor.append(Descriptors.MolLogP(compound_mol)) # LogP
    descriptor=np.array(descriptor)*ratio
    return descriptor
def get_eluent_descriptor(eluent_array):
    eluent=eluent_array
    des = np.zeros([6,])
    for i in range(eluent.shape[0]):
        if eluent[i] != 0:
            e_descriptors = get_descriptor(Eluent_smiles[i], eluent[i])
            des+=e_descriptors
    return des
def convert_eluent(eluent):
    ratio=[]
    PE=int(eluent.split('/')[0])
    EA=int(eluent.split('/')[1])
    ratio.append(get_eluent_descriptor(np.array([PE,EA,0,0,0])/(PE+EA)))
    return np.vstack(ratio)
def convert_CAS_to_smile(cas):
    try:
        # 搜索化合物信息
        c = pcp.get_compounds(cas, 'name')
        # 提取SMILES字符串
        smiles = c[0].isomeric_smiles
        return smiles
    except Exception as e:
        print("An error occurred:", e)
        return None
def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    Args:
        mol: rdkit mol object.
        n_iter(int): number of iterations. Default 12.
    Returns:
        list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """
    Args:
        smiles: smiles sequence.
    Returns:
        inchi.
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if not mol is None:  # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def check_smiles_validity(smiles):
    """
    Check whether the smile can't be converted to rdkit mol object.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception as e:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively.
    Args:
        mol: rdkit mol object.
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one.
    Args:
        mol_list(list): a list of rdkit mol object.
    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


def get_atom_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


def get_bond_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]


class CompoundKit(object):
    """
    CompoundKit
    """
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],

        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }
    # float features
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    ### functional groups
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    ### atom

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    ### fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)

            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """ tbd """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        TODO: to be remove in the future
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])

        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


class Compound3DKit(object):
    """the 3Dkit of Compound"""

    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return mol,atom_poses
        # try:
        #     new_mol = Chem.AddHs(mol)
        #     res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
        #     ### MMFF generates multiple conformations
        #     res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        #     new_mol = Chem.RemoveHs(new_mol)
        #     index = np.argmin([x[1] for x in res])
        #     energy = res[index][1]
        #     conf = new_mol.GetConformer(id=int(index))
        # except:
        #     new_mol = mol
        #     AllChem.Compute2DCoords(new_mol)
        #     energy = 0
        #     conf = new_mol.GetConformer()
        #
        # atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        # if return_energy:
        #     return new_mol, atom_poses, energy
        # else:
        #     return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""

        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0, ], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles, bond_angle_dirs


def new_smiles_to_graph_data(smiles, **kwargs):
    """
    Convert smiles to graph data.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = new_mol_to_graph_data(mol)
    return data


def new_mol_to_graph_data(mol):
    """
    mol_to_graph_data
    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    ### bond and bond features
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2

    #### self loop
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = get_bond_feature_dims([name])[0] - 1  # self loop: value = len - 1
        data[name] += [bond_feature_id] * N

    ### make ndarray and check length
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], 'int64')
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_graph_data(mol):
    """
    mol_to_graph_data
    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence",
        "formal_charge", "hybridization", "implicit_valence",
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]

    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    ### atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    ### bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1  # 0: OOV
            data[name] += [bond_feature_id] * 2

    ### self loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2  # N + 2: self loop
        data[name] += [bond_feature_id] * N

    ### check whether edge exists
    if len(data['edges']) == 0:  # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### make ndarray and check length
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_geognn_graph_data(mol, atom_poses, dir_type):
    """
    mol: rdkit molecule
    dir_type: direction type for bond_angle grpah
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol)

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
        Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['bond_angle'] = np.array(bond_angles, 'float32')
    return data


def mol_to_geognn_graph_data_MMFF3d(mol):
    """tbd"""
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')


def mol_to_geognn_graph_data_raw3d(mol):
    """tbd"""
    atom_poses = Compound3DKit.get_atom_poses(mol, mol.GetConformer())
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')

def obtain_3D_mol(smiles,name):
    mol = AllChem.MolFromSmiles(smiles)
    new_mol = Chem.AddHs(mol)
    res = AllChem.EmbedMultipleConfs(new_mol)
    ### MMFF generates multiple conformations
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
    new_mol = Chem.RemoveHs(new_mol)
    Chem.MolToMolFile(new_mol, name+'.mol')
    return new_mol

def measurement(y_test, y_pred, method,c='blue',s=15):
    MSE = np.sum(np.abs(y_test - y_pred) ** 2) / y_test.shape[0]
    RMSE = np.sqrt(MSE)
    MAE = np.sum(np.abs(y_test - y_pred)) / y_test.shape[0]
    R_square = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 1, figsize=(2, 2), dpi=300)
    plt.plot(np.linspace(0, max(np.max(y_test), np.max(y_pred)), 100),
             np.linspace(0, max(np.max(y_test), np.max(y_pred)), 100), linewidth=1, linestyle='--', color='black')

    plt.scatter(y_test, y_pred, c=c, s=s, alpha=0.4)
    # plt.xticks(x,[1,4,5,7,8,13,15,50,80,90,114,148,165],fontproperties='Arial',size=7)
    plt.xticks(fontproperties='Arial', size=7)
    plt.yticks(fontproperties='Arial', size=7)
    if method!='TLC':
        plt.ylim(-2, max(np.max(y_test), np.max(y_pred)) * 1.1)
        plt.xlim(-2, max(np.max(y_test), np.max(y_pred)) * 1.1)
    if method == 'TLC':
        plt.ylim(-0.1,  1.1)
        plt.xlim(-0.1, 1.1)
    #plt.grid(ls='--')
    #plt.xlabel('True value', fontproperties='Arial', size=7)
    #plt.ylabel("Predict value", fontproperties='Arial', size=7)
    #plt.title(f"Predict v.s. True", fontproperties='Arial', size=7)
    from matplotlib.ticker import MaxNLocator
    axes.yaxis.set_major_locator(MaxNLocator(4))
    axes.xaxis.set_major_locator(MaxNLocator(4))
    plt.savefig(f'plot_save/R_{method}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'plot_save/R_{method}.pdf', bbox_inches='tight', dpi=300)
    print([MSE, RMSE, MAE, R_square])
    return [MSE, RMSE, MAE, R_square],fig
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')

#============Parameter setting===============
MODEL = 'Test'  #['Train','Test','Test_other_method','Test_enantiomer','Test_excel']
test_mode='fixed' #fixed or random or enantiomer(extract enantimoers)
split_mode='data'#'compound'
Use_geometry_enhanced=True   #default:True
Use_column_info=False#True #default: True

atom_id_names = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]
bond_id_names = [
    "bond_dir", "bond_type", "is_in_ring"]

if Use_geometry_enhanced==True:
    bond_float_names = ["bond_length",'prop','e','m','V_e']

if Use_geometry_enhanced==False:
    bond_float_names=['prop','e','m','V_e']

bond_angle_float_names = ['bond_angle', '153', '278', '884', '885', '1273', '1594', '431', '1768','1769', '1288', '1521',
                          'MolWt','nRotB', 'HBD','HBA', 'LogP']


full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)


if Use_column_info==True:
    bond_float_names.extend(['diameter','column_length','density'])


calc = Calculator(descriptors, ignore_3D=False)


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)  # 不同维度的属性用不同的Embedding方法
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding

class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim + 5, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

class RBF(torch.nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = centers.reshape([1, -1])
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.reshape([-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))

class BondFloatRBF(torch.nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (nn.Parameter(torch.arange(0, 2, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
                # (centers, gamma)
                'prop': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'e': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'V_e': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'm': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
                'diameter': (nn.Parameter(torch.arange(0, 4, 0.2)), nn.Parameter(torch.Tensor([1.0]))),
                'column_length': (nn.Parameter(torch.arange(0, 30., 1)), nn.Parameter(torch.Tensor([1.0]))),
                'density': (nn.Parameter(torch.arange(0, 1, 0.05)), nn.Parameter(torch.Tensor([1.0]))),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers.to(device), gamma.to(device))
            self.rbf_list.append(rbf)
            linear = torch.nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[:, i].reshape(-1, 1)
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed

class BondAngleFloatRBF(torch.nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (nn.Parameter(torch.arange(0, torch.pi, 0.1)), nn.Parameter(torch.Tensor([10.0]))),
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = torch.nn.ModuleList()
        self.rbf_list = torch.nn.ModuleList()
        for name in self.bond_angle_float_names:
            if name == 'bond_angle':
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers.to(device), gamma.to(device))
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim)
                self.linear_list.append(linear)
            else:
                linear = nn.Linear(len(self.bond_angle_float_names) - 1, embed_dim)
                self.linear_list.append(linear)
                break

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            if name == 'bond_angle':
                x = bond_angle_float_features[:, i].reshape(-1, 1)
                rbf_x = self.rbf_list[i](x)
                out_embed += self.linear_list[i](rbf_x)
            else:
                x = bond_angle_float_features[:, 1:]
                out_embed += self.linear_list[i](x)
                break
        return out_embed

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                                 nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = edge_attr
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

# GNN to generate node embedding
class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", residual=False):
        """GIN Node Embedding Module
        采用多层GINConv实现图上结点的嵌入。
        """

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder=BondEncoder(emb_dim)
        self.bond_float_encoder=BondFloatRBF(bond_float_names,emb_dim)
        self.bond_angle_encoder=BondAngleFloatRBF(bond_angle_float_names,emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.convs_bond_angle=torch.nn.ModuleList()
        self.convs_bond_float=torch.nn.ModuleList()
        self.convs_bond_embeding=torch.nn.ModuleList()
        self.convs_angle_float=torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_ba = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.convs_bond_angle.append(GINConv(emb_dim))
            self.convs_bond_embeding.append(BondEncoder(emb_dim))
            self.convs_bond_float.append(BondFloatRBF(bond_float_names,emb_dim))
            self.convs_angle_float.append(BondAngleFloatRBF(bond_angle_float_names,emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_ba.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_atom_bond,batched_bond_angle):
        x, edge_index, edge_attr = batched_atom_bond.x, batched_atom_bond.edge_index, batched_atom_bond.edge_attr
        edge_index_ba,edge_attr_ba= batched_bond_angle.edge_index, batched_bond_angle.edge_attr
        # computing input node embedding
        h_list = [self.atom_encoder(x)]  # 先将类别型原子属性转化为原子嵌入

        if Use_geometry_enhanced==True:
            h_list_ba=[self.bond_float_encoder(edge_attr[:,len(bond_id_names):edge_attr.shape[1]+1].to(torch.float32))+self.bond_encoder(edge_attr[:,0:len(bond_id_names)].to(torch.int64))]
            for layer in range(self.num_layers):
                h = self.convs[layer](h_list[layer], edge_index, h_list_ba[layer])
                cur_h_ba=self.convs_bond_embeding[layer](edge_attr[:,0:len(bond_id_names)].to(torch.int64))+self.convs_bond_float[layer](edge_attr[:,len(bond_id_names):edge_attr.shape[1]+1].to(torch.float32))
                cur_angle_hidden=self.convs_angle_float[layer](edge_attr_ba)
                h_ba=self.convs_bond_angle[layer](cur_h_ba, edge_index_ba, cur_angle_hidden)

                if layer == self.num_layers - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                    h_ba = F.dropout(h_ba, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                    h_ba = F.dropout(F.relu(h_ba), self.drop_ratio, training=self.training)
                if self.residual:
                    h += h_list[layer]
                    h_ba+=h_list_ba[layer]
                h_list.append(h)
                h_list_ba.append(h_ba)


            # Different implementations of Jk-concat
            if self.JK == "last":
                node_representation = h_list[-1]
                edge_representation = h_list_ba[-1]
            elif self.JK == "sum":
                node_representation = 0
                edge_representation = 0
                for layer in range(self.num_layers + 1):
                    node_representation += h_list[layer]
                    edge_representation += h_list_ba[layer]

            return node_representation,edge_representation
        if Use_geometry_enhanced==False:
            for layer in range(self.num_layers):
                h = self.convs[layer](h_list[layer], edge_index,
                                      self.convs_bond_embeding[layer](edge_attr[:, 0:len(bond_id_names)].to(torch.int64)) +
                                      self.convs_bond_float[layer](
                                          edge_attr[:, len(bond_id_names):edge_attr.shape[1] + 1].to(torch.float32)))
                h = self.batch_norms[layer](h)
                if layer == self.num_layers - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

                if self.residual:
                    h += h_list[layer]

                h_list.append(h)

            # Different implementations of Jk-concat
            if self.JK == "last":
                node_representation = h_list[-1]
            elif self.JK == "sum":
                node_representation = 0
                for layer in range(self.num_layers + 1):
                    node_representation += h_list[layer]

            return node_representation

class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=2, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last", graph_pooling="attention",
                 descriptor_dim=1781):
        """GIN Graph Pooling Module

        此模块首先采用GINNodeEmbedding模块对图上每一个节点做嵌入，然后对节点嵌入做池化得到图的嵌入，最后用一层线性变换得到图的最终的表示（graph representation）。

        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表示的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.descriptor_dim=descriptor_dim
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GINNodeEmbedding(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool

        elif graph_pooling == "mean":
            self.pool = global_mean_pool

        elif graph_pooling == "max":
            self.pool = global_max_pool

        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))


        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Sequential(nn.Linear(self.emb_dim, self.num_tasks),
                                                   nn.ReLU())

        self.NN_descriptor = nn.Sequential(nn.Linear(self.descriptor_dim, self.emb_dim),
                                           nn.Sigmoid(),
                                           nn.Linear(self.emb_dim, self.emb_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, batched_atom_bond,batched_bond_angle):
        if Use_geometry_enhanced==True:
            h_node,h_node_ba= self.gnn_node(batched_atom_bond,batched_bond_angle)
        else:
            h_node= self.gnn_node(batched_atom_bond, batched_bond_angle)
        h_graph = self.pool(h_node, batched_atom_bond.batch)
        output = self.graph_pred_linear(h_graph)
        if self.training:
            return output,h_graph
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=1e8),h_graph

def mord(mol, nBits=1826, errors_as_zeros=True):
    try:
        result = calc(mol)
        desc_list = [r if not is_missing(r) else 0 for r in result]
        np_arr = np.array(desc_list)
        return np_arr
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)

def q_loss(q,y_true,y_pred):
    e = (y_true-y_pred)
    return torch.mean(torch.maximum(q*e, (q-1)*e))

def load_3D_mol():
    dir = 'mol_save/'
    for root, dirs, files in os.walk(dir):
        file_names = files
    file_names.sort(key=lambda x: int(x[x.find('_') + 5:x.find(".")]))  # 按照前面的数字字符排序
    mol_save = []
    for file_name in file_names:
        mol_save.append(Chem.MolFromMolFile(dir + file_name))
    return mol_save

def prepartion(args):
    save_dir = os.path.join('saves', args.task_name)
    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--task_name', type=str, default='GINGraphPooling',
                        help='task name')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset_root', type=str, default="dataset",
                        help='dataset root')
    args = parser.parse_args()

    return args

def calc_dragon_type_desc(mol):
    compound_mol = mol
    compound_MolWt = Descriptors.ExactMolWt(compound_mol)
    compound_TPSA = Chem.rdMolDescriptors.CalcTPSA(compound_mol)
    compound_nRotB = Descriptors.NumRotatableBonds(compound_mol)  # Number of rotable bonds
    compound_HBD = Descriptors.NumHDonors(compound_mol)  # Number of H bond donors
    compound_HBA = Descriptors.NumHAcceptors(compound_mol)  # Number of H bond acceptors
    compound_LogP = Descriptors.MolLogP(compound_mol)  # LogP
    return rdMolDescriptors.CalcAUTOCORR3D(mol) + rdMolDescriptors.CalcMORSE(mol) + \
           rdMolDescriptors.CalcRDF(mol) + rdMolDescriptors.CalcWHIM(mol) + \
           [compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP]


def save_3D_mol(all_smile,mol_save_dir):
    index=0
    bad_conformer=[]
    pbar=tqdm(all_smile)
    try:
        os.makedirs(f'{mol_save_dir}')
    except OSError:
        pass

    list_files=os.listdir(mol_save_dir)
    for smiles in pbar:
        if f'3D_mol_{index}.mol' not in list_files:
            try:
                obtain_3D_mol(smiles,f'{mol_save_dir}/3D_mol_{index}')
            except ValueError:
                bad_conformer.append(index)
                index += 1
                continue
            index += 1
        else:
            index+=1
    return bad_conformer

def save_dataset(charity_smile,mol_save_dir,name,bad_conformer):
    dataset=[]
    dataset_mord=[]
    dataset_attribute=[]
    pbar = tqdm(charity_smile)
    index=0
    for smile in pbar:
        if index in bad_conformer:
            index+=1
            continue
        mol=Chem.MolFromMolFile(f"{mol_save_dir}/3D_mol_{index}.mol")
        #mol = AllChem.MolFromSmiles(smile)
        descriptor=mord(mol)
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        dataset.append(data)
        dataset_mord.append(descriptor)
        MolWt = Descriptors.ExactMolWt(mol)
        nRotB = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds
        HBD = Descriptors.NumHDonors(mol)  # Number of H bond donors
        HBA = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
        LogP = Descriptors.MolLogP(mol)  # LogP
        dataset_attribute.append([MolWt,nRotB,HBD,HBA,LogP])
        index+=1

    dataset_mord=np.array(dataset_mord)
    dataset_attribute=np.array(dataset_attribute)
    np.save(f"dataset_save/dataset_{name}.npy",dataset,allow_pickle=True)
    np.save(f'dataset_save/dataset_morder_{name}.npy',dataset_mord)
    np.save(f'dataset_save/dataset_attribute_{name}.npy',dataset_attribute)


def eval(model, device, loader_atom_bond,loader_bond_angle):
    model.eval()
    y_true_1 = []
    y_pred_1 = []
    y_pred_10_1=[]
    y_pred_90_1=[]
    y_true_2 = []
    y_pred_2 = []
    y_pred_10_2=[]
    y_pred_90_2=[]
    with torch.no_grad():
        for _, batch in enumerate(zip(loader_atom_bond,loader_bond_angle)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            pred = model(batch_atom_bond,batch_bond_angle)[0]

            y_true_1.append(batch_atom_bond.y[:,0].detach().cpu().reshape(-1))
            y_pred_1.append(pred[:,1].detach().cpu())
            y_pred_10_1.append(pred[:,0].detach().cpu())
            y_pred_90_1.append(pred[:,2].detach().cpu())
            y_true_2.append(batch_atom_bond.y[:, 1].detach().cpu().reshape(-1))
            y_pred_2.append(pred[:, 4].detach().cpu())
            y_pred_10_2.append(pred[:, 3].detach().cpu())
            y_pred_90_2.append(pred[:, 5].detach().cpu())
    y_true_1 = torch.cat(y_true_1, dim=0)
    y_pred_1 = torch.cat(y_pred_1, dim=0)
    y_pred_10_1 = torch.cat(y_pred_10_1, dim=0)
    y_pred_90_1 = torch.cat(y_pred_90_1, dim=0)
    y_true_2 = torch.cat(y_true_2, dim=0)
    y_pred_2 = torch.cat(y_pred_2, dim=0)
    y_pred_10_2 = torch.cat(y_pred_10_2, dim=0)
    y_pred_90_2 = torch.cat(y_pred_90_2, dim=0)

    return torch.mean((y_true_1 - y_pred_1) ** 2).data.numpy(),torch.mean((y_true_2 - y_pred_2) ** 2).data.numpy()

def test(model, device, loader_atom_bond,loader_bond_angle):
    model.eval()
    y_true_1 = []
    y_pred_1 = []
    y_pred_10_1 = []
    y_pred_90_1 = []
    y_true_2 = []
    y_pred_2 = []
    y_pred_10_2 = []
    y_pred_90_2 = []
    with torch.no_grad():
        for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            pred = model(batch_atom_bond, batch_bond_angle)[0]

            y_true_1.append(batch_atom_bond.y[:, 0].detach().cpu().reshape(-1))
            y_pred_1.append(pred[:, 1].detach().cpu())
            y_pred_10_1.append(pred[:, 0].detach().cpu())
            y_pred_90_1.append(pred[:, 2].detach().cpu())
            y_true_2.append(batch_atom_bond.y[:, 1].detach().cpu().reshape(-1))
            y_pred_2.append(pred[:, 4].detach().cpu())
            y_pred_10_2.append(pred[:, 3].detach().cpu())
            y_pred_90_2.append(pred[:, 5].detach().cpu())
    y_true_1 = torch.cat(y_true_1, dim=0)
    y_pred_1 = torch.cat(y_pred_1, dim=0)
    y_pred_10_1 = torch.cat(y_pred_10_1, dim=0)
    y_pred_90_1 = torch.cat(y_pred_90_1, dim=0)
    y_true_2 = torch.cat(y_true_2, dim=0)
    y_pred_2 = torch.cat(y_pred_2, dim=0)
    y_pred_10_2 = torch.cat(y_pred_10_2, dim=0)
    y_pred_90_2 = torch.cat(y_pred_90_2, dim=0)

    R_square_1 = 1 - (((y_true_1 - y_pred_1) ** 2).sum() / ((y_true_1 - y_true_1.mean()) ** 2).sum())
    test_mae_1=torch.mean((y_true_1 - y_pred_1) ** 2)
    R_square_2 = 1 - (((y_true_2 - y_pred_2) ** 2).sum() / ((y_true_2 - y_true_2.mean()) ** 2).sum())
    test_mae_2=torch.mean((y_true_2 - y_pred_2) ** 2)
    return y_true_1,y_pred_1,y_true_2,y_pred_2,R_square_1,test_mae_1,R_square_2,test_mae_2

def cal_prob(prediction):
    '''
    calculate the separation probability Sp
    '''
    #input  prediction=[pred_1,pred_2]
    #output: Sp
    a=prediction[0][0]
    b=prediction[1][0]
    if a[2]<b[0]:
        return 1
    elif a[0]>b[2]:
        return 1
    else:
        length=min(a[2],b[2])-max(a[0],b[0])
        all=max(a[2],b[2])-min(a[0],b[0])
        return 1-length/(all)

def Construct_dataset(dataset,data_index, T1_s,T2_s,speed, eluent,e,m,V_e):
    graph_atom_bond = []
    graph_bond_angle = []
    big_index = []
    all_descriptor = np.load('dataset_save/dataset_morder_1124.npy')[:,[153, 278, 884, 885, 1273, 1594, 431, 1768,1769, 1288, 1521]]
    all_attribute= np.load('dataset_save/dataset_attribute_1124.npy')
    all_descriptor=np.hstack((all_attribute,all_descriptor))
    np.save('dataset_save/X_max_eluent.npy', np.max(eluent, axis=0))
    np.save('dataset_save/X_min_eluent.npy', np.min(eluent, axis=0))
    np.save('dataset_save/X_max_descriptor.npy', np.max(all_descriptor, axis=0))
    np.save('dataset_save/X_min_descriptor.npy', np.min(all_descriptor, axis=0))
    all_descriptor = (all_descriptor - np.min(all_descriptor, axis=0)) / (np.max(all_descriptor, axis=0) - np.min(all_descriptor, axis=0) + 1e-8)
    eluent = (eluent - np.min(eluent, axis=0)) / (np.max(eluent, axis=0) - np.min(eluent, axis=0) + 1e-8)
    all_descriptor= torch.from_numpy(np.array(all_descriptor)).to(torch.int64)
    for i in range(len(dataset)):
        data = dataset[i]
        atom_feature = []
        bond_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([T1_s[i]*speed[i],T2_s[i]*speed[i]]).reshape(1,2)
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int=torch.from_numpy(np.array(data_index[i])).to(torch.int64)
        if y[0,0] > 60:
            continue
        if y[0,1]> 120:
            continue
        prop = torch.ones([bond_feature.shape[0], eluent.shape[1]])*eluent[i]
        e_x = torch.ones([bond_feature.shape[0]]) * e[i]
        m_x = torch.ones([bond_feature.shape[0]]) * m[i]
        V_e_x = torch.ones([bond_feature.shape[0]]) * V_e[i]


        bond_angle_descriptor=torch.ones([bond_angle_feature.shape[0],all_descriptor.shape[1]])*all_descriptor[i]
        if Use_geometry_enhanced == True:
            bond_feature=torch.cat([bond_feature,bond_float_feature.reshape(-1,1)],dim=1)
        bond_feature = torch.cat([bond_feature, prop], dim=1)
        bond_feature = torch.cat([bond_feature, e_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, m_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, V_e_x.reshape(-1, 1)], dim=1)
        if Use_column_info==True:
            diameter=torch.ones([bond_feature.shape[0]]) * 1.5
            column_length=torch.ones([bond_feature.shape[0]]) * 6.6
            density=torch.ones([bond_feature.shape[0]]) * 0.4458
            bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, column_length.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, density.reshape(-1, 1)], dim=1)

        bond_angle_feature=bond_angle_feature.reshape(-1,1)
        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), bond_angle_descriptor], dim=1)


        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y,data_index=data_index_int)
        data_bond_angle= Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)
    return graph_atom_bond,graph_bond_angle


def Construct_dataset_8g(dataset,data_index, T1_s,T2_s,speed, eluent,e,m,V_e):
    graph_atom_bond = []
    graph_bond_angle = []
    big_index = []
    all_descriptor = np.load('dataset_save/dataset_morder_8g.npy')[:,[153, 278, 884, 885, 1273, 1594, 431, 1768,1769, 1288, 1521]]
    all_attribute= np.load('dataset_save/dataset_attribute_8g.npy')
    all_descriptor=np.hstack((all_attribute,all_descriptor))
    # np.save('dataset_save/X_max_eluent.npy', np.max(eluent, axis=0))
    # np.save('dataset_save/X_min_eluent.npy', np.min(eluent, axis=0))
    # np.save('dataset_save/X_max_descriptor.npy', np.max(all_descriptor, axis=0))
    # np.save('dataset_save/X_min_descriptor.npy', np.min(all_descriptor, axis=0))
    all_descriptor = (all_descriptor - np.min(all_descriptor, axis=0)) / (np.max(all_descriptor, axis=0) - np.min(all_descriptor, axis=0) + 1e-8)
    eluent = (eluent - np.min(eluent, axis=0)) / (np.max(eluent, axis=0) - np.min(eluent, axis=0) + 1e-8)
    all_descriptor= torch.from_numpy(np.array(all_descriptor)).to(torch.int64)
    for i in range(len(dataset)):
        data = dataset[i]
        atom_feature = []
        bond_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([T1_s[i]*speed[i],T2_s[i]*speed[i]]).reshape(1,2)
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int=torch.from_numpy(np.array(data_index[i])).to(torch.int64)
        if y[0,0] > 60:
            continue
        if y[0,1]> 120:
            continue
        prop=torch.ones([bond_feature.shape[0],eluent.shape[1]])*eluent[i]
        e_x = torch.ones([bond_feature.shape[0]]) * e[i]
        m_x = torch.ones([bond_feature.shape[0]]) * m[i]
        V_e_x = torch.ones([bond_feature.shape[0]]) * V_e[i]


        bond_angle_descriptor=torch.ones([bond_angle_feature.shape[0],all_descriptor.shape[1]])*all_descriptor[i]
        if Use_geometry_enhanced == True:
            bond_feature=torch.cat([bond_feature,bond_float_feature.reshape(-1,1)],dim=1)
        bond_feature = torch.cat([bond_feature, prop], dim=1)
        bond_feature = torch.cat([bond_feature, e_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, m_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, V_e_x.reshape(-1, 1)], dim=1)
        if Use_column_info==True:
            diameter = torch.ones([bond_feature.shape[0]]) * 1.5
            column_length = torch.ones([bond_feature.shape[0]]) * 13.2
            density = torch.ones([bond_feature.shape[0]]) * 0.4458
            bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, column_length.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, density.reshape(-1, 1)], dim=1)

        bond_angle_feature=bond_angle_feature.reshape(-1,1)
        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), bond_angle_descriptor], dim=1)


        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y,data_index=data_index_int)
        data_bond_angle= Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)
    return graph_atom_bond,graph_bond_angle

def Construct_dataset_25g(dataset,data_index, T1_s,T2_s,speed, eluent,e,m,V_e):
    graph_atom_bond = []
    graph_bond_angle = []
    big_index = []
    all_descriptor = np.load('dataset_save/dataset_morder_25g.npy')[:,[153, 278, 884, 885, 1273, 1594, 431, 1768,1769, 1288, 1521]]
    all_attribute= np.load('dataset_save/dataset_attribute_25g.npy')
    all_descriptor=np.hstack((all_attribute,all_descriptor))
    # np.save('dataset_save/X_max_eluent.npy', np.max(eluent, axis=0))
    # np.save('dataset_save/X_min_eluent.npy', np.min(eluent, axis=0))
    # np.save('dataset_save/X_max_descriptor.npy', np.max(all_descriptor, axis=0))
    # np.save('dataset_save/X_min_descriptor.npy', np.min(all_descriptor, axis=0))
    all_descriptor = (all_descriptor - np.min(all_descriptor, axis=0)) / (np.max(all_descriptor, axis=0) - np.min(all_descriptor, axis=0) + 1e-8)
    eluent = (eluent - np.min(eluent, axis=0)) / (np.max(eluent, axis=0) - np.min(eluent, axis=0) + 1e-8)
    all_descriptor= torch.from_numpy(np.array(all_descriptor)).to(torch.int64)
    for i in range(len(dataset)):
        data = dataset[i]
        atom_feature = []
        bond_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([T1_s[i]*speed[i],T2_s[i]*speed[i]]).reshape(1,2)
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int=torch.from_numpy(np.array(data_index[i])).to(torch.int64)
        if y[0,0] > 60:
            continue
        if y[0,1]> 120:
            continue
        prop=torch.ones([bond_feature.shape[0],eluent.shape[1]])*eluent[i]
        e_x = torch.ones([bond_feature.shape[0]]) * e[i]
        m_x = torch.ones([bond_feature.shape[0]]) * m[i]
        V_e_x = torch.ones([bond_feature.shape[0]]) * V_e[i]


        bond_angle_descriptor=torch.ones([bond_angle_feature.shape[0],all_descriptor.shape[1]])*all_descriptor[i]
        if Use_geometry_enhanced == True:
            bond_feature=torch.cat([bond_feature,bond_float_feature.reshape(-1,1)],dim=1)
        bond_feature = torch.cat([bond_feature, prop], dim=1)
        bond_feature = torch.cat([bond_feature, e_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, m_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, V_e_x.reshape(-1, 1)], dim=1)
        if Use_column_info==True:
            diameter = torch.ones([bond_feature.shape[0]]) * 2.15
            column_length = torch.ones([bond_feature.shape[0]]) * 15.6
            density = torch.ones([bond_feature.shape[0]]) * 0.5248
            bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, column_length.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, density.reshape(-1, 1)], dim=1)

        bond_angle_feature=bond_angle_feature.reshape(-1,1)
        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), bond_angle_descriptor], dim=1)


        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y,data_index=data_index_int)
        data_bond_angle= Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)
    return graph_atom_bond,graph_bond_angle

def Construct_dataset_40g(dataset,data_index, T1_s,T2_s,speed, eluent,e,m,V_e):
    graph_atom_bond = []
    graph_bond_angle = []
    big_index = []
    all_descriptor = np.load('dataset_save/dataset_morder_40g.npy')[:,[153, 278, 884, 885, 1273, 1594, 431, 1768,1769, 1288, 1521]]
    all_attribute= np.load('dataset_save/dataset_attribute_40g.npy')
    all_descriptor=np.hstack((all_attribute,all_descriptor))
    # np.save('dataset_save/X_max_eluent.npy', np.max(eluent, axis=0))
    # np.save('dataset_save/X_min_eluent.npy', np.min(eluent, axis=0))
    # np.save('dataset_save/X_max_descriptor.npy', np.max(all_descriptor, axis=0))
    # np.save('dataset_save/X_min_descriptor.npy', np.min(all_descriptor, axis=0))
    all_descriptor = (all_descriptor - np.min(all_descriptor, axis=0)) / (np.max(all_descriptor, axis=0) - np.min(all_descriptor, axis=0) + 1e-8)
    eluent = (eluent - np.min(eluent, axis=0)) / (np.max(eluent, axis=0) - np.min(eluent, axis=0) + 1e-8)
    all_descriptor= torch.from_numpy(np.array(all_descriptor)).to(torch.int64)
    for i in range(len(dataset)):
        data = dataset[i]
        atom_feature = []
        bond_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        y = torch.Tensor([T1_s[i]*speed[i],T2_s[i]*speed[i]]).reshape(1,2)
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        data_index_int=torch.from_numpy(np.array(data_index[i])).to(torch.int64)
        if y[0,0] > 150:
            continue
        if y[0,1]> 200:
            continue
        prop=torch.ones([bond_feature.shape[0],eluent.shape[1]])*eluent[i]
        e_x = torch.ones([bond_feature.shape[0]]) * e[i]
        m_x = torch.ones([bond_feature.shape[0]]) * m[i]
        V_e_x = torch.ones([bond_feature.shape[0]]) * V_e[i]


        bond_angle_descriptor=torch.ones([bond_angle_feature.shape[0],all_descriptor.shape[1]])*all_descriptor[i]
        if Use_geometry_enhanced == True:
            bond_feature=torch.cat([bond_feature,bond_float_feature.reshape(-1,1)],dim=1)
        bond_feature = torch.cat([bond_feature, prop], dim=1)
        bond_feature = torch.cat([bond_feature, e_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, m_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, V_e_x.reshape(-1, 1)], dim=1)
        if Use_column_info==True:
            diameter = torch.ones([bond_feature.shape[0]]) * 2.15
            column_length = torch.ones([bond_feature.shape[0]]) * 15.6
            density = torch.ones([bond_feature.shape[0]]) * 0.5248
            bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, column_length.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, density.reshape(-1, 1)], dim=1)

        bond_angle_feature=bond_angle_feature.reshape(-1,1)
        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), bond_angle_descriptor], dim=1)


        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y,data_index=data_index_int)
        data_bond_angle= Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)
    return graph_atom_bond,graph_bond_angle

def train(model, device, loader_atom_bond, loader_bond_angle, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(zip(loader_atom_bond,loader_bond_angle)):
        batch_atom_bond=batch[0]
        batch_bond_angle=batch[1]
        batch_atom_bond = batch_atom_bond.to(device)
        batch_bond_angle=batch_bond_angle.to(device)
        pred = model(batch_atom_bond,batch_bond_angle)[0]#.view(-1, )
        true_1=batch_atom_bond.y[:,0]
        true_2=batch_atom_bond.y[:,1]
        optimizer.zero_grad()
        loss_1=q_loss(0.1,true_1,pred[:,0])+torch.mean((true_1-pred[:,1])**2)+q_loss(0.9,true_1,pred[:,2])\
             +torch.mean(torch.relu(pred[:,0]-pred[:,1]))+torch.mean(torch.relu(pred[:,1]-pred[:,2]))
        loss_2=q_loss(0.1,true_2,pred[:,3])+torch.mean((true_2-pred[:,4])**2)+q_loss(0.9,true_2,pred[:,5])\
             +torch.mean(torch.relu(pred[:,3]-pred[:,4]))+torch.mean(torch.relu(pred[:,4]-pred[:,5]))
        loss = loss_1+loss_2*0.5
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def QGeoGNN(data,MODEL):
    # bad_mol = save_3D_mol(data['smiles'], '3D_mol_1124')
    # np.save('3D_mol_1124/bad_mol.npy', np.array(bad_mol))
    # bad_mol=np.load('3D_mol_1124/bad_mol.npy')
    # save_dataset(data['smiles'], '3D_mol_1124', '1124', bad_mol)
    args = parse_args()
    prepartion(args)
    nn_params = {
        'num_tasks': 6,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    dataset = np.load('dataset_save/dataset_1124.npy', allow_pickle=True).tolist()
    data_index=np.arange(0,len(dataset),1)
    dataset_graph_atom_bond, dataset_graph_bond_angle= Construct_dataset(dataset,data_index,data['t1'],data['t2'],data['speed'],
                                                                         data['eluent'],data['e'],data['m'],data['V_e'])
    total_num = len(dataset_graph_atom_bond)
    print(total_num)

    train_ratio = 0.8
    validate_ratio = 0.1
    test_ratio = 0.1

    random.seed(525)
    np.random.seed(1101)
    if split_mode=='data':
    # automatic dataloading and splitting
        data_array = np.arange(0, total_num, 1)
        np.random.shuffle(data_array)
        torch.random.manual_seed(525)


        train_num = int(len(data_array) * train_ratio)
        test_num = int(len(data_array) * test_ratio)
        val_num = int(len(data_array) * validate_ratio)

        train_index = data_array[0:train_num]
        valid_index = data_array[train_num:train_num + val_num]
        if test_mode == 'fixed':
            test_index = data_array[total_num - test_num:]
        if test_mode == 'random':
            test_index = data_array[train_num + val_num:train_num + val_num + test_num]
    if split_mode == 'compound':
        efficient_index=np.load('dataset_save/compound_index.npy')
        compound_index = np.unique(efficient_index)
        all_index=np.arange(0,len(efficient_index),1)
        state = np.random.get_state()
        np.random.shuffle(compound_index)
        train_num = int(train_ratio * compound_index.shape[0])
        val_num = int(validate_ratio * compound_index.shape[0])
        test_num = int(test_ratio* compound_index.shape[0])
        compound_index = compound_index.tolist()
        compound_train = compound_index[0:train_num]
        compound_valid = compound_index[train_num:train_num + val_num]
        compound_test = compound_index[train_num + val_num:train_num + val_num + test_num]
        train_index=all_index[np.isin(efficient_index,compound_train)]
        valid_index = all_index[np.isin(efficient_index, compound_valid)]
        test_index = all_index[np.isin(efficient_index, compound_test)]
        print(test_index.shape)

    train_data_atom_bond = []
    valid_data_atom_bond = []
    test_data_atom_bond = []
    train_data_bond_angle = []
    valid_data_bond_angle = []
    test_data_bond_angle = []
    for i in test_index:
        test_data_atom_bond.append(dataset_graph_atom_bond[i])
        test_data_bond_angle.append(dataset_graph_bond_angle[i])
    for i in valid_index:
        valid_data_atom_bond.append(dataset_graph_atom_bond[i])
        valid_data_bond_angle.append(dataset_graph_bond_angle[i])
    for i in train_index:
        train_data_atom_bond.append(dataset_graph_atom_bond[i])
        train_data_bond_angle.append(dataset_graph_bond_angle[i])

    print(test_data_atom_bond[0].y, test_data_atom_bond[0].data_index)


    train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

    device = args.device
    criterion_fn = torch.nn.MSELoss()
    model = GINGraphPooling(**nn_params).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=args.save_dir)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    print('===========Data Prepared================')

    if MODEL == 'Train':
        if Use_geometry_enhanced==True:
            try:
                os.makedirs(f'saves/model_GeoGNN_1124')
            except OSError:
                pass
        if Use_geometry_enhanced == False:
            try:
                os.makedirs(f'saves/model_GNN_1124')
            except OSError:
                pass

        for epoch in tqdm(range(1000)):

            train_mae = train(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, criterion_fn)

            if (epoch + 1) % 50 == 0:
                valid_mae_1,valid_mae_2 = eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                y_true_1,y_pred_1,y_true_2,y_pred_2,R_square_1, test_mae_1, R_square_2,test_mae_2 = test(model, device, test_loader_atom_bond,
                                                                                test_loader_bond_angle)
                if Use_geometry_enhanced == True:
                    with open(f"saves/model_GeoGNN_1124/GeoGNN.log", "a+") as f:
                        f.write(
                            f'epoch: {epoch+1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                            f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')
                if Use_geometry_enhanced == False:
                    with open(f"saves/model_GNN_1124/GNN.log", "a+") as f:
                        f.write(
                            f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                            f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')

                print(train_mae, valid_mae_1,valid_mae_2, R_square_1, test_mae_1, R_square_2,test_mae_2)
                if Use_geometry_enhanced==True:
                    torch.save(model.state_dict(), f'saves/model_GeoGNN_1124/model_save_{epoch + 1}.pth')
                if Use_geometry_enhanced==False:
                    torch.save(model.state_dict(), f'saves/model_GNN_1124/model_save_{epoch + 1}.pth')

    if MODEL == 'Test':
        if split_mode=='data':
            if Use_geometry_enhanced==False:
                model.load_state_dict(
                    torch.load(f'saves/model_GNN_1031/model_save_950.pth'))
            if Use_geometry_enhanced == True:
                model.load_state_dict(
                    torch.load(f'saves/model_GeoGNN_1124/model_save_300.pth'))
        elif split_mode=='compound':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_compound_1031/model_save_950.pth'))
        y_true_1,y_pred_1,y_true_2,y_pred_2,R_square_1,test_mae_1,R_square_2,test_mae_2= test(model, device, test_loader_atom_bond,
                                                                        test_loader_bond_angle)

        y_pred_t1 = y_pred_1.cpu().data.numpy()
        y_true_t1 = y_true_1.cpu().data.numpy()
        y_pred_t2 = y_pred_2.cpu().data.numpy()
        y_true_t2 = y_true_2.cpu().data.numpy()
        if split_mode == 'data':
            if Use_geometry_enhanced == False:
                measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GNN',c='#CD5C5C')
                measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GNN',c='#6495ED')
                df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                                   'true_t2': y_true_t2.reshape(-1, ),
                                   'pred_t1': y_pred_t1.reshape(-1, ),
                                   'pred_t2': y_pred_t2.reshape(-1, )})
                df.to_csv(f'result_save/GNN.csv')
            else:
                measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN',c='#CD5C5C')
                measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN',c='#6495ED')
                df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                                   'true_t2': y_true_t2.reshape(-1, ),
                                   'pred_t1': y_pred_t1.reshape(-1, ),
                                   'pred_t2': y_pred_t2.reshape(-1, )})
                df.to_csv(f'result_save/GeoGNN.csv')
                with open(f"result_save/GeoGNN.log", "w") as f:
                    f.write(
                        f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                    f.write(
                        f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        elif split_mode=='compound':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_compound',c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_compound',c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_compound.csv')
            with open(f"result_save/GeoGNN_compound.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
            with open(f"result_save/GeoGNN_compound.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        return y_true_1,y_pred_1,y_true_2,y_pred_2

def QGeoGNN_cycle(data,MODEL):
    # bad_mol = save_3D_mol(data['smiles'], '3D_mol')
    # np.save('3D_mol/bad_mol.npy', np.array(bad_mol))
    # bad_mol=np.load('3D_mol/bad_mol.npy')
    # save_dataset(data['smiles'], '3D_mol', '1013', bad_mol)
    args = parse_args()
    prepartion(args)
    nn_params = {
        'num_tasks': 6,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    dataset = np.load('dataset_save/dataset_1124.npy', allow_pickle=True).tolist()
    data_index=np.arange(0,len(dataset),1)
    dataset_graph_atom_bond, dataset_graph_bond_angle= Construct_dataset(dataset,data_index,data['t1'],data['t2'],data['speed'],
                                                                         data['eluent'],data['e'],data['m'],data['V_e'])
    total_num = len(dataset_graph_atom_bond)
    print(total_num)

    train_ratio = 0.9
    validate_ratio = 0.05
    test_ratio = 0.05

    for cross_iter in range(20):
        if split_mode == 'compound':
            efficient_index=np.load('dataset_save/compound_index.npy')
            compound_index = np.unique(efficient_index)
            print(len(compound_index))
            all_index=np.arange(0,len(efficient_index),1)
            train_num = int(train_ratio * compound_index.shape[0])
            val_num = int(validate_ratio * compound_index.shape[0])
            test_num = int(test_ratio* compound_index.shape[0])
            print(f'=============Cross_iter {cross_iter} start!====================')
            compound_test = all_index[cross_iter * test_num:(cross_iter + 1) * test_num]
            delete_cross = np.arange(cross_iter * test_num, (cross_iter + 1) * test_num, 1)
            res_data_array = np.delete(all_index, delete_cross)
            compound_train = res_data_array[0:train_num]
            compound_valid = res_data_array[train_num:train_num + val_num]


            compound_index = compound_index.tolist()
            train_index=all_index[np.isin(efficient_index,compound_train)]
            valid_index = all_index[np.isin(efficient_index, compound_valid)]
            test_index = all_index[np.isin(efficient_index, compound_test)]
            print(test_index.shape)

        train_data_atom_bond = []
        valid_data_atom_bond = []
        test_data_atom_bond = []
        train_data_bond_angle = []
        valid_data_bond_angle = []
        test_data_bond_angle = []
        for i in test_index:
            test_data_atom_bond.append(dataset_graph_atom_bond[i])
            test_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in valid_index:
            valid_data_atom_bond.append(dataset_graph_atom_bond[i])
            valid_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in train_index:
            train_data_atom_bond.append(dataset_graph_atom_bond[i])
            train_data_bond_angle.append(dataset_graph_bond_angle[i])

        print(test_data_atom_bond[0].y, test_data_atom_bond[0].data_index)
        index_tuple={'train_index':train_index,'valid_index':valid_index,'test_index':test_index}
        np.save(f'result_save/cross_iter/index_{cross_iter}_1124.npy',index_tuple)

        train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
        valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
        test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers)
        train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)

        device = args.device
        criterion_fn = torch.nn.MSELoss()
        model = GINGraphPooling(**nn_params).to(device)
        num_params = sum(p.numel() for p in model.parameters())

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
        writer = SummaryWriter(log_dir=args.save_dir)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        print('===========Data Prepared================')

        if MODEL == 'Train':
            try:
                os.makedirs(f'saves/model_{cross_iter}_1124')
            except OSError:
                pass

            for epoch in tqdm(range(1000)):

                train_mae = train(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, criterion_fn)

                if (epoch + 1) % 50 == 0:
                    valid_mae_1,valid_mae_2 = eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                    y_true_1,y_pred_1,y_true_2,y_pred_2,R_square_1, test_mae_1, R_square_2,test_mae_2 = test(model, device, test_loader_atom_bond,
                                                                                    test_loader_bond_angle)

                    with open(f"saves/model_{cross_iter}_1124/GeoGNN.log", "a+") as f:
                            f.write(
                                f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                                f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')

                    print(train_mae, valid_mae_1,valid_mae_2, R_square_1, test_mae_1, R_square_2,test_mae_2)
                    torch.save(model.state_dict(), f'saves/model_{cross_iter}_1124/model_save_{epoch + 1}.pth')

        if MODEL == 'Test':
            best_epoch=[50,600,350,750,50,50,400,600,100,50,100,500,100,50,50,750,800,650,150,50]
            model.load_state_dict(
                torch.load(f'saves/model_{cross_iter}/model_save_{best_epoch[cross_iter]}.pth'))
            y_true_1,y_pred_1,y_true_2,y_pred_2,R_square_1,test_mae_1,R_square_2,test_mae_2= test(model, device, test_loader_atom_bond,
                                                                            test_loader_bond_angle)

            y_pred_t1 = y_pred_1.cpu().data.numpy()
            y_true_t1 = y_true_1.cpu().data.numpy()
            y_pred_t2 = y_pred_2.cpu().data.numpy()
            y_true_t2 = y_true_2.cpu().data.numpy()
            if split_mode == 'data':
                if Use_geometry_enhanced == False:
                    measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GNN')
                    measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GNN')
                    df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                                       'true_t2': y_true_t2.reshape(-1, ),
                                       'pred_t1': y_pred_t1.reshape(-1, ),
                                       'pred_t2': y_pred_t2.reshape(-1, )})
                    df.to_csv(f'result_save/cross_iter/GNN_{cross_iter}.csv')
                else:
                    measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN')
                    measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN')
                    df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                                       'true_t2': y_true_t2.reshape(-1, ),
                                       'pred_t1': y_pred_t1.reshape(-1, ),
                                       'pred_t2': y_pred_t2.reshape(-1, )})
                    df.to_csv(f'result_save/cross_iter/GeoGNN_{cross_iter}.csv')
            elif split_mode=='compound':
                measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_compound_{cross_iter}')
                measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_compound_{cross_iter}')
                df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                                   'true_t2': y_true_t2.reshape(-1, ),
                                   'pred_t1': y_pred_t1.reshape(-1, ),
                                   'pred_t2': y_pred_t2.reshape(-1, ),
                                   'test_index':test_index})
                df.to_csv(f'result_save/cross_iter/GeoGNN_compound_{cross_iter}.csv')
            #return y_true_1,y_pred_1,y_true_2,y_pred_2

def QGeoGNN_transfer_8g(data,MODEL):
    split_mode = 'data'
    transfer_mode='no_transfer'
    # bad_mol = save_3D_mol(data['smiles'], '3D_mol_8g')
    # np.save('3D_mol_8g/bad_mol.npy', np.array(bad_mol))
    # bad_mol=np.load('3D_mol_8g/bad_mol.npy')
    # save_dataset(data['smiles'], '3D_mol_8g', '8g', bad_mol)
    args = parse_args()
    prepartion(args)
    nn_params = {
        'num_tasks': 6,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    dataset = np.load('dataset_save/dataset_8g.npy', allow_pickle=True).tolist()
    data_index = np.arange(0, len(dataset), 1)
    dataset_graph_atom_bond, dataset_graph_bond_angle = Construct_dataset_8g(dataset, data_index, data['t1'], data['t2'],
                                                                          data['speed'],
                                                                          data['eluent'], data['e'], data['m'],
                                                                          data['V_e'])
    total_num = len(dataset_graph_atom_bond)
    print(total_num)

    train_ratio = 0.8
    validate_ratio = 0.1
    test_ratio = 0.1

    random.seed(525)
    np.random.seed(1101)
    if split_mode == 'data':
        # automatic dataloading and splitting
        data_array = np.arange(0, total_num, 1)
        np.random.shuffle(data_array)
        torch.random.manual_seed(525)

        train_num = int(len(data_array) * train_ratio)
        test_num = int(len(data_array) * test_ratio)
        val_num = int(len(data_array) * validate_ratio)

        train_index = data_array[0:train_num]
        valid_index = data_array[train_num:train_num + val_num]
        if test_mode == 'fixed':
            test_index = data_array[total_num - test_num:]
        if test_mode == 'random':
            test_index = data_array[train_num + val_num:train_num + val_num + test_num]
    if split_mode == 'compound':
        efficient_index = np.load('dataset_save/compound_index.npy')
        compound_index = np.unique(efficient_index)
        all_index = np.arange(0, len(efficient_index), 1)
        state = np.random.get_state()
        np.random.shuffle(compound_index)
        train_num = int(train_ratio * compound_index.shape[0])
        val_num = int(validate_ratio * compound_index.shape[0])
        test_num = int(test_ratio * compound_index.shape[0])
        compound_index = compound_index.tolist()
        compound_train = compound_index[0:train_num]
        compound_valid = compound_index[train_num:train_num + val_num]
        compound_test = compound_index[train_num + val_num:train_num + val_num + test_num]
        train_index = all_index[np.isin(efficient_index, compound_train)]
        valid_index = all_index[np.isin(efficient_index, compound_valid)]
        test_index = all_index[np.isin(efficient_index, compound_test)]
        print(test_index.shape)

    train_data_atom_bond = []
    valid_data_atom_bond = []
    test_data_atom_bond = []
    train_data_bond_angle = []
    valid_data_bond_angle = []
    test_data_bond_angle = []
    for i in test_index:
        test_data_atom_bond.append(dataset_graph_atom_bond[i])
        test_data_bond_angle.append(dataset_graph_bond_angle[i])
    for i in valid_index:
        valid_data_atom_bond.append(dataset_graph_atom_bond[i])
        valid_data_bond_angle.append(dataset_graph_bond_angle[i])
    for i in train_index:
        train_data_atom_bond.append(dataset_graph_atom_bond[i])
        train_data_bond_angle.append(dataset_graph_bond_angle[i])

    print(test_data_atom_bond[0].y, test_data_atom_bond[0].data_index)

    train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

    device = args.device
    criterion_fn = torch.nn.MSELoss()
    model = GINGraphPooling(**nn_params).to(device)
    if transfer_mode=='transfer':
        model.load_state_dict(
            torch.load(f'saves/model_GeoGNN_1031/model_save_400.pth'))
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=args.save_dir)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    print('===========Data Prepared================')

    if MODEL == 'Train':
        if transfer_mode=='direct_train':
            try:
                os.makedirs(f'saves/model_GeoGNN_direct_train_8g')
            except OSError:
                pass
        elif transfer_mode=='transfer':
            try:
                os.makedirs(f'saves/model_GeoGNN_transfer_8g')
            except OSError:
                pass

        for epoch in tqdm(range(1000)):

            train_mae = train(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, criterion_fn)

            if (epoch + 1) % 50 == 0:
                valid_mae_1, valid_mae_2 = eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model,
                                                                                                              device,
                                                                                                              test_loader_atom_bond,
                                                                                                              test_loader_bond_angle)
                if transfer_mode=='transfer':
                    with open(f"saves/model_GeoGNN_transfer_8g/GeoGNN.log", "a+") as f:
                        f.write(
                            f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                            f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')
                if transfer_mode=='direct_train':
                    with open(f"saves/model_GeoGNN_direct_train_8g/GeoGNN.log", "a+") as f:
                        f.write(
                            f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                            f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')
                print(train_mae, valid_mae_1, valid_mae_2, R_square_1, test_mae_1, R_square_2, test_mae_2)
                if transfer_mode=='transfer':
                    torch.save(model.state_dict(), f'saves/model_GeoGNN_transfer_8g/model_save_{epoch + 1}.pth')
                if transfer_mode == 'direct_train':
                    torch.save(model.state_dict(), f'saves/model_GeoGNN_direct_train_8g/model_save_{epoch + 1}.pth')

    if MODEL == 'Test':
        if transfer_mode == 'direct_train':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_direct_train_8g/model_save_950.pth'))
        if transfer_mode == 'transfer':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_transfer_8g/model_save_50.pth'))
        if transfer_mode == 'no_transfer':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_1031/model_save_400.pth'))
        y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model, device,
                                                                                                      test_loader_atom_bond,
                                                                                                      test_loader_bond_angle)

        y_pred_t1 = y_pred_1.cpu().data.numpy()
        y_true_t1 = y_true_1.cpu().data.numpy()
        y_pred_t2 = y_pred_2.cpu().data.numpy()
        y_true_t2 = y_true_2.cpu().data.numpy()
        if transfer_mode=='direct_train':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_direct_train_8g',c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_direct_train_8g',c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_direct_train_8g.csv')
            with open(f"result_save/GeoGNN_direct_train_8g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        if transfer_mode=='transfer':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_transfer_8g',c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_transfer_8g',c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_transfer_8g.csv')
            with open(f"result_save/GeoGNN_transfer_8g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        if transfer_mode == 'no_transfer':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_no_transfer_8g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_no_transfer_8g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_no_transfer_8g.csv')
            with open(f"result_save/GeoGNN_no_transfer_8g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        return y_true_1, y_pred_1, y_true_2, y_pred_2
def QGeoGNN_transfer_25g(data,MODEL):
    split_mode = 'data'
    transfer_mode='no_transfer'
    # bad_mol = save_3D_mol(data['smiles'], '3D_mol_25g')
    # np.save('3D_mol_25g/bad_mol.npy', np.array(bad_mol))
    # bad_mol=np.load('3D_mol_25g/bad_mol.npy')
    # save_dataset(data['smiles'], '3D_mol_25g', '25g', bad_mol)
    args = parse_args()
    prepartion(args)
    nn_params = {
        'num_tasks': 6,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    dataset = np.load('dataset_save/dataset_25g.npy', allow_pickle=True).tolist()
    data_index = np.arange(0, len(dataset), 1)
    dataset_graph_atom_bond, dataset_graph_bond_angle = Construct_dataset_25g(dataset, data_index, data['t1'], data['t2'],
                                                                          data['speed'],
                                                                          data['eluent'], data['e'], data['m'],
                                                                          data['V_e'])
    total_num = len(dataset_graph_atom_bond)
    print(total_num)

    train_ratio = 0.8
    validate_ratio = 0.1
    test_ratio = 0.1

    random.seed(525)
    np.random.seed(1101)
    if split_mode == 'data':
        # automatic dataloading and splitting
        data_array = np.arange(0, total_num, 1)
        np.random.shuffle(data_array)
        torch.random.manual_seed(525)

        train_num = int(len(data_array) * train_ratio)
        test_num = int(len(data_array) * test_ratio)
        val_num = int(len(data_array) * validate_ratio)

        train_index = data_array[0:train_num]
        valid_index = data_array[train_num:train_num + val_num]
        if test_mode == 'fixed':
            test_index = data_array[total_num - test_num:]
        if test_mode == 'random':
            test_index = data_array[train_num + val_num:train_num + val_num + test_num]
    if split_mode == 'compound':
        efficient_index = np.load('dataset_save/compound_index.npy')
        compound_index = np.unique(efficient_index)
        all_index = np.arange(0, len(efficient_index), 1)
        state = np.random.get_state()
        np.random.shuffle(compound_index)
        train_num = int(train_ratio * compound_index.shape[0])
        val_num = int(validate_ratio * compound_index.shape[0])
        test_num = int(test_ratio * compound_index.shape[0])
        compound_index = compound_index.tolist()
        compound_train = compound_index[0:train_num]
        compound_valid = compound_index[train_num:train_num + val_num]
        compound_test = compound_index[train_num + val_num:train_num + val_num + test_num]
        train_index = all_index[np.isin(efficient_index, compound_train)]
        valid_index = all_index[np.isin(efficient_index, compound_valid)]
        test_index = all_index[np.isin(efficient_index, compound_test)]
        print(test_index.shape)

    train_data_atom_bond = []
    valid_data_atom_bond = []
    test_data_atom_bond = []
    train_data_bond_angle = []
    valid_data_bond_angle = []
    test_data_bond_angle = []
    for i in test_index:
        test_data_atom_bond.append(dataset_graph_atom_bond[i])
        test_data_bond_angle.append(dataset_graph_bond_angle[i])
    for i in valid_index:
        valid_data_atom_bond.append(dataset_graph_atom_bond[i])
        valid_data_bond_angle.append(dataset_graph_bond_angle[i])
    for i in train_index:
        train_data_atom_bond.append(dataset_graph_atom_bond[i])
        train_data_bond_angle.append(dataset_graph_bond_angle[i])

    print(test_data_atom_bond[0].y, test_data_atom_bond[0].data_index)

    train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

    device = args.device
    criterion_fn = torch.nn.MSELoss()
    model = GINGraphPooling(**nn_params).to(device)
    if transfer_mode=='transfer':
        model.load_state_dict(
            torch.load(f'saves/model_GeoGNN_1031/model_save_400.pth'))

    num_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=args.save_dir)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    print('===========Data Prepared================')

    if MODEL == 'Train':
        if transfer_mode=='direct_train':
            try:
                os.makedirs(f'saves/model_GeoGNN_direct_train_25g')
            except OSError:
                pass
        elif transfer_mode=='transfer':
            try:
                os.makedirs(f'saves/model_GeoGNN_transfer_25g')
            except OSError:
                pass

        for epoch in tqdm(range(200)):

            train_mae = train(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, criterion_fn)

            if (epoch + 1) % 10 == 0:
                valid_mae_1, valid_mae_2 = eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model,
                                                                                                              device,
                                                                                                              test_loader_atom_bond,
                                                                                                              test_loader_bond_angle)
                if transfer_mode=='transfer':
                    with open(f"saves/model_GeoGNN_transfer_25g/GeoGNN.log", "a+") as f:
                        f.write(
                            f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                            f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')

                if transfer_mode=='direct_train':
                    with open(f"saves/model_GeoGNN_direct_train_25g/GeoGNN.log", "a+") as f:
                        f.write(
                            f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                            f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')
                print(train_mae, valid_mae_1, valid_mae_2, R_square_1, test_mae_1, R_square_2, test_mae_2)
                if transfer_mode=='transfer':
                    torch.save(model.state_dict(), f'saves/model_GeoGNN_transfer_25g/model_save_{epoch + 1}.pth')
                if transfer_mode == 'direct_train':
                    torch.save(model.state_dict(), f'saves/model_GeoGNN_direct_train_25g/model_save_{epoch + 1}.pth')

    if MODEL == 'Test':
        if transfer_mode == 'direct_train':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_direct_train_25g/model_save_1000.pth'))
        if transfer_mode == 'transfer':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_transfer_25g/model_save_90.pth'))
        if transfer_mode == 'no_transfer':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_1031/model_save_400.pth'))
        y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model, device,
                                                                                                      test_loader_atom_bond,
                                                                                                      test_loader_bond_angle)

        y_pred_t1 = y_pred_1.cpu().data.numpy()
        y_true_t1 = y_true_1.cpu().data.numpy()
        y_pred_t2 = y_pred_2.cpu().data.numpy()
        y_true_t2 = y_true_2.cpu().data.numpy()
        if transfer_mode == 'direct_train':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_direct_train_25g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_direct_train_25g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_direct_train_25g.csv')
            with open(f"result_save/GeoGNN_direct_train_25g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')

        if transfer_mode == 'transfer':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_transfer_25g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_transfer_25g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_transfer_25g.csv')
            with open(f"result_save/GeoGNN_transfer_25g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        if transfer_mode == 'no_transfer':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_no_transfer_25g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_no_transfer_25g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_no_transfer_25g.csv')
            with open(f"result_save/GeoGNN_no_transfer_25g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')

        return y_true_1, y_pred_1, y_true_2, y_pred_2
def QGeoGNN_transfer_40g(data,MODEL):
    split_mode = 'data'
    transfer_mode='no_transfer'
    # bad_mol = save_3D_mol(data['smiles'], '3D_mol_40g')
    # np.save('3D_mol_40g/bad_mol.npy', np.array(bad_mol))
    # bad_mol=np.load('3D_mol_40g/bad_mol.npy')
    # save_dataset(data['smiles'], '3D_mol_40g', '40g', bad_mol)
    args = parse_args()
    prepartion(args)
    nn_params = {
        'num_tasks': 6,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    dataset = np.load('dataset_save/dataset_40g.npy', allow_pickle=True).tolist()
    data_index = np.arange(0, len(dataset), 1)
    dataset_graph_atom_bond, dataset_graph_bond_angle = Construct_dataset_40g(dataset, data_index, data['t1'], data['t2'],
                                                                          data['speed'],
                                                                          data['eluent'], data['e'], data['m'],
                                                                          data['V_e'])
    total_num = len(dataset_graph_atom_bond)
    print(total_num)

    train_ratio = 0.8
    validate_ratio = 0.1
    test_ratio = 0.1

    random.seed(525)
    np.random.seed(1101)
    if split_mode == 'data':
        # automatic dataloading and splitting
        data_array = np.arange(0, total_num, 1)
        np.random.shuffle(data_array)
        torch.random.manual_seed(525)

        train_num = int(len(data_array) * train_ratio)
        test_num = int(len(data_array) * test_ratio)
        val_num = int(len(data_array) * validate_ratio)

        train_index = data_array[0:train_num]
        valid_index = data_array[train_num:train_num + val_num]
        if test_mode == 'fixed':
            test_index = data_array[total_num - test_num:]
        if test_mode == 'random':
            test_index = data_array[train_num + val_num:train_num + val_num + test_num]
    if split_mode == 'compound':
        efficient_index = np.load('dataset_save/compound_index.npy')
        compound_index = np.unique(efficient_index)
        all_index = np.arange(0, len(efficient_index), 1)
        state = np.random.get_state()
        np.random.shuffle(compound_index)
        train_num = int(train_ratio * compound_index.shape[0])
        val_num = int(validate_ratio * compound_index.shape[0])
        test_num = int(test_ratio * compound_index.shape[0])
        compound_index = compound_index.tolist()
        compound_train = compound_index[0:train_num]
        compound_valid = compound_index[train_num:train_num + val_num]
        compound_test = compound_index[train_num + val_num:train_num + val_num + test_num]
        train_index = all_index[np.isin(efficient_index, compound_train)]
        valid_index = all_index[np.isin(efficient_index, compound_valid)]
        test_index = all_index[np.isin(efficient_index, compound_test)]
        print(test_index.shape)

    train_data_atom_bond = []
    valid_data_atom_bond = []
    test_data_atom_bond = []
    train_data_bond_angle = []
    valid_data_bond_angle = []
    test_data_bond_angle = []
    for i in test_index:
        test_data_atom_bond.append(dataset_graph_atom_bond[i])
        test_data_bond_angle.append(dataset_graph_bond_angle[i])
    for i in valid_index:
        valid_data_atom_bond.append(dataset_graph_atom_bond[i])
        valid_data_bond_angle.append(dataset_graph_bond_angle[i])
    for i in train_index:
        train_data_atom_bond.append(dataset_graph_atom_bond[i])
        train_data_bond_angle.append(dataset_graph_bond_angle[i])

    print(test_data_atom_bond[0].y, test_data_atom_bond[0].data_index)

    train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

    device = args.device
    criterion_fn = torch.nn.MSELoss()
    model = GINGraphPooling(**nn_params).to(device)
    if transfer_mode=='transfer':
        model.load_state_dict(
            torch.load(f'saves/model_GeoGNN_1031/model_save_400.pth'))

    num_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=args.save_dir)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    print('===========Data Prepared================')

    if MODEL == 'Train':
        if transfer_mode=='direct_train':
            try:
                os.makedirs(f'saves/model_GeoGNN_direct_train_40g')
            except OSError:
                pass
        elif transfer_mode=='transfer':
            try:
                os.makedirs(f'saves/model_GeoGNN_transfer_40g')
            except OSError:
                pass

        for epoch in tqdm(range(500)):

            train_mae = train(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, criterion_fn)

            if (epoch + 1) % 10 == 0:
                valid_mae_1, valid_mae_2 = eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model,
                                                                                                              device,
                                                                                                              test_loader_atom_bond,
                                                                                                              test_loader_bond_angle)
                if transfer_mode=='transfer':
                    with open(f"saves/model_GeoGNN_transfer_40g/GeoGNN.log", "a+") as f:
                        f.write(
                            f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                            f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')

                if transfer_mode=='direct_train':
                    with open(f"saves/model_GeoGNN_direct_train_40g/GeoGNN.log", "a+") as f:
                        f.write(
                            f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                            f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')
                print(train_mae, valid_mae_1, valid_mae_2, R_square_1, test_mae_1, R_square_2, test_mae_2)
                if transfer_mode=='transfer':
                    torch.save(model.state_dict(), f'saves/model_GeoGNN_transfer_40g/model_save_{epoch + 1}.pth')
                if transfer_mode == 'direct_train':
                    torch.save(model.state_dict(), f'saves/model_GeoGNN_direct_train_40g/model_save_{epoch + 1}.pth')

    if MODEL == 'Test':
        if transfer_mode == 'direct_train':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_direct_train_40g/model_save_1000.pth'))
        if transfer_mode == 'transfer':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_transfer_40g/model_save_460.pth'))
        if transfer_mode == 'no_transfer':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_1031/model_save_400.pth'))
        y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model, device,
                                                                                                      test_loader_atom_bond,
                                                                                                      test_loader_bond_angle)

        y_pred_t1 = y_pred_1.cpu().data.numpy()
        y_true_t1 = y_true_1.cpu().data.numpy()
        y_pred_t2 = y_pred_2.cpu().data.numpy()
        y_true_t2 = y_true_2.cpu().data.numpy()
        if transfer_mode == 'direct_train':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_direct_train_40g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_direct_train_40g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_direct_train_40g.csv')
            with open(f"result_save/GeoGNN_direct_train_40g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')

        if transfer_mode == 'transfer':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_transfer_40g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_transfer_40g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_transfer_40g.csv')
            with open(f"result_save/GeoGNN_transfer_40g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        if transfer_mode == 'no_transfer':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_no_transfer_40g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_no_transfer_40g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_no_transfer_40g.csv')
            with open(f"result_save/GeoGNN_no_transfer_40g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')

        return y_true_1, y_pred_1, y_true_2, y_pred_2
def QGeoGNN_transfer_column_info(data,data_8,data_25,MODEL):
    Use_column_info = True
    split_mode = 'data'

    args = parse_args()
    prepartion(args)
    nn_params = {
        'num_tasks': 6,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    dataset_25g = np.load('dataset_save/dataset_25g.npy', allow_pickle=True).tolist()
    dataset_8g = np.load('dataset_save/dataset_8g.npy', allow_pickle=True).tolist()
    dataset_4g = np.load('dataset_save/dataset_1013.npy', allow_pickle=True).tolist()
    data_index_25g = np.arange(0, len(dataset_25g), 1)
    data_index_8g = np.arange(0, len(dataset_8g), 1)
    data_index_4g = np.arange(0, len(dataset_4g), 1)
    dataset_graph_atom_bond_25g, dataset_graph_bond_angle_25g = Construct_dataset_25g(dataset_25g, data_index_25g, data_25['t1'], data_25['t2'],
                                                                          data_25['speed'],
                                                                          data_25['eluent'], data_25['e'], data_25['m'],
                                                                          data_25['V_e'])
    dataset_graph_atom_bond_8g, dataset_graph_bond_angle_8g = Construct_dataset_8g(dataset_8g, data_index_8g, data_8['t1'], data_8['t2'],
                                                                          data_8['speed'],
                                                                          data_8['eluent'], data_8['e'], data_8['m'],
                                                                          data_8['V_e'])
    dataset_graph_atom_bond_4g, dataset_graph_bond_angle_4g = Construct_dataset(dataset_4g, data_index_4g, data['t1'], data['t2'],
                                                                          data['speed'],
                                                                          data['eluent'], data['e'], data['m'],
                                                                          data['V_e'])
    def split_data(dataset_graph_atom_bond,dataset_graph_bond_angle):
        total_num = len(dataset_graph_atom_bond)
        print(total_num)

        train_ratio = 0.8
        validate_ratio = 0.1
        test_ratio = 0.1

        random.seed(525)
        np.random.seed(1101)
        if split_mode == 'data':
            # automatic dataloading and splitting
            data_array = np.arange(0, total_num, 1)
            np.random.shuffle(data_array)
            torch.random.manual_seed(525)

            train_num = int(len(data_array) * train_ratio)
            test_num = int(len(data_array) * test_ratio)
            val_num = int(len(data_array) * validate_ratio)

            train_index = data_array[0:train_num]
            valid_index = data_array[train_num:train_num + val_num]
            if test_mode == 'fixed':
                test_index = data_array[total_num - test_num:]
            if test_mode == 'random':
                test_index = data_array[train_num + val_num:train_num + val_num + test_num]
        if split_mode == 'compound':
            efficient_index = np.load('dataset_save/compound_index.npy')
            compound_index = np.unique(efficient_index)
            all_index = np.arange(0, len(efficient_index), 1)
            state = np.random.get_state()
            np.random.shuffle(compound_index)
            train_num = int(train_ratio * compound_index.shape[0])
            val_num = int(validate_ratio * compound_index.shape[0])
            test_num = int(test_ratio * compound_index.shape[0])
            compound_index = compound_index.tolist()
            compound_train = compound_index[0:train_num]
            compound_valid = compound_index[train_num:train_num + val_num]
            compound_test = compound_index[train_num + val_num:train_num + val_num + test_num]
            train_index = all_index[np.isin(efficient_index, compound_train)]
            valid_index = all_index[np.isin(efficient_index, compound_valid)]
            test_index = all_index[np.isin(efficient_index, compound_test)]
            print(test_index.shape)

        train_data_atom_bond = []
        valid_data_atom_bond = []
        test_data_atom_bond = []
        train_data_bond_angle = []
        valid_data_bond_angle = []
        test_data_bond_angle = []
        for i in test_index:
            test_data_atom_bond.append(dataset_graph_atom_bond[i])
            test_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in valid_index:
            valid_data_atom_bond.append(dataset_graph_atom_bond[i])
            valid_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in train_index:
            train_data_atom_bond.append(dataset_graph_atom_bond[i])
            train_data_bond_angle.append(dataset_graph_bond_angle[i])

        print(test_data_atom_bond[0].y, test_data_atom_bond[0].data_index)


        return train_data_atom_bond,valid_data_atom_bond, test_data_atom_bond,train_data_bond_angle,valid_data_bond_angle,test_data_bond_angle

    train_data_atom_bond_4g, valid_data_atom_bond_4g, test_data_atom_bond_4g, train_data_bond_angle_4g, valid_data_bond_angle_4g, test_data_bond_angle_4g=split_data(dataset_graph_atom_bond_4g,dataset_graph_bond_angle_4g)
    train_data_atom_bond_25g, valid_data_atom_bond_25g, test_data_atom_bond_25g, train_data_bond_angle_25g, valid_data_bond_angle_25g, test_data_bond_angle_25g = split_data(
        dataset_graph_atom_bond_25g, dataset_graph_bond_angle_25g)
    train_data_atom_bond_8g, valid_data_atom_bond_8g, test_data_atom_bond_8g, train_data_bond_angle_8g, valid_data_bond_angle_8g, test_data_bond_angle_8g = split_data(
        dataset_graph_atom_bond_8g, dataset_graph_bond_angle_8g)
    train_data_atom_bond=train_data_atom_bond_4g+train_data_atom_bond_8g+train_data_atom_bond_25g
    valid_data_atom_bond=valid_data_atom_bond_4g+valid_data_atom_bond_8g+valid_data_atom_bond_25g
    test_data_atom_bond = test_data_atom_bond_4g + test_data_atom_bond_8g + test_data_atom_bond_25g
    train_data_bond_angle=train_data_bond_angle_4g+train_data_bond_angle_8g+train_data_bond_angle_25g
    valid_data_bond_angle = valid_data_bond_angle_4g + valid_data_bond_angle_8g + valid_data_bond_angle_25g
    test_data_bond_angle = test_data_bond_angle_4g +test_data_bond_angle_8g + test_data_bond_angle_25g
    train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    device = args.device
    criterion_fn = torch.nn.MSELoss()
    model = GINGraphPooling(**nn_params).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=args.save_dir)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    print('===========Data Prepared================')

    if MODEL == 'Train':

        try:
            os.makedirs(f'saves/model_GeoGNN_column_info')
        except OSError:
            pass

        for epoch in tqdm(range(2000)):

            train_mae = train(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, criterion_fn)

            if (epoch + 1) % 50 == 0:
                valid_mae_1, valid_mae_2 = eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model,
                                                                                                              device,
                                                                                                              test_loader_atom_bond,
                                                                                                              test_loader_bond_angle)
                print(train_mae, valid_mae_1, valid_mae_2, R_square_1, test_mae_1, R_square_2, test_mae_2)
                with open(f"saves/model_GeoGNN_column_info/GeoGNN.log", "a+") as f:
                    f.write(
                        f'epoch: {epoch + 1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                        f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')


                torch.save(model.state_dict(), f'saves/model_GeoGNN_column_info/model_save_{epoch + 1}.pth')


    if MODEL == 'Test':
        if transfer_mode == 'direct_train':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_direct_train_25g/model_save_1000.pth'))
        if transfer_mode == 'transfer':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_transfer_25g/model_save_90.pth'))
        if transfer_mode == 'no_transfer':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_1031/model_save_400.pth'))
        y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model, device,
                                                                                                      test_loader_atom_bond,
                                                                                                      test_loader_bond_angle)

        y_pred_t1 = y_pred_1.cpu().data.numpy()
        y_true_t1 = y_true_1.cpu().data.numpy()
        y_pred_t2 = y_pred_2.cpu().data.numpy()
        y_true_t2 = y_true_2.cpu().data.numpy()
        if transfer_mode == 'direct_train':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_direct_train_25g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_direct_train_25g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_direct_train_25g.csv')
            with open(f"result_save/GeoGNN_direct_train_25g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')

        if transfer_mode == 'transfer':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_transfer_25g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_transfer_25g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_transfer_25g.csv')
            with open(f"result_save/GeoGNN_transfer_25g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        if transfer_mode == 'no_transfer':
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN_no_transfer_25g', c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN_no_transfer_25g', c='#6495ED')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/GeoGNN_no_transfer_25g.csv')
            with open(f"result_save/GeoGNN_no_transfer_25g.log", "w") as f:
                f.write(
                    f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                f.write(
                    f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')

        return y_true_1, y_pred_1, y_true_2, y_pred_2

def QGeoGNN_different_data_num(data,MODEL):
    # bad_mol = save_3D_mol(data['smiles'], '3D_mol')
    # np.save('3D_mol/bad_mol.npy', np.array(bad_mol))
    # bad_mol=np.load('3D_mol/bad_mol.npy')
    # save_dataset(data['smiles'], '3D_mol', '1013', bad_mol)
    split_mode = 'data'
    args = parse_args()
    prepartion(args)
    nn_params = {
        'num_tasks': 6,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    dataset = np.load('dataset_save/dataset_1124.npy', allow_pickle=True).tolist()
    data_index=np.arange(0,len(dataset),1)
    dataset_graph_atom_bond, dataset_graph_bond_angle= Construct_dataset(dataset,data_index,data['t1'],data['t2'],data['speed'],
                                                                         data['eluent'],data['e'],data['m'],data['V_e'])
    total_num = len(dataset_graph_atom_bond)
    print(total_num)

    for train_ratio in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        validate_ratio = 0.1
        test_ratio = 0.1

        random.seed(525)
        np.random.seed(1101)
        if split_mode=='data':
        # automatic dataloading and splitting
            data_array = np.arange(0, total_num, 1)
            np.random.shuffle(data_array)
            torch.random.manual_seed(525)


            train_num = int(len(data_array) * train_ratio)
            test_num = int(len(data_array) * test_ratio)
            val_num = int(len(data_array) * validate_ratio)

            train_index = data_array[0:train_num]
            valid_index = data_array[train_num:train_num + val_num]
            if test_mode == 'fixed':
                test_index = data_array[total_num - test_num:]
            if test_mode == 'random':
                test_index = data_array[train_num + val_num:train_num + val_num + test_num]
        if split_mode == 'compound':
            efficient_index=np.load('dataset_save/compound_index.npy')
            compound_index = np.unique(efficient_index)
            all_index=np.arange(0,len(efficient_index),1)
            state = np.random.get_state()
            np.random.shuffle(compound_index)
            train_num = int(train_ratio * compound_index.shape[0])
            val_num = int(validate_ratio * compound_index.shape[0])
            test_num = int(test_ratio* compound_index.shape[0])
            compound_index = compound_index.tolist()
            compound_train = compound_index[0:train_num]
            compound_valid = compound_index[train_num:train_num + val_num]
            compound_test = compound_index[train_num + val_num:train_num + val_num + test_num]
            train_index=all_index[np.isin(efficient_index,compound_train)]
            valid_index = all_index[np.isin(efficient_index, compound_valid)]
            test_index = all_index[np.isin(efficient_index, compound_test)]
            print(test_index.shape)

        train_data_atom_bond = []
        valid_data_atom_bond = []
        test_data_atom_bond = []
        train_data_bond_angle = []
        valid_data_bond_angle = []
        test_data_bond_angle = []
        for i in test_index:
            test_data_atom_bond.append(dataset_graph_atom_bond[i])
            test_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in valid_index:
            valid_data_atom_bond.append(dataset_graph_atom_bond[i])
            valid_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in train_index:
            train_data_atom_bond.append(dataset_graph_atom_bond[i])
            train_data_bond_angle.append(dataset_graph_bond_angle[i])

        print(test_data_atom_bond[0].y, test_data_atom_bond[0].data_index)
        print(len(train_data_atom_bond),len(test_data_atom_bond))

        train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
        valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
        test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers)
        train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)

        device = args.device
        criterion_fn = torch.nn.MSELoss()
        model = GINGraphPooling(**nn_params).to(device)
        num_params = sum(p.numel() for p in model.parameters())

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
        writer = SummaryWriter(log_dir=args.save_dir)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        print('===========Data Prepared================')

        if MODEL == 'Train':
            if Use_geometry_enhanced==True:
                try:
                    os.makedirs(f'saves/model_GeoGNN_different_num_{train_ratio}')
                except OSError:
                    pass


            for epoch in tqdm(range(1000)):

                train_mae = train(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, criterion_fn)

                if (epoch + 1) % 50 == 0:
                    valid_mae_1,valid_mae_2 = eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                    y_true_1,y_pred_1,y_true_2,y_pred_2,R_square_1, test_mae_1, R_square_2,test_mae_2 = test(model, device, test_loader_atom_bond,
                                                                                    test_loader_bond_angle)
                    if Use_geometry_enhanced == True:
                        with open(f"saves/model_GeoGNN_different_num_{train_ratio}/GeoGNN.log", "a+") as f:
                            f.write(
                                f'epoch: {epoch+1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                                f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')


                    print(train_mae, valid_mae_1,valid_mae_2, R_square_1, test_mae_1, R_square_2,test_mae_2)
                    if Use_geometry_enhanced==True:
                        torch.save(model.state_dict(), f'saves/model_GeoGNN_different_num_{train_ratio}/model_save_{epoch + 1}.pth')

        if MODEL == 'Test':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_different_num_{train_ratio}/model_save_950.pth'))
            y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model, device,
                                                                                                          test_loader_atom_bond,
                                                                                                          test_loader_bond_angle)
            print( R_square_1, test_mae_1, R_square_2, test_mae_2)
            y_pred_t1 = y_pred_1.cpu().data.numpy()
            y_true_t1 = y_true_1.cpu().data.numpy()
            y_pred_t2 = y_pred_2.cpu().data.numpy()
            y_true_t2 = y_true_2.cpu().data.numpy()
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/different_num/GeoGNN_different_num_{train_ratio}.csv')

def QGeoGNN_different_noise_level(data,MODEL):
    # bad_mol = save_3D_mol(data['smiles'], '3D_mol')
    # np.save('3D_mol/bad_mol.npy', np.array(bad_mol))
    # bad_mol=np.load('3D_mol/bad_mol.npy')
    # save_dataset(data['smiles'], '3D_mol', '1013', bad_mol)


    for noise_level in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]:
        args = parse_args()
        prepartion(args)
        nn_params = {
            'num_tasks': 6,
            'num_layers': args.num_layers,
            'emb_dim': args.emb_dim,
            'drop_ratio': args.drop_ratio,
            'graph_pooling': args.graph_pooling,
            'descriptor_dim': 1827
        }
        dataset = np.load('dataset_save/dataset_1124.npy', allow_pickle=True).tolist()
        data_index = np.arange(0, len(dataset), 1)
        dataset_graph_atom_bond, dataset_graph_bond_angle = Construct_dataset(dataset, data_index, data['t1'],
                                                                              data['t2'], data['speed'],
                                                                              data['eluent'], data['e'], data['m'],
                                                                              data['V_e'])
        total_num = len(dataset_graph_atom_bond)
        print(total_num)
        train_ratio=0.8
        validate_ratio = 0.1
        test_ratio = 0.1

        random.seed(525)
        np.random.seed(1101)
        if split_mode=='data':
        # automatic dataloading and splitting
            data_array = np.arange(0, total_num, 1)
            np.random.shuffle(data_array)
            torch.random.manual_seed(525)


            train_num = int(len(data_array) * train_ratio)
            test_num = int(len(data_array) * test_ratio)
            val_num = int(len(data_array) * validate_ratio)

            train_index = data_array[0:train_num]
            valid_index = data_array[train_num:train_num + val_num]
            if test_mode == 'fixed':
                test_index = data_array[total_num - test_num:]
            if test_mode == 'random':
                test_index = data_array[train_num + val_num:train_num + val_num + test_num]
        if split_mode == 'compound':
            efficient_index=np.load('dataset_save/compound_index.npy')
            compound_index = np.unique(efficient_index)
            all_index=np.arange(0,len(efficient_index),1)
            state = np.random.get_state()
            np.random.shuffle(compound_index)
            train_num = int(train_ratio * compound_index.shape[0])
            val_num = int(validate_ratio * compound_index.shape[0])
            test_num = int(test_ratio* compound_index.shape[0])
            compound_index = compound_index.tolist()
            compound_train = compound_index[0:train_num]
            compound_valid = compound_index[train_num:train_num + val_num]
            compound_test = compound_index[train_num + val_num:train_num + val_num + test_num]
            train_index=all_index[np.isin(efficient_index,compound_train)]
            valid_index = all_index[np.isin(efficient_index, compound_valid)]
            test_index = all_index[np.isin(efficient_index, compound_test)]
            print(test_index.shape)

        train_data_atom_bond = []
        valid_data_atom_bond = []
        test_data_atom_bond = []
        train_data_bond_angle = []
        valid_data_bond_angle = []
        test_data_bond_angle = []
        for i in test_index:
            test_data_atom_bond.append(dataset_graph_atom_bond[i])
            test_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in valid_index:
            valid_data_atom_bond.append(dataset_graph_atom_bond[i])
            valid_data_bond_angle.append(dataset_graph_bond_angle[i])
        for i in train_index:
            train_data_atom_bond.append(dataset_graph_atom_bond[i])
            train_data_bond_angle.append(dataset_graph_bond_angle[i])


        for i in range(len(train_data_atom_bond)):
            train_data_atom_bond[i].y*=(1+noise_level* np.random.uniform(-1, 1))

        train_loader_atom_bond = DataLoader(train_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
        valid_loader_atom_bond = DataLoader(valid_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)
        test_loader_atom_bond = DataLoader(test_data_atom_bond, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers)
        train_loader_bond_angle = DataLoader(train_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        valid_loader_bond_angle = DataLoader(valid_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        test_loader_bond_angle = DataLoader(test_data_bond_angle, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers)

        device = args.device
        criterion_fn = torch.nn.MSELoss()
        model = GINGraphPooling(**nn_params).to(device)
        num_params = sum(p.numel() for p in model.parameters())

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
        writer = SummaryWriter(log_dir=args.save_dir)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        print('===========Data Prepared================')

        if MODEL == 'Train':
            if Use_geometry_enhanced==True:
                try:
                    os.makedirs(f'saves/model_GeoGNN_different_noise_{noise_level}')
                except OSError:
                    pass


            for epoch in tqdm(range(1000)):

                train_mae = train(model, device, train_loader_atom_bond, train_loader_bond_angle, optimizer, criterion_fn)

                if (epoch + 1) % 50 == 0:
                    valid_mae_1,valid_mae_2 = eval(model, device, valid_loader_atom_bond, valid_loader_bond_angle)
                    y_true_1,y_pred_1,y_true_2,y_pred_2,R_square_1, test_mae_1, R_square_2,test_mae_2 = test(model, device, test_loader_atom_bond,
                                                                                    test_loader_bond_angle)
                    if Use_geometry_enhanced == True:
                        with open(f"saves/model_GeoGNN_different_noise_{noise_level}/GeoGNN.log", "a+") as f:
                            f.write(
                                f'epoch: {epoch+1}, MSE_train: {train_mae}, valid_t1:{valid_mae_1},  valid_t2:{valid_mae_2},'
                                f'  R_2_t1_test:{R_square_1.item()}, R_2_t2_test:{R_square_2.item()}\n')


                    print(train_mae, valid_mae_1,valid_mae_2, R_square_1, test_mae_1, R_square_2,test_mae_2)
                    if Use_geometry_enhanced==True:
                        torch.save(model.state_dict(), f'saves/model_GeoGNN_different_noise_{noise_level}/model_save_{epoch + 1}.pth')

        if MODEL == 'Test':
            model.load_state_dict(
                torch.load(f'saves/model_GeoGNN_different_noise_{noise_level}/model_save_950.pth'))
            y_true_1, y_pred_1, y_true_2, y_pred_2, R_square_1, test_mae_1, R_square_2, test_mae_2 = test(model, device,
                                                                                                          test_loader_atom_bond,
                                                                                                          test_loader_bond_angle)

            y_pred_t1 = y_pred_1.cpu().data.numpy()
            y_true_t1 = y_true_1.cpu().data.numpy()
            y_pred_t2 = y_pred_2.cpu().data.numpy()
            y_true_t2 = y_true_2.cpu().data.numpy()
            measure_t1, plot_t1 = measurement(y_true_t1, y_pred_t1, f't1_GeoGNN')
            measure_t2, plot_t2 = measurement(y_true_t2, y_pred_t2, f't2_GeoGNN')
            df = pd.DataFrame({'true_t1': y_true_t1.reshape(-1, ),
                               'true_t2': y_true_t2.reshape(-1, ),
                               'pred_t1': y_pred_t1.reshape(-1, ),
                               'pred_t2': y_pred_t2.reshape(-1, )})
            df.to_csv(f'result_save/different_noise/GeoGNN_different_noise_{noise_level}.csv')

def predict_separate(input,eluent,e,m,V_e,use_input='smile'):
    dataset = []
    dataset_mord = []
    dataset_attribute = []
    eluent = convert_eluent(eluent)
    e=convert_e(e)
    for i in range(len(input)):
        if use_input=='smile':
            smile=input[i]
        if use_input=='CAS':
            smile=convert_CAS_to_smile(input[i])
        obtain_3D_mol(smile,f'3D_mol_single/3D_mol')
        mol = Chem.MolFromMolFile(f'3D_mol_single/3D_mol.mol')
        # mol = AllChem.MolFromSmiles(smile)
        descriptor = mord(mol)
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        dataset.append(data)
        dataset_mord.append(descriptor)
        MolWt = Descriptors.ExactMolWt(mol)
        nRotB = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds
        HBD = Descriptors.NumHDonors(mol)  # Number of H bond donors
        HBA = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
        LogP = Descriptors.MolLogP(mol)  # LogP
        dataset_attribute.append([MolWt, nRotB, HBD, HBA, LogP])

    dataset_mord = np.array(dataset_mord)
    # for i in range(dataset_mord.shape[1]):
    #     if dataset_mord[0,i]!=dataset_mord[1,i]:
    #         print(i,dataset_mord[0,i],dataset_mord[1,i])
    dataset_attribute = np.array(dataset_attribute)


    graph_atom_bond = []
    graph_bond_angle = []
    big_index = []
    all_descriptor = dataset_mord[:,[153, 278, 884, 885, 1273, 1594, 431, 1768, 1769, 1288, 1521]]
    all_attribute =  dataset_attribute
    all_descriptor = np.hstack((all_attribute, all_descriptor))
    X_max_eluent=np.load('dataset_save/X_max_eluent.npy')
    X_min_eluent=np.load('dataset_save/X_min_eluent.npy')
    X_max_descriptor=np.load('dataset_save/X_max_descriptor.npy')
    X_min_descriptor=np.load('dataset_save/X_min_descriptor.npy')
    all_descriptor = (all_descriptor -X_min_descriptor) / (
                X_max_descriptor - X_min_descriptor + 1e-8)
    eluent = (eluent - X_min_eluent) / ( X_max_eluent - X_min_eluent + 1e-8)
    all_descriptor = torch.from_numpy(np.array(all_descriptor)).to(torch.int64)
    for i in range(len(dataset)):
        data = dataset[i]
        atom_feature = []
        bond_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        for name in bond_id_names:
            bond_feature.append(data[name])
        atom_feature = torch.from_numpy(np.array(atom_feature).T).to(torch.int64)
        bond_feature = torch.from_numpy(np.array(bond_feature).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        bond_angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
        edge_index = torch.from_numpy(data['edges'].T).to(torch.int64)
        bond_index = torch.from_numpy(data['BondAngleGraph_edges'].T).to(torch.int64)
        y = torch.Tensor([float(1)])
        prop = torch.ones([bond_feature.shape[0], eluent.shape[1]]) * eluent
        e_x = torch.ones([bond_feature.shape[0]]) * e
        m_x = torch.ones([bond_feature.shape[0]]) * m
        V_e_x = torch.ones([bond_feature.shape[0]]) * V_e

        bond_angle_descriptor = torch.ones([bond_angle_feature.shape[0], all_descriptor.shape[1]]) * all_descriptor[i]
        if Use_geometry_enhanced == True:
            bond_feature = torch.cat([bond_feature, bond_float_feature.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, prop], dim=1)
        bond_feature = torch.cat([bond_feature, e_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, m_x.reshape(-1, 1)], dim=1)
        bond_feature = torch.cat([bond_feature, V_e_x.reshape(-1, 1)], dim=1)
        if Use_column_info == True:
            diameter = torch.ones([bond_feature.shape[0]]) * 1.5
            column_length = torch.ones([bond_feature.shape[0]]) * 6.6
            density = torch.ones([bond_feature.shape[0]]) * 0.4458
            bond_feature = torch.cat([bond_feature, diameter.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, column_length.reshape(-1, 1)], dim=1)
            bond_feature = torch.cat([bond_feature, density.reshape(-1, 1)], dim=1)

        bond_angle_feature = bond_angle_feature.reshape(-1, 1)
        bond_angle_feature = torch.cat([bond_angle_feature.reshape(-1, 1), bond_angle_descriptor], dim=1)

        data_atom_bond = Data(atom_feature, edge_index, bond_feature, y)
        data_bond_angle = Data(edge_index=bond_index, edge_attr=bond_angle_feature)
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)

    args = parse_args()
    prepartion(args)
    nn_params = {
        'num_tasks': 6,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'descriptor_dim': 1827
    }
    loader_atom_bond= DataLoader(graph_atom_bond, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
    loader_bond_angle = DataLoader(graph_bond_angle, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

    device = args.device
    criterion_fn = torch.nn.MSELoss()
    model = GINGraphPooling(**nn_params).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    model.load_state_dict(
        torch.load(f'saves/model_GeoGNN_1124/model_save_300.pth'))
    model.eval()
    y_true_1 = []
    y_pred_1 = []
    y_pred_10_1 = []
    y_pred_90_1 = []
    y_true_2 = []
    y_pred_2 = []
    y_pred_10_2 = []
    y_pred_90_2 = []
    with torch.no_grad():
        for _, batch in enumerate(zip(loader_atom_bond, loader_bond_angle)):
            batch_atom_bond = batch[0]
            batch_bond_angle = batch[1]
            batch_atom_bond = batch_atom_bond.to(device)
            batch_bond_angle = batch_bond_angle.to(device)
            pred = model(batch_atom_bond, batch_bond_angle)[0]


            y_pred_1.append(pred[:, 1].detach().cpu())
            y_pred_10_1.append(pred[:, 0].detach().cpu())
            y_pred_90_1.append(pred[:, 2].detach().cpu())

            y_pred_2.append(pred[:, 4].detach().cpu())
            y_pred_10_2.append(pred[:, 3].detach().cpu())
            y_pred_90_2.append(pred[:, 5].detach().cpu())

    y_pred_1 = torch.cat(y_pred_1, dim=0)
    y_pred_10_1 = torch.cat(y_pred_10_1, dim=0)
    y_pred_90_1 = torch.cat(y_pred_90_1, dim=0)

    y_pred_2 = torch.cat(y_pred_2, dim=0)
    y_pred_10_2 = torch.cat(y_pred_10_2, dim=0)
    y_pred_90_2 = torch.cat(y_pred_90_2, dim=0)
    print(y_pred_10_1,y_pred_1,y_pred_90_1)
    print(y_pred_10_2,y_pred_2,y_pred_90_2)





















