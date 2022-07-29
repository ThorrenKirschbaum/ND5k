#!/usr/bin/env python

# coding: utf-8


# before switching to python on hemera5: change DataLoader import, comment out get_ipython()...


# # Imports

import datetime
import gc
import multiprocessing
import os
import random
import shutil

import ase
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from dscribe.descriptors import SOAP
from sklearn.decomposition import IncrementalPCA
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential as Seq, Linear, LeakyReLU, GRU
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data import DataLoader  # !#
from torch_geometric.nn import NNConv, Set2Set

# from torch_geometric.loader import DataLoader

# first change directory to THIS file for safety:
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # !
print(os.listdir())


## Datasets

# Convernience function to extract atom coordinates from dataFrame row:
def get_atoms_coordinates_from_row_columnname(row, columnname):
    assert (columnname in ["XTBCoordinates",
                           "xyz_pbe_relaxed"]), "Internal Error in function get_atoms_coordinates_from_row_columnname: Unknown columnname specified"

    # Get Coordinate Column (for Nd5k or Oe62)
    if columnname == "XTBCoordinates":
        i = 8
    elif columnname == "xyz_pbe_relaxed":
        i = 1

    # Get Coordinates and Columns
    xyz = np.array(row[columnname].split()[i:])
    n_atoms = int(len(xyz) / 4)
    xyz = xyz.reshape((n_atoms, 4))
    xyz = np.array_split(xyz, n_atoms, axis=1)
    atoms = xyz[0].reshape(n_atoms)
    x = xyz[1].astype('float32')
    y = xyz[2].astype('float32')
    z = xyz[3].astype('float32')

    xyz = np.stack([x, y, z], axis=1).reshape((n_atoms, 3))

    return atoms, xyz


# A Custom Dataset class to get some custom InMemory Datasets
class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        files = os.listdir(self.raw_dir)
        return files

    @property
    def processed_file_names(self):
        files = os.listdir(self.processed_dir)
        return files

    # a convenience function to save the data to disk
    def process_data(self, method):
        i = 0
        for raw_path in tqdm.tqdm(self.raw_paths):
            # Read data from `raw_path`.
            data = torch.load(raw_path)

            data = method(data)

            torch.save(data, self.processed_dir + '/data_{}.pt'.format(i))
            i += 1

    def len(self):
        return len(self.processed_paths)

    # get with the transform keyword
    def get(self, idx):
        data = torch.load(self.processed_dir + '/data_{}.pt'.format(idx))
        if self.transform is not None:
            data = self.transform(data)
        return data

    # save the data to disk, but transform it before using the pre_transform
    def save_to_one_file(self, path, raw):
        data_list = []
        paths = self.raw_paths if raw else self.processed_paths
        # i=0
        for processed_path in paths:
            # Read data from `raw_path`.
            data = torch.load(processed_path)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            # print(i)
            # i+=1
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), path)


# A InMemoryDataset class to read an entire Dataset into RAM for training
class MyDatasetInMemory(InMemoryDataset):
    def __init__(self, root, name='', transform=None, pre_transform=None):
        super(MyDatasetInMemory, self).__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.raw_dir + f'/data_slices{name}.pt')

    def save(self, name):
        torch.save((self.data, self.slices), self.raw_dir + f'/data_slices{name}.pt')

    # use transform to transform the dataset in place
    def transformDataset(self, transform, name):
        results = []
        my_len = self.__len__()

        results = [transform(self.__getitem__(i)) for i in range(my_len)]
        data, slices = InMemoryDataset.collate(results)
        torch.save((data, slices), self.raw_dir + f'/data_slices{name}.pt')
        del self.data, self.slices
        gc.collect()
        new_dataset = MyDatasetInMemory(root=self.root, name=name)
        return new_dataset


# Create a global_feature vector to mark NDs for later analysis
def get_global_features(row):
    # dictionary for ND sized
    nd_sizes_dict = {"ad": (10, 16),
                     "di": (14, 20),
                     "tri": (18, 24),
                     "123tet": (22, 28),
                     "121tet": (22, 28),
                     "1-2-3tet": (22, 28),
                     "1234pent": (26, 32),
                     "1212pent": (26, 32),
                     "12312hex": (26, 30),
                     "c35h36": (35, 36),
                     "c48h48": (48, 48),
                     "c53h48": (53, 48),
                     "c68h64": (68, 64),
                     "c74h64": (74, 64),
                     "c88h80": (88, 80),
                     "c104h78": (104, 78),
                     "c109h80": (109, 80),
                     'c133h100': (133, 100),
                     'c145h108': (145, 108),
                     'c147h100': (147, 100),
                     'c178h126': (178, 126),
                     'c231h146': (231, 146),
                     'c281h172': (281, 172)}
    # global feature vector:
    # [base ND: n C atoms, base ND: n H atoms, base nd: n total atoms including surface termination
    # total number of atoms, total number of C atoms, total number of heavy atoms,
    # one-hot-encoding of surface termination, one-hot-encoding of dopants (+0.5 per dopant)]
    # a1 to a6 are normalized to [0,1] by dividing through maximum number
    a1 = nd_sizes_dict[row["ND"]][0] / 109
    a2 = nd_sizes_dict[row["ND"]][1] / 80
    a3 = nd_sizes_dict[row["ND"]][1]
    if row["Surface"] == "OH":
        a3 *= 2
    elif row["Surface"] == "NH2":
        a3 *= 3
    a3 = a3 / 240
    a4 = row["XTBCoordinates"].split()[0]
    a4 = float(a4) / 349
    a5, a6 = 0, 0
    atoms, _ = get_atoms_coordinates_from_row_columnname(row, "XTBCoordinates")
    for atom in atoms:
        if atom == "C":
            a5 += 1
        if atom != "H":
            a6 += 1
    a5 = a5 / 109
    a6 = a6 / 191
    a7, a8, a9, a10 = 0, 0, 0, 0
    if row["Surface"] == "H":
        a7 = 1
    elif row["Surface"] == "F":
        a8 = 1
    elif row["Surface"] == "OH":
        a9 = 1
    elif row["Surface"] == "NH2":
        a10 = 1
    a11, a12, a13, a14 = 0, 0, 0, 0
    for dopant in [row["Dopant1"], row["Dopant2"]]:
        if dopant == "B":
            a11 += 0.5
        elif dopant == "N":
            a12 += 0.5
        elif dopant == "Si":
            a13 += 0.5
        elif dopant == "P":
            a14 += 0.5
    a15 = 1
    if row["Surface"] == "NH2" and row["Dopant1"] == "N" and row["Dopant2"] == "N":
        a15 = (atoms.tolist().count(row["Surface"][0]) - 2) / nd_sizes_dict[row["ND"]][1]
    elif row["Surface"] == "NH2" and (row["Dopant1"] == "N" or row["Dopant2"] == "N"):
        a15 = (atoms.tolist().count(row["Surface"][0]) - 1) / nd_sizes_dict[row["ND"]][1]
    elif row["Surface"] != "H":
        a15 = atoms.tolist().count(row["Surface"][0]) / nd_sizes_dict[row["ND"]][1]  # surface species coverage
    if ((np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14]) > 1).any() or a15 > 1.1) and \
            nd_sizes_dict[row["ND"]][0] < 133:
        print("Warning: Global feature normalization not correct")
        print([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15])
        print(row["XTBCoordinates"])
    return [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]


# Similar function for OE62, eventhough these features would be used mainly for ML
# taken from https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c00687 (their github)
# NOT TESTED
def get_global_features_Oe62k(inchi):
    # this is the implementation of the 'Molecular Descriptors' from the paper the nn architecture is based on
    mol = rdChem.MolFromInchi(inchi)
    u = []
    # Now get some specific features
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)

    # First get some basic features
    natoms = mol.GetNumAtoms()
    nbonds = mol.GetNumBonds()
    mw = Descriptors.ExactMolWt(mol)
    HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(mol)
    NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
    ''' # These four descriptors are producing the value of infinity for refcode_csd = YOLJUF (CCOP(=O)(Cc1ccc(cc1)NC(=S)NP(OC(C)C)(OC(C)C)[S])OCC\t\n)
    MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(mol)
    MaxPartialCharge = Descriptors.MaxPartialCharge(mol)
    MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(mol)
    MinPartialCharge = Descriptors.MinPartialCharge(mol)
    '''

    # Get some features using chemical feature factory
    nbrAcceptor = 0
    nbrDonor = 0
    nbrHydrophobe = 0
    nbrLumpedHydrophobe = 0
    nbrPosIonizable = 0
    nbrNegIonizable = 0

    for j in range(len(feats)):
        # print(feats[j].GetFamily(), feats[j].GetType())
        if ('Acceptor' == (feats[j].GetFamily())):
            nbrAcceptor = nbrAcceptor + 1
        elif ('Donor' == (feats[j].GetFamily())):
            nbrDonor = nbrDonor + 1
        elif ('Hydrophobe' == (feats[j].GetFamily())):
            nbrHydrophobe = nbrHydrophobe + 1
        elif ('LumpedHydrophobe' == (feats[j].GetFamily())):
            nbrLumpedHydrophobe = nbrLumpedHydrophobe + 1
        elif ('PosIonizable' == (feats[j].GetFamily())):
            nbrPosIonizable = nbrPosIonizable + 1
        elif ('NegIonizable' == (feats[j].GetFamily())):
            nbrNegIonizable = nbrNegIonizable + 1
        else:
            pass
            # print(feats[j].GetFamily())

    # Now get some features using rdMolDescriptors
    moreGlobalFeatures = [rdm.CalcNumRotatableBonds(mol), rdm.CalcChi0n(mol), rdm.CalcChi0v(mol), \
                          rdm.CalcChi1n(mol), rdm.CalcChi1v(mol), rdm.CalcChi2n(mol), rdm.CalcChi2v(mol), \
                          rdm.CalcChi3n(mol), rdm.CalcChi4n(mol), rdm.CalcChi4v(mol), \
                          rdm.CalcFractionCSP3(mol), rdm.CalcHallKierAlpha(mol), rdm.CalcKappa1(mol), \
                          rdm.CalcKappa2(mol), rdm.CalcLabuteASA(mol), \
                          rdm.CalcNumAliphaticCarbocycles(mol), rdm.CalcNumAliphaticHeterocycles(mol), \
                          rdm.CalcNumAliphaticRings(mol), rdm.CalcNumAmideBonds(mol), \
                          rdm.CalcNumAromaticCarbocycles(mol), rdm.CalcNumAromaticHeterocycles(mol), \
                          rdm.CalcNumAromaticRings(mol), rdm.CalcNumBridgeheadAtoms(mol), rdm.CalcNumHBA(mol), \
                          rdm.CalcNumHBD(mol), rdm.CalcNumHeteroatoms(mol), rdm.CalcNumHeterocycles(mol), \
                          rdm.CalcNumLipinskiHBA(mol), rdm.CalcNumLipinskiHBD(mol), rdm.CalcNumRings(mol), \
                          rdm.CalcNumSaturatedCarbocycles(mol), rdm.CalcNumSaturatedHeterocycles(mol), \
                          rdm.CalcNumSaturatedRings(mol), rdm.CalcNumSpiroAtoms(mol), rdm.CalcTPSA(mol)]

    u = [natoms, nbonds, mw, HeavyAtomMolWt, NumValenceElectrons, \
         nbrAcceptor, nbrDonor, nbrHydrophobe, nbrLumpedHydrophobe, \
         nbrPosIonizable, nbrNegIonizable]

    u = u + moreGlobalFeatures
    u = np.array(u).T
    # Some of the descriptors produice NAN. We can convert them to 0
    # If you are getting outliers in the training or validation set this could be
    # Because some important features were set to zero here because it produced NAN
    # Removing those features from the feature set might remove the outliers

    # u[np.isnan(u)] = 0

    # u = torch.tensor(u, dtype=torch.float)
    return (u)


# Function to create a list of PyTorch datapoints from the ND5k DataFrame
def get_ND_datalist_from_dataframe(df, target, fingerprint):
    ND_dataset = []
    # atom = node features #
    # dictionary to assign node feature vectors to elements
    # [at. number, valence, radius in pm, Pauling electronegativity,
    #  electron affinity in kJ/mol, 1st ionization energy in 10E2 kJ/mol]
    atom_feature_dict = {"H": [1, 1, 25, 2.20, 73, 13.12, 1, 0, 0, 0, 0, 0, 0, 0],
                         "B": [5, 3, 85, 2.04, 27, 8.00, 0, 1, 0, 0, 0, 0, 0, 0],
                         "C": [6, 4, 70, 2.55, 122, 10.87, 0, 0, 1, 0, 0, 0, 0, 0],
                         "N": [7, 3, 65, 3.04, -7, 14.02, 0, 0, 0, 1, 0, 0, 0, 0],
                         "O": [8, 2, 60, 3.44, 141, 13.14, 0, 0, 0, 0, 1, 0, 0, 0],
                         "F": [9, 1, 50, 3.98, 328, 16.81, 0, 0, 0, 0, 0, 1, 0, 0],
                         "Si": [15, 4, 110, 1.90, 134, 7.87, 0, 0, 0, 0, 0, 0, 1, 0],
                         "P": [16, 3, 100, 2.19, 72, 10.12, 0, 0, 0, 0, 0, 0, 0, 1]}
    descriptor_dict = {'ad': 1, 'di': 2, 'tri': 3, '1-2-3tet': 4, '121tet': 5, '123tet': 6,
                       '12312hex': 7, '1212pent': 8, '1234pent': 9, 'c35h36': 10, 'c48h48': 11,
                       'c53h48': 12, 'c68h64': 13, 'c74h64': 14, 'c88h80': 15, 'c104h78': 16, 'c109h80': 17,
                       'c133h100': 18, 'c145h108': 19, 'c147h100': 20, 'c178h126': 21, 'c231h146': 22, 'c281h172': 23,
                       "H": 0, "NH2": 1, "OH": 2, "F": 3,
                       "C": 0, "B": 1, "N": 2, "Si": 3, "P": 4}

    # iterate through df

    for i, row in df.iterrows():
        atoms, xyz = get_atoms_coordinates_from_row_columnname(row, "XTBCoordinates")

        # atom = node positions #
        pos = torch.tensor(xyz, dtype=torch.float)
        # normalization around origin (only needed if master node at [0,0,0] is inserted)
        # pos = pos - pos.mean(dim=-2, keepdim=True)

        # if not soap or soap pca, use "classical" node features (from atom_feature_dict)
        if fingerprint is None:
            node_attr_size = len(atom_feature_dict["H"])
            node_attributes = np.zeros((len(atoms), node_attr_size), dtype=np.float32)
            if len(atoms) == 0: print("Warning: NDs: No atoms found.")
            z = []
            i = 0
            norm_factors = [16, 4, 110, 3.98, 328, 16.81, 1, 1, 1, 1, 1, 1, 1, 1]

            norm_factors = np.array(norm_factors)
            for atom in atoms:
                x_new = atom_feature_dict[atom]
                z.append(x_new[0])
                x_new = np.array(x_new) / norm_factors  # normalize all features to 0,1
                node_attributes[i] = x_new
                i += 1
            if (node_attributes > 1).any(): print("Warning: ND node feature normalization not correct")
            x = torch.tensor(node_attributes, dtype=torch.float)

        # just keep atomic number otherwise
        else:
            z = []
            for atom in atoms:
                z.append(atom_feature_dict[atom][0])
            x = torch.zeros(len(atoms), dtype=torch.float)
        z = torch.tensor(z, dtype=torch.long)

        # graph attributes: HOMO and LUMO #
        HOMO = torch.tensor(row["HOMO"], dtype=torch.float)
        LUMO = torch.tensor(row["LUMO"], dtype=torch.float)
        y = HOMO if target == "HOMO" else LUMO

        # global features of the graph: ND index from dataset and descriptive vector for linear regression (see above) #
        index = row["ND_index"]
        index = torch.tensor(index, dtype=torch.int)
        global_features = get_global_features(row)
        global_features = torch.tensor(global_features, dtype=torch.float)

        # the descriptor is an integer number that encodes the information on ND type, surface termination, surface coverage (maximal or mixed with H), dopant 1, dopant 2
        descriptor = descriptor_dict[row["ND"]] * 10000 + descriptor_dict[row["Surface"]] * 1000 + int(
            row["Maximal_coverage"]) * 100 + descriptor_dict[row["Dopant1"]] * 10 + descriptor_dict[row["Dopant2"]] * 1
        descriptor = torch.tensor(descriptor, dtype=torch.int)

        # make a new datapoint
        new_datapoint = Data(x=x, y=y, z=z, HOMO=HOMO, LUMO=LUMO, pos=pos, global_features=global_features,
                             descriptor=descriptor, index=index)
        ND_dataset.append(new_datapoint)
    return ND_dataset


# similar Function for the OE dataset
def get_OE62_datalist_from_dataframe(df, target, fingerprint):
    OE62_dataset = []
    for i, row in tqdm.tqdm(df.iterrows()):
        atoms, xyz = get_atoms_coordinates_from_row_columnname(row, "xyz_pbe_relaxed")

        # atom = node positions #
        pos = torch.tensor(xyz, dtype=torch.float)
        # normalization around origin (only needed if master node at [0,0,0] is inserted)
        # pos = pos - pos.mean(dim=-2, keepdim=True)

        # atom = node features #
        # dictionary to assign node feature vectors to elements
        # [at. number, valence, radius in pm, Pauling electronegativity,
        #  electron affinity in kJ/mol, 1st ionization energy in 10E2 kJ/mol]
        atom_feature_dict = {"H": [1, 1, 25, 2.20, 73, 13.12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             "Li": [3, 1, 145, 0.98, 60, 5.20, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                             "B": [5, 3, 85, 2.04, 27, 8.00, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             "C": [6, 4, 70, 2.55, 122, 10.87, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             "N": [7, 3, 65, 3.04, -7, 14.02, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             "O": [8, 2, 60, 3.44, 141, 13.14, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             "F": [9, 1, 50, 3.98, 328, 16.81, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             "Si": [15, 4, 110, 1.90, 134, 7.87, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             "P": [16, 3, 100, 2.19, 72, 10.12, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             "S": [16, 2, 100, 2.58, 200, 10.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                             "Cl": [17, 1, 100, 3.16, 349, 12.51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                             "As": [33, 3, 115, 2.18, 78, 9.47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             "Se": [34, 2, 115, 2.55, 195, 9.41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                             "Br": [35, 1, 115, 2.96, 325, 11.40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             "Te": [52, 2, 140, 2.10, 190, 8.69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                             "I": [53, 1, 140, 2.66, 295, 10.08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             }

        node_attributes = np.zeros((len(atoms), len(atom_feature_dict["H"])))
        element_numbers = np.zeros((len(atoms)))
        if len(atoms) == 0: print("Warning: OE62: No atoms found.")
        i = 0
        norm_factors = np.array([53, 4, 145, 3.98, 349, 16.81, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        for atom in atoms:
            x_new = atom_feature_dict[atom]
            z = x_new[0]
            if fingerprint is not None:
                x_new = z
            else:
                x_new = np.array(x_new) / norm_factors  # normalize all features to 0,1
            node_attributes[i] = x_new
            element_numbers[i] = z
            i += 1
        if (node_attributes > 1).any() and fingerprint is None: print(
            "Warning: OE62 node feature normalization not correct")
        x = torch.tensor(node_attributes, dtype=torch.float)

        # graph attributes: HOMO and LUMO #
        HOMO = torch.tensor(row["HOMO"], dtype=torch.float)
        LUMO = torch.tensor(row["LUMO"], dtype=torch.float)
        y = HOMO if target == "HOMO" else LUMO

        # global features of the graph #
        # global_features = get_global_features_Oe62k(row['inchi'])
        # !!! This wont be in the dataset, check below
        global_features = torch.tensor(0, dtype=torch.float)
        index = torch.tensor(i, dtype=torch.int)

        new_datapoint = Data(x=x, y=y, z=element_numbers, HOMO=HOMO, LUMO=LUMO, pos=pos, index=index,
                             global_features=global_features)
        OE62_dataset.append(new_datapoint)
    return OE62_dataset


# Now create some datasets
def get_any_dataset(csv_path, target, graph_radius, dataset_type='nd', datapaths=["./data/", "./data_inmemory/"],
                    normalize_targets=True,
                    testing=False, print_info=False, fingerprint=None, pre_transform=None, save=True):
    assert (target in ["HOMO", "LUMO"]), "Unknown learning target."

    # get data from csv as pandas dataframe
    if dataset_type == 'nd' or dataset_type == 'big nd':
        df = pd.read_csv(csv_path)
    else:
        # get data from json as pandas dataframe
        df = pd.read_json(csv_path, orient='split')

        # delete too small molecules
        df.drop(37260, inplace=True)
        df.drop(37271, inplace=True)

        # drop unused columns
        for c in df.columns:
            if c not in ["xyz_pbe_relaxed", "energies_occ_pbe0_vac_tier2", "energies_unocc_pbe0_vac_tier2", 'inchi']:
                df.drop(c, axis=1, inplace=True)

        for i, row in df.iterrows():
            df.at[i, 'HOMO'] = row["energies_occ_pbe0_vac_tier2"][-1]
            df.at[i, 'LUMO'] = row["energies_unocc_pbe0_vac_tier2"][0]

        df.drop(["energies_occ_pbe0_vac_tier2", "energies_unocc_pbe0_vac_tier2"], axis=1, inplace=True)

    if testing:
        df = df[:256]  # !
    if normalize_targets:
        df["HOMO"] = (df["HOMO"] + 10.6) / 10.7  # !
        df["LUMO"] = (df["LUMO"] + 5.6) / 7.2  # !
        normed_HOMOs = np.array(df["HOMO"])
        normed_LUMOs = np.array(df["LUMO"])
        # assert that normalization gives the expected results
        if (normed_HOMOs < 0).any() or (normed_HOMOs > 1).any():
            print("HOMO normailzation not correct")
        if dataset_type == 'nd' or dataset_type == 'big nd':
            if "c133h100" not in df.values and (
                    (np.count_nonzero(normed_LUMOs < 0) != 6 and not testing) or (
                    (normed_LUMOs < 0).any() and testing) or (
                            normed_LUMOs > 1).any()):
                print("LUMO normailzation not correct: max {}, min {}, {} < 0".format(max(normed_LUMOs),
                                                                                      min(normed_LUMOs),
                                                                                      np.count_nonzero(
                                                                                          normed_LUMOs < 0)))
                print("c133h100" not in df.values, (np.count_nonzero(normed_LUMOs < 0) != 6 and not testing),
                      ((normed_LUMOs < 0).any() and testing), (normed_LUMOs > 1).any())
        elif dataset_type == 'Oe62k':
            if (normed_LUMOs < 0).any() or (normed_LUMOs > 1).any():
                print("OE62 LUMO normailzation not correct")

    # make list of datapoints
    if dataset_type == 'nd' or dataset_type == 'big nd':
        data_list = get_ND_datalist_from_dataframe(df=df, target=target, fingerprint=fingerprint)
    elif dataset_type == 'Oe62k':
        data_list = get_OE62_datalist_from_dataframe(df=df, target=target, fingerprint=fingerprint)

    data_file_names = []

    # clear old files
    old_raw_files = os.listdir(datapaths[0] + "raw")
    for file in old_raw_files:
        os.remove(datapaths[0] + "raw/" + file)

    # save new files
    i = 0
    for datapoint in data_list:
        data_file_names.append(datapaths[0] + 'raw/datapoint_{}.pt'.format(i))
        torch.save(datapoint, data_file_names[-1])
        i += 1

    # delete previous processed files and initialize dataset
    processed_files = os.listdir(datapaths[0] + "/processed")
    for file in processed_files:
        os.remove(datapaths[0] + "processed/" + file)
    nd_dataset = MyDataset(root=datapaths[0], pre_transform=pre_transform)

    # process and save data - Create a Radius Graph with radius graph_radius
    nd_dataset.process_data(
        method=T.Compose([T.RadiusGraph(r=graph_radius, loop=True, max_num_neighbors=256), T.Distance()]))
    nd_dataset.save_to_one_file(path=datapaths[1] + 'raw/data_slices.pt', raw=False)
    del nd_dataset
    gc.collect()
    print(datapaths[1])
    # delete previous processed files and initialize the in memory dataset
    processed_files = os.listdir(datapaths[1] + "processed")
    for file in processed_files:
        os.remove(datapaths[1] + "processed/" + file)
    nd_dataset = MyDatasetInMemory(root=datapaths[1])

    # print info
    if print_info:
        # get dataset info
        print(f'Dataset: {nd_dataset}:')
        print(f'Number of graphs: {len(nd_dataset)}')
        print(f'Number of features: {nd_dataset.num_features}')
        print(f'Number of classes: {nd_dataset.num_classes}')

        data = nd_dataset[0]  # Get the first graph object.
        print()
        print("First datapoint:", data)
        # Gather some statistics about the first graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
        print(f'Contains self-loops: {data.contains_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')

        # check for isolated nodes in the graphs
        for i in range(len(nd_dataset)):
            if nd_dataset[i].contains_isolated_nodes(): print("Structure", i, "contains isolated nodes")
    print(type(nd_dataset))
    return nd_dataset


def get_min_max_temp_pca_func(dataset, transform):
    transformed = transform(dataset.data.x)
    min_ = transformed.min(axis=0)
    max_ = transformed.max(axis=0)

    return min_, max_


def get_max_min_manually(dataset, transform):
    test_x_vector = transform(dataset.get(0)).x
    x_max_so_far = torch.ones(test_x_vector.shape[-1], dtype=torch.float32) * -1.e16
    x_min_so_far = torch.ones(test_x_vector.shape[-1], dtype=torch.float32) * 1.e16
    for entry in tqdm.tqdm(dataset):
        vectors = transform(entry).x
        for atom in range(len(vectors)):
            atom_vector = vectors[atom]
            x_max_so_far[x_max_so_far < atom_vector] = atom_vector[x_max_so_far < atom_vector]
            x_min_so_far[x_min_so_far > atom_vector] = atom_vector[x_min_so_far > atom_vector]

    if (x_max_so_far == -1e-16).any() or (x_min_so_far == 1e-16).any():
        print(f'ATTENTION: Norm Failed! {transform}')
    if (x_max_so_far.isnan()).any() or (x_min_so_far.isnan()).any():
        print(f'ATTENTION: Norm Failed! Nan!')

    return x_min_so_far, x_max_so_far


class SoapTransform(object):
    '''
    This Calculates the SOAP Fingerprint at every Atoms Position with the parameters from fingerprint_dict
    and will put the fingerprints in the data.x attribute
    these will be normed according to min_ and max_
    '''

    def __init__(self, fingerprint_dict, dataset_type, min_=0, max_=1, pre_norm=False):
        """
        :param fingerprint_dict: parameters for the fingerprint in question
                                needs keys: rcut, nmax, lmax, rbf, weighting and sigma (check dScribe Documentation)
        :param dataset_type: nd, big nd, oe62
        :param min_: in case of norming: minimum value after SOAP transform
        :param max_: max value after raw SOAP transform
        :param pre_norm: whether to norm the dataset after the SOAP transform
        """

        if dataset_type == 'nd' or dataset_type == 'big nd':
            species = ["H", "B", "C", "N", "O", "F", "Si", "P", "S"]
        elif dataset_type == 'Oe62k':
            species = ["H", "Li", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I"]

        self.pre_norm = pre_norm

        self.soap = SOAP(rcut=fingerprint_dict['rcut'], nmax=fingerprint_dict['nmax'], lmax=fingerprint_dict['lmax'],
                         rbf=fingerprint_dict['rbf'], weighting=fingerprint_dict['weighting'],
                         sigma=fingerprint_dict['sigma'], species=species)

        soap_feature_length = self.soap.get_number_of_features()
        print(f'soap-Features: {soap_feature_length}')
        self.active_cores = multiprocessing.cpu_count() - 1

        self.soap_max = max_
        self.soap_min = min_
        self.adjusted_max = max_ - min_

        if type(max_) != int and type(min_) != int:
            self.adjusted_max[self.adjusted_max == 0] = 1  # just in case max==min

    def __call__(self, data):
        ase_atoms = ase.Atoms(symbols=data.z, positions=data.pos)

        soap_features = self.soap.create(ase_atoms, n_jobs=self.active_cores)
        soap_features = torch.tensor(soap_features, dtype=torch.float)
        soap_features = (soap_features - self.soap_min) / self.adjusted_max

        if soap_features.isnan().any(): print('Warning Soap Nan!', 'min==max?',
                                              (self.soap_min == self.soap_max).any())

        if not self.pre_norm and ((soap_features < 0).any() or (soap_features > 1).any()): print(
            'Warning Soap Norm not correct!')

        data.x = soap_features
        return data


# just a function to get the Soap min and max value
def get_soap_min_max(dataset, fingerprint_dict, dataset_type='nd', soap_max=1, soap_min=0):
    transform = SoapTransform(fingerprint_dict, dataset_type, pre_norm=True,
                              max_=soap_max, min_=soap_min)

    min_, max_ = get_max_min_manually(dataset, transform)

    return min_, max_


# fit a pca transform to the x (node) attributes of a dataset
def get_pca_transform(dataset):
    pca_transform = IncrementalPCA(n_components=pca_components)
    pca_transform.fit(dataset.data.x)
    print('No PCA Components:', pca_transform.explained_variance_.shape)
    return pca_transform


# To transform the Dataset
class pcaTranformDataset():
    def __init__(self, pcaTransform, min_, max_):
        """
        :param pcaTransform: A sklearn pcaTransform class fit to the dataset
        :param min_: min and max vectors after the transform to norm the dataset
        :param max_:
        """
        self.min_ = min_
        self.adjusted_max = max_ - min_
        # to avoid nans if min = max:
        if type(max_) != int and type(min_) != int:
            self.adjusted_max[self.adjusted_max == 0] = 1
        self.pcaTransform = pcaTransform

    def __call__(self, data):
        data.x = self.pcaTransform.transform(data.x)
        data.x = torch.tensor(data.x, dtype=torch.float)
        data.x -= self.min_
        data.x /= self.adjusted_max
        data.x = data.x.float()
        if (data.x < -0.1).any() or (data.x > 1.1).any():
            print(f'Norm incorrect{data.x.max()},{data.x.min()}')
        return data


class DatasetLogger:
    """
    This class saves and keeps track of already created datasets such that already created datasets dont have to be re-
    calculated.
    This is a little dangerous as it can delete and overwrite files under log path
    """

    def __init__(self, log_path):
        """
        :param log_path: The Path under which datasets are saved and created
        """
        self.path = log_path
        self.logfile = log_path + '/datasets.df'
        self.buffer = None

        # use pickle for io, since the Dataframe contains lists dicts, tensors, etc. and pickle recovers them
        if os.path.isfile(self.logfile):
            self.data = pd.read_pickle(self.logfile)
        else:
            self.data = pd.DataFrame(
                columns=['dataset_type', 'target', 'paths', 'testing', 'fingerprint', 'fingerprint_dict', 'radius',
                         'self_loops',
                         'normalize_targets', 'date', 'min', 'max', 'pca_transform'])

        if not os.path.isdir(self.path):
            os.makedirs(self.path)
            os.makedirs(self.path + '/tmp')

    def find_row(self, dict):
        # just a helper to find a matching row in a dataFrame
        matching_indeces = []
        temp_df = self.data.replace(np.nan, "NONE!")

        for idx, row in temp_df.iterrows():
            if (row[dict.keys()].compare(pd.Series(dict).replace(np.nan, "NONE!"), keep_shape=True).isna()).all(
                    axis=None):
                matching_indeces.append(idx)

        if len(matching_indeces) > 1:
            print('Warning: More than 1 Matching Dataset found!')
            matching_indeces = [matching_indeces[-1]]
        elif len(matching_indeces) == 0:
            return None, None

        return self.data.iloc[matching_indeces], matching_indeces[-1]

    def build_file_structure(self, paths):
        # create the file strucuture to save datasets
        for path in paths:
            os.makedirs(path + 'raw', exist_ok=True)
            os.makedirs(path + 'processed', exist_ok=True)

    def build_a_normed_dataset(self, dataset_type, target='HOMO', radius=3.5,
                               datapaths=["./data/", "./data_inmemory/"], normalize_targets=True, testing=False,
                               fingerprint=None, fingerprint_dict=None):
        """
        This function will create and return a InMemoryDataset of given specification
        :param dataset_type: 'nd', 'big nd' or 'oe62k'
        :param target: HOMO or LUMO
        :param radius: 'graph_radius'
        :param datapaths: where the data will be stored, just dont touch this
        :param normalize_targets: whether to norm the target values between 0 and 1
        :param testing: just use a few samples for testing purposes
        :param fingerprint: 'soap' or 'soap_pca'
        :param fingerprint_dict: 'params for SOAP
        :return: MyDatasetInMemory containing the Dataset
        """

        if dataset_type == 'nd':
            csv = "./data_sorted.csv"

        elif dataset_type == 'big nd':
            csv = "./testdata_big_nds.csv"

        elif dataset_type == 'Oe62k':
            csv = "./df_62k_original.json"

        # get a dataset first
        dataset = get_any_dataset(csv_path=csv, target=target, graph_radius=radius, dataset_type=dataset_type,
                                  datapaths=datapaths, normalize_targets=normalize_targets, testing=testing,
                                  print_info=testing, fingerprint=fingerprint, save=(fingerprint is None))

        if fingerprint == 'soap' or fingerprint == 'soap_pca':
            # we will have to get the min and max values for the soap vectors if its the first dataset
            if fingerprint == 'soap':
                # calculate the elementwise min and max vectors for a soap dataset to norm
                min_, max_ = get_soap_min_max(dataset, fingerprint_dict, dataset_type=dataset_type)
            else:
                min_, max_ = 0, 1
            step_before_pca = (fingerprint != 'soap_pca')
            del dataset
            gc.collect()
            # get a normed soap transformed dataset if soap or a not normed soap transformed dataset otherwise
            dataset = get_any_dataset(csv_path=csv, target=target, graph_radius=radius, dataset_type=dataset_type,
                                      datapaths=datapaths, normalize_targets=normalize_targets, testing=testing,
                                      print_info=False, fingerprint=fingerprint,
                                      pre_transform=SoapTransform(fingerprint_dict, dataset_type=dataset_type,
                                                                  min_=min_, max_=max_,
                                                                  pre_norm=(dataset_type != 'soap')),
                                      save=step_before_pca)

        if fingerprint is None:
            min_, max_ = 0, 1

        return dataset, min_, max_

    def get_dataset(self, dataset_type, target, radius, self_loops=True, normalize_targets=True, fingerprint=None,
                    fingerprint_dict=None,
                    testing=False, reload=False):
        """
        This essentially does the same job as build a normed dataset, it gets datasets and returns them, however this
        also checks the dataframe
        under dataset_path to identify whether the dataset has already been stored and just loads it in this case.
        :param dataset_type:
        :param target:
        :param radius:
        :param self_loops:
        :param normalize_targets:
        :param fingerprint:
        :param fingerprint_dict:
        :param testing:
        :param reload: whether to override the stored datasets
        :return:
        """
        if testing: print('Getting Dataset')
        assert dataset_type in ['nd', 'big nd', 'Oe62k']
        # base path is a basic encoding of the Datset Properties
        dataset_path = f'{self.path}/{dataset_type}_dataset_{target}_r_{radius}_f_{fingerprint}'

        param_dict = {'dataset_type': dataset_type,
                      'target': target,
                      'radius': radius,
                      'self_loops': self_loops,
                      'normalize_targets': normalize_targets,
                      'fingerprint': fingerprint,
                      'fingerprint_dict': fingerprint_dict,
                      'testing': testing}

        row, idx = self.find_row(param_dict)

        if (row is None) or reload:
            if reload and (row is not None):
                # delete directory if it already exists
                paths = self.data.at[idx, 'paths']
                shutil.rmtree('/'.join(paths[0].split('/')[:-2]) + '/', ignore_errors=True)
                self.data.drop(index=idx, inplace=True)

            # we have to create a new dataset if nothing has been found
            # find an empty directory:
            dataset_dirs = os.listdir(self.path)
            for i in range(len(dataset_dirs) + 10):
                working_path = dataset_path + str(i)
                if not os.path.exists(working_path):
                    break

            if working_path != dataset_path + str(i):
                print('Warning - No Empty Dataset Directory found!')
                return None

            # and create a dataset according to the specifics
            print(f'Creating new Dataset at{working_path}')
            paths = [f'{working_path}/data/', f'{working_path}/data_inmemory/']
            self.build_file_structure(paths)

            dataset, min_, max_ = self.build_a_normed_dataset(dataset_type, target=target, radius=radius,
                                                              datapaths=paths,
                                                              normalize_targets=normalize_targets,
                                                              fingerprint=fingerprint,
                                                              fingerprint_dict=fingerprint_dict, testing=testing)
            param_dict['min_'] = min_
            param_dict['max_'] = max_
            param_dict['paths'] = paths
            param_dict['date'] = pd.to_datetime(datetime.datetime.now(), dayfirst=True)

            self.data = self.data.append(param_dict, ignore_index=True)
            self.data.to_pickle(self.logfile)
            idx = len(self.data) - 1
            row = self.data.iloc[idx]

        else:
            paths = self.data.at[idx, 'paths']
            print(f'reloading old dataset at {paths[1]}')
            dataset = MyDatasetInMemory(root=paths[1])

        return dataset, paths[1]  # !

    def clean_dir(self):
        # this should be used occasionally to delete unused Datasets in a directory

        # print(self.data['paths'][0][0].split('/'))
        # create a list of the folders that should be in the directory
        used_folders = [row['paths'][0].split('/')[2] for _, row in self.data.iterrows() if
                        len(row['paths'][0].split('/')) > 3]

        os.chdir(self.path)
        folders = os.listdir()
        for folder in folders:
            if os.path.isfile(folder):
                continue
            elif folder in used_folders:
                continue
            else:
                print(folder)

                shutil.rmtree(folder, ignore_errors=True)

        os.chdir('..')


## Learning

def stratified_split_train_val_test(dataset, target, norm_targets,
                                    plot_distribution=False):
    # this creates a stratified split of the ND5k dataset
    nd_configurations_in_val_test_dataset = []
    indexes_train, indexes_val_test = [], []
    i = 0
    a = False
    for datapoint in dataset:
        if target == "LUMO" and (not norm_targets) and (float(datapoint["LUMO"])) < -6:  # removes 6 outliers
            i += 1
            continue
        if target == "LUMO" and norm_targets and float(datapoint["LUMO"]) < 0:  # removes 6 outliers
            i += 1
            continue
        nd_configuration = int(datapoint.descriptor)
        # if no dopants are present choose randomly (alternating) to which sub-dataset the point is assigned
        if nd_configuration % 100 == 0:
            if a:
                indexes_train.append(i)
            else:
                indexes_val_test.append(i)
            a = not a
        # otherwise assign datapoint to val+test only if this ND configuration 
        # is not yet present in the val+test dataset
        elif nd_configuration in nd_configurations_in_val_test_dataset:
            indexes_train.append(i)
        else:
            indexes_val_test.append(i)
            nd_configurations_in_val_test_dataset.append(nd_configuration)
        i += 1

    train_dataset = dataset[indexes_train]
    val_test_dataset = dataset[indexes_val_test]
    val_dataset = val_test_dataset[round(len(val_test_dataset) / 2):]
    test_dataset = val_test_dataset[:round(len(val_test_dataset) / 2)]

    if (np.array([len(train_dataset), len(val_dataset), len(test_dataset)]) == 0).any():
        return 0

    if plot_distribution:
        # create dataframes 
        train_list, val_list, test_list = [], [], []
        for data in train_dataset:
            train_list.append([float(data["HOMO"]), float(data["LUMO"])])
        for data in val_dataset:
            val_list.append([float(data["HOMO"]), float(data["LUMO"])])
        for data in test_dataset:
            test_list.append([float(data["HOMO"]), float(data["LUMO"])])
        train_df = pd.DataFrame(train_list, columns=["HOMO", "LUMO"])
        val_df = pd.DataFrame(val_list, columns=["HOMO", "LUMO"])
        test_df = pd.DataFrame(test_list, columns=["HOMO", "LUMO"])
        del train_list, val_list, test_list
        # HOMO
        plt.hist(train_df["HOMO"], bins=np.linspace(-11, 1, 50), alpha=0.5, label='Train dataset')
        plt.hist(val_df["HOMO"], bins=np.linspace(-11, 1, 50), alpha=0.5, label='Val. dataset')
        plt.hist(test_df["HOMO"], bins=np.linspace(-11, 1, 50), alpha=0.5, label='Test dataset')
        plt.xlabel("HOMO energy / eV")
        plt.ylabel("Number of NDs")
        plt.legend()
        plt.title("HOMO")
        plt.show()
        plt.close()
        # LUMO
        plt.hist(train_df["LUMO"], bins=np.linspace(-6, 2, 50), alpha=0.5, label='Train dataset')
        plt.hist(val_df["LUMO"], bins=np.linspace(-6, 2, 50), alpha=0.5, label='Val. dataset')
        plt.hist(test_df["LUMO"], bins=np.linspace(-6, 2, 50), alpha=0.5, label='Test dataset')
        plt.xlabel("LUMO energy / eV")
        plt.ylabel("Number of NDs")
        plt.legend()
        plt.title("LUMO")
        plt.show()
        plt.close()

    return train_dataset, val_dataset, test_dataset


def plot_errors_training(model, data_loader, target, norm_targets, seed, epoch, device, path=".", big_nds=False):
    """
    This will save some plots of the losses and scatterplots of ND categories (size, surface, doping)
    and the target predictions. Plots will be saved under path
    :param model:
    :param data_loader:
    :param target:
    :param norm_targets:
    :param seed:
    :param epoch:
    :param device:
    :param path:
    :param big_nds:
    :return:
    """
    truths, preds, labels = [], [], []
    model.eval()
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        truth = data.y
        label = data.descriptor
        # rescale data
        if norm_targets and target == "HOMO":
            output = (output * 10.7) - 10.6
            truth = (truth * 10.7) - 10.6
        elif norm_targets and target == "LUMO":
            output = (output * 7.2) - 5.6
            truth = (truth * 7.2) - 5.6
        truths = truths + truth.tolist()
        preds = preds + output.tolist()
        labels = labels + label.tolist()

    # a reminder on how the ND descriptor is built:
    # descriptor = descriptor_dict[row["ND"]]*10000 + descriptor_dict[row["Surface"]]*1000 + int(row["Maximal_coverage"])*100 + 
    #              descriptor_dict[row["Dopant1"]]*10 + descriptor_dict[row["Dopant2"]]*1
    labels_nds = [int(str(lab)[:-4]) for lab in labels]
    if big_nds: labels_nds = [label - 17 for label in labels_nds]
    labels_surf = [int(str(lab)[-4]) for lab in labels]
    labels_d1 = [int(str(lab)[-2]) for lab in labels]
    labels_d2 = [int(str(lab)[-1]) for lab in labels]

    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    # get_ipython().run_line_magic('matplotlib', 'inline')
    # !
    plt.rc('font', size=15)

    # generic plot
    plt.figure(figsize=(8, 8))
    plt.scatter(preds, truths, s=15, label="Test Set Predictions")
    if target == "HOMO":
        plt.plot([-11, 1], [-11, 1], linewidth=0.7, color="red")
        if (np.array(preds) < -11).any() or (np.array(preds) > 1).any():
            print("Warning: prediction outside figure frame in " + target + " error plot")
    if target == "LUMO":
        plt.plot([-6, 2], [-6, 2], linewidth=0.7, color="red")
        if (np.array(preds) < -6).any() or (np.array(preds) > 2).any():
            print("Warning: prediction outside figure frame in " + target + " error plot")
    plt.xlabel("Predicted {} Energy / eV".format(target))
    plt.ylabel("Truth / eV")
    plt.legend()
    plt.savefig(path + "/errorplot_" + target + "_" + "ep" + str(epoch) + "_" + str(seed) + ".png")
    # plt.show() #!
    plt.close()

    # nd-size color coding
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = plt.scatter(preds, truths, s=15, c=labels_nds, cmap="plasma", label="Test Set Predictions")
    if target == "HOMO":
        plt.plot([-11, 1], [-11, 1], linewidth=0.7, color="red")
        if (np.array(preds) < -11).any() or (np.array(preds) > 1).any():
            print("Warning: prediction outside figure frame in " + target + " error plot")
    if target == "LUMO":
        plt.plot([-6, 2], [-6, 2], linewidth=0.7, color="red")
        if (np.array(preds) < -6).any() or (np.array(preds) > 2).any():
            print("Warning: prediction outside figure frame in " + target + " error plot")
    plt.xlabel("Predicted {} Energy / eV".format(target))
    plt.ylabel("Truth / eV")
    if not big_nds:
        cbar = plt.colorbar(ticks=[1, 5, 9, 13, 17])
        cbar.ax.set_yticklabels(['C$_{10}$H$_{16}$', 'C$_{22}$H$_{28}$', 'C$_{26}$H$_{32}$',
                                 'C$_{68}$H$_{64}$', 'C$_{109}$H$_{80}$'])
    else:
        cbar = plt.colorbar(ticks=[1, 2, 3, 4, 5, 6])
        cbar.ax.set_yticklabels(['C$_{133}$H$_{100}$', 'C$_{145}$H$_{108}$', 'C$_{147}$H$_{100}$',
                                 'C$_{178}$H$_{126}$', 'C$_{231}$H$_{146}$', 'C$_{281}$H$_{172}$'])
    plt.savefig(path + "/errorplot_" + target + "_" + "ep" + str(epoch) + "_" + str(seed) + "_nds.png")
    # plt.show() #!
    plt.close()

    # surface color coding
    plt.figure(figsize=(8, 8))
    surf_cmap = ListedColormap(["darkgray", "blue", "red", "deepskyblue"])
    plt.scatter(preds, truths, s=15, c=labels_surf, cmap=surf_cmap, label="Test Set Predictions")
    if target == "HOMO":
        plt.plot([-11, 1], [-11, 1], linewidth=0.7, color="red")
        if (np.array(preds) < -11).any() or (np.array(preds) > 1).any():
            print("Warning: prediction outside figure frame in " + target + " error plot")
    if target == "LUMO":
        plt.plot([-6, 2], [-6, 2], linewidth=0.7, color="red")
        if (np.array(preds) < -6).any() or (np.array(preds) > 2).any():
            print("Warning: prediction outside figure frame in " + target + " error plot")
    plt.xlabel("Predicted {} Energy / eV".format(target))
    plt.ylabel("Truth / eV")

    legend_elements = [
        Line2D([0], [0], marker='o', color='white', markerfacecolor='darkgray', label='H', markersize=15),
        Line2D([0], [0], marker='o', color='white', markerfacecolor='blue', label='NH$_2$', markersize=15),
        Line2D([0], [0], marker='o', color='white', markerfacecolor='red', label='OH', markersize=15),
        Line2D([0], [0], marker='o', color='white', markerfacecolor='deepskyblue', label='F', markersize=15)]
    plt.legend(handles=legend_elements)
    plt.savefig(path + "/errorplot_" + target + "_" + "ep" + str(epoch) + "_" + str(seed) + "_surf.png")
    # plt.show() #!
    plt.close()

    # dopant coding
    plt.figure(figsize=(8, 8))
    # "C":0, "B":1, "N":2, "Si":3, "P":4
    dopants_col_dict = {0: "black", 1: "hotpink", 2: "blue", 3: "olive", 4: "darkorange"}
    dopants_marker_dict = {0: "o", 1: "v", 2: "^", 3: "s", 4: "D"}
    for pr, tr, d1, d2 in zip(preds, truths, labels_d1, labels_d2):
        plt.scatter(pr, tr, s=15, c=dopants_col_dict[d1], marker=dopants_marker_dict[d2])
    if target == "HOMO":
        plt.plot([-11, 1], [-11, 1], linewidth=0.7, color="red")
        if (np.array(preds) < -11).any() or (np.array(preds) > 1).any():
            print("Warning: prediction outside figure frame in " + target + " error plot")
    if target == "LUMO":
        plt.plot([-6, 2], [-6, 2], linewidth=0.7, color="red")
        if (np.array(preds) < -6).any() or (np.array(preds) > 2).any():
            print("Warning: prediction outside figure frame in " + target + " error plot")
    plt.xlabel("Predicted {} Energy / eV".format(target))
    plt.ylabel("Truth / eV")

    legend_elements = [Line2D([0], [0], marker='o', color='white', markerfacecolor='gray', label='C', markersize=15),
                       Line2D([0], [0], marker='v', color='white', markerfacecolor='gray', label='B', markersize=15),
                       Line2D([0], [0], marker='^', color='white', markerfacecolor='gray', label='N', markersize=15),
                       Line2D([0], [0], marker='s', color='white', markerfacecolor='gray', label='Si', markersize=15),
                       Line2D([0], [0], marker='D', color='white', markerfacecolor='gray', label='P', markersize=15),
                       Line2D([0], [0], marker=9, color='white', markerfacecolor='black', label='C', markersize=15),
                       Line2D([0], [0], marker=9, color='white', markerfacecolor='hotpink', label='B', markersize=15),
                       Line2D([0], [0], marker=9, color='white', markerfacecolor='blue', label='N', markersize=15),
                       Line2D([0], [0], marker=9, color='white', markerfacecolor='olive', label='Si', markersize=15),
                       Line2D([0], [0], marker=9, color='white', markerfacecolor='darkorange', label='P',
                              markersize=15)]
    plt.legend(handles=legend_elements)
    plt.savefig(path + "/errorplot_" + target + "_" + "ep" + str(epoch) + "_" + str(seed) + "_dopants.png")
    # plt.show() #!
    plt.close()


def plot_graph_radius_node_degrees(probe_radii=[2.1, 2.3, 2.5, 2.75, 3, 4, 5, 7, 10]):
    # some values on graph radius vs. average node degree in the graphs
    # takes several minutes to execute
    # just a convenience function
    import matplotlib.pyplot as plt

    avg_node_degrees = []

    for test_radius in probe_radii:
        if dataset_type == 'nd' or dataset_type == 'big nd':
            csv_path = "./data_sorted.csv"
        elif dataset_type == 'Oe62k':
            csv_path = 'df_62k_original.json'
        dataset = get_any_dataset(dataset_type=dataset_type, csv_path=csv_path, target="HOMO", graph_radius=test_radius,
                                  testing=False, print_info=True, fingerprint=None)
        avg_node_deg = 0
        for i in range(len(dataset)):
            datapoint = dataset[i]
            avg_node_deg += datapoint.num_edges / datapoint.num_nodes
        avg_node_deg = round(avg_node_deg / len(dataset), 4)
        avg_node_degrees.append(avg_node_deg)
        print("Radius", test_radius, ": Global average node degree", avg_node_deg)

    plt.plot(probe_radii, avg_node_degrees)
    plt.xlabel("graph radius")
    plt.ylabel("node degree")
    plt.show()
    plt.close()


# ## Standard Network Training


class Net(torch.nn.Module):
    def __init__(self, device, p1, p2, p_dropout, n_layers, n_preprocessing, max_radius, nbr_node_features=14):
        """
        This class creates the Network used for training.
        :param device: cuda, cpu, ?
        :param p1: The number of hidden neurons for the 3 layer NN used for edge conditioned convolution
        :param p2: hidden dimension of the node states
        :param p_dropout: dropout rate after the message passing phase for the final prediction NN
        :param n_layers: Message passing layers
        :param n_preprocessing: number of layers in the preprocessing NN
        :param max_radius: max_radius of the radius graph
        :param nbr_node_features: size of the node features
        """
        super(Net, self).__init__()  # "Add" aggregation (Step 5).

        self.device = device
        self.p_dropout = p_dropout
        self.n_layers = n_layers
        self.max_radius = max_radius

        nbr_edge_features = 2
        p3 = p2 * 2

        # preprocessing Network
        pre_processing_list = []
        pre_process_gradient = (nbr_node_features - p2) / (n_preprocessing)
        if n_preprocessing > 1:
            for i in range(1, n_preprocessing + 1):
                feature_size = round(pre_process_gradient * (n_preprocessing - (i)) + p2)
                feature_size_previous = round(pre_process_gradient * (n_preprocessing - (i - 1)) + p2)
                print('Now', feature_size, 'Last', feature_size_previous)
                if i == 1:
                    pre_processing_list.append(torch.nn.Linear(nbr_node_features, feature_size, bias=True))
                    pre_processing_list.append(BN(feature_size))
                elif i < n_preprocessing:
                    pre_processing_list.append(torch.nn.Linear(feature_size_previous, feature_size, bias=True))
                    pre_processing_list.append(BN(feature_size))
                elif i == n_preprocessing:
                    pre_processing_list.append(torch.nn.Linear(feature_size_previous, p2, bias=True))
                    pre_processing_list.append(BN(p2))
                pre_processing_list.append(LeakyReLU())
        elif n_preprocessing == 1:
            pre_processing_list.append(torch.nn.Linear(nbr_node_features, p2, bias=False))
            pre_processing_list.append(BN(p2))
            pre_processing_list.append(LeakyReLU())

        print(pre_processing_list)
        print(*pre_processing_list)
        self.preprocess_nn = Seq(*pre_processing_list)

        # Edge conditioned Convolution NN
        nn = Seq(Linear(nbr_edge_features, p1, bias=False), BN(p1), LeakyReLU(), Linear(p1, p2 * p2, bias=False),
                 BN(p2 * p2))

        # message passing step things
        self.conv = NNConv(p2, p2, nn, aggr='mean')
        self.set2set = Set2Set(p2, processing_steps=3)
        self.gru = GRU(p2, p2)

        # Readout NN
        self.lin1 = torch.nn.Linear(p3, round(p3 / 2))
        self.lin2 = torch.nn.Linear(round(p3 / 2), round(p3 / 4))
        self.lin_final = torch.nn.Linear(round(p3 / 4), 1)

    def forward(self, data):
        y = None

        x, edge_index, edge_attr_1, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr_2 = self.max_radius - edge_attr_1
        edge_attr = torch.cat((edge_attr_1 / self.max_radius, edge_attr_2 / self.max_radius), 1)

        # preprocessing
        out = self.preprocess_nn(x)
        h = out.unsqueeze(0)

        # Message passing
        for i in range(self.n_layers):
            m = F.leaky_relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # Readout
        y = self.set2set(out, batch)
        y = F.dropout(y, p=self.p_dropout, training=self.training)
        y = F.leaky_relu(self.lin1(y))
        y = F.leaky_relu(self.lin2(y))
        y = self.lin_final(y)
        y = y.squeeze(-1)
        return y


def train(model, optimizer, train_loader, device):
    # train
    model.train()
    criterion = torch.nn.MSELoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        del loss, output, label
    return model


def test(model, data_loader, crit, device, target, norm_targets):
    assert (crit in ["mse", "mae"]), "Unknown error criterion."
    criterion = None
    if crit == "mse":
        criterion = torch.nn.MSELoss()
    elif crit == "mae":
        criterion = torch.nn.L1Loss()

    model.eval()
    loss_all = 0
    n_data = 0
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        label = data.y.to(device)
        if norm_targets and target == "HOMO":
            output = output * 10.7
            label = label * 10.7
        elif norm_targets and target == "LUMO":
            output = output * 7.2
            label = label * 7.2
        loss = criterion(output, label)
        loss_all += data.num_graphs * loss.item()
        n_data += data.num_graphs
        del loss, output, label

    return loss_all / n_data


def train_new_network(target, norm_targets, hyperparameters, logger, fingerprint=None, fingerprint_dict=None,
                      test_big_nds=False, new_dataset=True, stratified_split=True, n_epochs=None, lr_params=None,
                      plot_losses=False, seed=100, testing=False):
    """
    :param target: 'HOMO- LUMO
    :param norm_targets:
    :param hyperparameters: list of network hyperparameters, see below
    :param logger: dataset logger to get dataset from
    :param fingerprint: soap, soap_pca, None
    :param fingerprint_dict: soap parameters
    :param test_big_nds: whether to use the larger NDs for testing
    :param new_dataset: whether to override existing datasets
    :param stratified_split: stratified split for Nd5k?
    :param n_epochs: learning epochs
    :param lr_params:
    :param plot_losses: create plots during training
    :param seed: seed used for training
    :param testing: whether this is just a training run
    :return:
    """

    # sanity check
    assert (len(hyperparameters) == 8), "Incorrect number of hyperparameters given. Expected 7, got {}".format(
        len(hyperparameters))

    # get hyperparameters
    graph_radius, batchsize, learning_rate, n_preprocess, p1, p2, p_dropout, n_layers = hyperparameters[0], \
                                                                                        hyperparameters[1], \
                                                                                        hyperparameters[2], \
                                                                                        hyperparameters[3], \
                                                                                        hyperparameters[4], \
                                                                                        hyperparameters[5], \
                                                                                        hyperparameters[6], \
                                                                                        hyperparameters[7]
    print("Current hyperparameters:", hyperparameters)

    # initialize methods for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == "cuda": torch.cuda.empty_cache()
    # print("using", device)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    ### ND-data training ###
    dataset, dataset_path = logger.get_dataset(dataset_type=dataset_type, target=target, radius=graph_radius,
                                               normalize_targets=norm_targets,
                                               testing=testing, fingerprint=fingerprint,
                                               fingerprint_dict=fingerprint_dict, reload=new_dataset)
    pca = (fingerprint == "soap_pca")

    # split into train, valildation, test dataset
    dataset = dataset.shuffle(seed)
    if stratified_split:
        train_dataset, val_dataset, test_dataset = stratified_split_train_val_test(dataset, target, norm_targets)
    else:
        # train 70 %, val 15 %, test 15 %
        cut = int(np.floor(dataset.len() * 0.15))
        train_dataset = dataset[:len(dataset) - (2 * cut)]
        val_dataset = dataset[len(dataset) - (2 * cut):len(dataset) - cut]
        test_dataset = dataset[len(dataset) - cut:]

    if test_big_nds:
        train_dataset = train_dataset + test_dataset
        test_dataset, test_path = logger.get_dataset(dataset_type='big nd', target=target, radius=graph_radius,
                                                     normalize_targets=norm_targets,
                                                     testing=False, fingerprint=fingerprint,
                                                     fingerprint_dict=fingerprint_dict, reload=new_dataset)

    print("Dataset sizes: {} Train, {} Val, {} Test".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    if pca:
        if 'logfile.pkl' in os.listdir():
            logs = pd.read_pickle('./logfile.pkl')
            print('log in listdir')
            if (logs == pd.Series(
                    [seed, dataset_type, target, graph_radius, norm_targets, testing, fingerprint, fingerprint_dict,
                     test_big_nds])).all():
                print('loading buffered pca Dataset')
                train_dataset = MyDatasetInMemory(root=dataset_path, name='train')
                test_dataset = MyDatasetInMemory(root=dataset_path, name='test')
                val_dataset = MyDatasetInMemory(root=dataset_path, name='val')

        if train_dataset.__getitem__(0).x.shape[-1] != pca_components:
            print('PCA!')
            transform = get_pca_transform(train_dataset)

            print('finished')
            min_, max_ = get_min_max_temp_pca_func(dataset, transform.transform)

            datasetTransform = pcaTranformDataset(transform, min_, max_)
            print('transforming!')
            train_dataset = train_dataset.transformDataset(datasetTransform, name='train')
            test_dataset = test_dataset.transformDataset(datasetTransform, name='test')
            val_dataset = val_dataset.transformDataset(datasetTransform, name='val')
            pd.Series([seed, dataset_type, target, graph_radius, norm_targets, testing, fingerprint, fingerprint_dict,
                       test_big_nds]).to_pickle('./logfile.pkl')

    node_feature_size = train_dataset.__getitem__(0).x.shape[-1]

    print(train_dataset.__getitem__(0))

    model = Net(device, p1, p2, p_dropout, n_layers, n_preprocess, graph_radius,
                nbr_node_features=node_feature_size).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batchsize)
    val_loader = DataLoader(val_dataset, batch_size=batchsize)
    test_loader = DataLoader(test_dataset, batch_size=batchsize)

    # print(type(train_loader.get(0)))

    train_losses_mse, train_losses_mae, val_losses_mse, val_losses_mae, test_losses_mse, test_losses_mae = [], [], [], [], [], []
    best_model_parameters, best_model_hyperparameters = None, None

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_params['lr_decay'],
                                                           patience=lr_params['decay_patience_epochs'])

    for epoch in range(1, n_epochs * 2):
        model = train(model, optimizer, train_loader, device)

        # compute losses
        loss_train_mse = test(model, train_loader, "mse", device, target, norm_targets)
        loss_train_mae = test(model, train_loader, "mae", device, target, norm_targets)
        loss_val_mse = test(model, val_loader, "mse", device, target, norm_targets)
        loss_val_mae = test(model, val_loader, "mae", device, target, norm_targets)
        loss_test_mse = test(model, test_loader, "mse", device, target, norm_targets)
        loss_test_mae = test(model, test_loader, "mae", device, target, norm_targets)
        train_losses_mse.append(loss_train_mse)
        train_losses_mae.append(loss_train_mae)
        val_losses_mse.append(loss_val_mse)
        val_losses_mae.append(loss_val_mae)
        test_losses_mse.append(loss_test_mse)
        test_losses_mae.append(loss_test_mae)

        scheduler.step(loss_val_mse)

        # print loss info
        message = "Epoch {}: Train loss: {} MSE, {} MAE / Val. loss: {} MSE, {} MAE".format(epoch,
                                                                                            round(loss_train_mse, 4),
                                                                                            round(loss_train_mae, 4),
                                                                                            round(loss_val_mse, 4),
                                                                                            round(loss_val_mae, 4))
        if loss_val_mse < loss_train_mse or loss_val_mae < loss_train_mae:
            message = message + " / "
            if loss_val_mse < loss_train_mse:
                message = message + "MSE "
            if loss_val_mae < loss_train_mae:
                message = message + "MAE "
            message = message + "Loss Warning"
        print(message)

        # store parameters and hyperparameters of the best model
        if epoch == 1:
            best_model_parameters = list(model.parameters())
            best_model_hyperparameters = list(hyperparameters)
        elif loss_val_mse < min(val_losses_mse[:-1]):
            # save network parameters
            best_model_parameters = list(model.parameters())
            best_model_hyperparameters = list(hyperparameters)

        if plot_losses and epoch == 30:
            plot_errors_training(model, test_loader, target, norm_targets, seed, epoch, device)

        # look for stop signs: either max number of epochs or increasing loss
        if epoch == n_epochs:
            break

        # early stopping
        temp_val_losses = np.asarray(val_losses_mse)
        if epoch > lr_params['early_stopping_patience']:
            if temp_val_losses[-lr_params['early_stopping_patience']:].min() > (temp_val_losses.min() + 1e-4):
                break

        print(optimizer.state_dict()['param_groups'][0]['lr'])

    print("Minimum validation MSE loss after epoch {}: {}".format(val_losses_mse.index(min(val_losses_mse)) + 1,
                                                                  round(min(val_losses_mse), 4)))

    test_loss_mse = test_losses_mse[val_losses_mse.index(min(val_losses_mse))]
    test_loss_mae = test_losses_mae[val_losses_mse.index(min(val_losses_mse))]
    print("Test loss of minimum validation MSE loss network: {} MSE, {} MAE".format(round(test_loss_mse, 4),
                                                                                    round(test_loss_mae, 4)))

    train_loss_mse = train_losses_mse[val_losses_mse.index(min(val_losses_mse))]
    train_loss_mae = train_losses_mae[val_losses_mse.index(min(val_losses_mse))]

    losses = [round(train_loss_mse, 4), round(train_loss_mae, 4), round(min(val_losses_mse), 4),
              round(min(val_losses_mae), 4), round(test_loss_mse, 4), round(test_loss_mae, 4)]

    if plot_losses:
        import matplotlib.pyplot as plt
        # get_ipython().run_line_magic('matplotlib', 'inline')
        # !
        plt.rc('font', size=15)

        plt.figure(figsize=(7.5, 5))
        plt.plot(np.linspace(1, epoch, epoch, dtype=int), train_losses_mae, label="Training Loss")
        plt.plot(np.linspace(1, epoch, epoch, dtype=int), val_losses_mae, label="Validation Loss")
        plt.ylabel("MAE / eV")
        plt.xlabel("Epoch")
        plt.ylim(0, 0.7)  # max(train_losses_mae+val_losses_mae))
        plt.xticks(np.linspace(1, epoch, num=5, dtype=int))
        plt.legend()
        plt.savefig("losses_" + target + "_" + str(seed) + ".png")
        plt.close()

        plot_errors_training(model, test_loader, target, norm_targets, seed, epoch, device)

    del dataset, model, optimizer, train_loader, val_loader, test_loader
    gc.collect()

    return losses, best_model_parameters, best_model_hyperparameters


def optimize_hyperparameters(name, target, logger, norm_targets, fingerprint_range, fingerprint_list,
                             graph_radius_range, batchsize_range, learning_rate_range, n_preprocess_range, p1_range,
                             p2_range, p_dropout_range, n_layers_range, test_big_nds=False, stratified_split=True,
                             n_epochs=None, n_seeds=1, testing=False, new_dataset=False, lr_params=None,
                             manual_seeds=None):
    """
    This will perform a grid based search based on the lists passed to the function
    Oh Boi, dont touch this, most of these parameters are what they say they are, just see in the funciton below what these do
    :param name:
    :param target:
    :param logger:
    :param norm_targets:
    :param fingerprint_range:
    :param fingerprint_list:
    :param graph_radius_range:
    :param batchsize_range:
    :param learning_rate_range:
    :param n_preprocess_range:
    :param p1_range:
    :param p2_range:
    :param p_dropout_range:
    :param n_layers_range:
    :param test_big_nds:
    :param stratified_split:
    :param n_epochs:
    :param n_seeds:
    :param testing:
    :param new_dataset:
    :param lr_params:
    :param manual_seeds:
    :return:
    """
    if manual_seeds is None:
        seeds = [random.randint(1, 10000) for i in range(n_seeds)]
    else:
        seeds = manual_seeds
    print("seeds:", seeds)
    hyperparameters_losses = []
    best_val_mse_loss = 1000
    best_parameters, best_hyperparameters = [], []
    csv_name = f"hyp_loss_{target}_{name}_{pd.to_datetime(datetime.datetime.now(), dayfirst=True)}.pkl".format(target,
                                                                                                               name)

    for fingerprint in fingerprint_range:
        if fingerprint == None:
            temp_name = name + 'NONE'
            for graph_radius in graph_radius_range:
                for batchsize in batchsize_range:
                    for learning_rate in learning_rate_range:
                        for p1 in p1_range:
                            for p2 in p2_range:
                                for p_dropout in p_dropout_range:
                                    for n_layers in n_layers_range:
                                        for n_preprocess in n_preprocess_range:
                                            losses = [0, 0, 0, 0, 0, 0]
                                            for seed in seeds:
                                                hyperparameters = [graph_radius, batchsize, learning_rate, n_preprocess,
                                                                   p1, p2, p_dropout, n_layers]
                                                losses_tmp, best_model_parameters, best_model_hyperparameters = train_new_network(
                                                    target, norm_targets, hyperparameters, logger=logger,
                                                    fingerprint=None, fingerprint_dict=None, test_big_nds=test_big_nds,
                                                    new_dataset=new_dataset, stratified_split=stratified_split,
                                                    n_epochs=n_epochs, plot_losses=False, seed=seed, testing=testing,
                                                    lr_params=lr_params)
                                                if losses_tmp[2] < best_val_mse_loss:
                                                    best_parameters = list(best_model_parameters)
                                                    best_hyperparameters = list(best_model_hyperparameters)
                                                    best_val_mse_loss = losses_tmp[2]
                                                losses = [losses[i] + losses_tmp[i] for i in range(len(losses))]
                                            losses = [losses[i] / n_seeds for i in range(len(losses))]
                                            hyperparameters_losses.append(
                                                [n_seeds] + [fingerprint, None] + hyperparameters + losses)
                                            hyp_loss_df = pd.DataFrame(hyperparameters_losses,
                                                                       columns=['n_seeds', 'fingerprint',
                                                                                'fingerprint_dict', 'graph_radius',
                                                                                'batchsize', 'learning_rate',
                                                                                'n_preprocess', 'p1', 'p2', 'p_dropout',
                                                                                'n_layers',
                                                                                'train_loss_mse', 'train_loss_mae',
                                                                                'val_loss_mse', 'val_loss_mae',
                                                                                'test_loss_mse', 'test_loss_mae'])
                                            hyp_loss_df.to_pickle(csv_name)



        elif fingerprint == 'soap' or fingerprint == 'soap_pca':
            soap_cutoff_range, max_n_range, max_l_range, rbf_range, weighting_range, sigma_range = fingerprint_list
            # if fingerprint == 'soap_pca':
            pca = (fingerprint == 'soap_pca')
            for weighting in weighting_range:
                for nmax in max_n_range:
                    for lmax in max_l_range:
                        if lmax is None:
                            lmax = nmax
                        for rbf in rbf_range:
                            for rcut in soap_cutoff_range:
                                if weighting is not None:
                                    if weighting['r0'] is not None:
                                        rcut = None
                                for sigma in sigma_range:
                                    if lmax > nmax:
                                        continue
                                    temp_name = name + '_soap_r_' + str(rcut) + '_n_' + str(nmax) + '_l_' + str(
                                        lmax) + '_rbf_' + rbf + '_weigh_' + str(weighting != None) + '_sig_' + str(
                                        sigma)
                                    print(name)
                                    fingerprint_dict = {'rcut': rcut,
                                                        'nmax': nmax,
                                                        'lmax': lmax,
                                                        'rbf': rbf,
                                                        'weighting': weighting,
                                                        'sigma': sigma}

                                    for graph_radius in graph_radius_range:
                                        for batchsize in batchsize_range:
                                            for learning_rate in learning_rate_range:
                                                for p1 in p1_range:
                                                    for p2 in p2_range:
                                                        for p_dropout in p_dropout_range:
                                                            for n_layers in n_layers_range:
                                                                for n_preprocess in n_preprocess_range:
                                                                    losses = [0, 0, 0, 0, 0, 0]
                                                                    for seed in seeds:
                                                                        print(fingerprint_dict)
                                                                        hyperparameters = [graph_radius, batchsize,
                                                                                           learning_rate, n_preprocess,
                                                                                           p1, p2, p_dropout, n_layers]
                                                                        losses_tmp, best_model_parameters, best_model_hyperparameters = train_new_network(
                                                                            target, norm_targets, hyperparameters,
                                                                            logger=logger, fingerprint=fingerprint,
                                                                            fingerprint_dict=fingerprint_dict,
                                                                            test_big_nds=test_big_nds,
                                                                            new_dataset=new_dataset,
                                                                            stratified_split=stratified_split,
                                                                            n_epochs=n_epochs, plot_losses=False,
                                                                            seed=seed, testing=testing,
                                                                            lr_params=lr_params, pca=pca)
                                                                        if losses_tmp[2] < best_val_mse_loss:
                                                                            best_parameters = list(
                                                                                best_model_parameters)
                                                                            best_hyperparameters = list(
                                                                                best_model_hyperparameters)
                                                                            best_val_mse_loss = losses_tmp[2]
                                                                        losses = [losses[i] + losses_tmp[i] for i in
                                                                                  range(len(losses))]
                                                                    losses = [losses[i] / n_seeds for i in
                                                                              range(len(losses))]
                                                                    hyperparameters_losses.append(
                                                                        [n_seeds] + [fingerprint,
                                                                                     fingerprint_dict] + hyperparameters + losses)
                                                                    hyp_loss_df = pd.DataFrame(hyperparameters_losses,
                                                                                               columns=['n_seeds',
                                                                                                        'fingerprint',
                                                                                                        'fingerprint_dict',
                                                                                                        'graph_radius',
                                                                                                        'batchsize',
                                                                                                        'learning_rate',
                                                                                                        'n_preprocess',
                                                                                                        'p1', 'p2',
                                                                                                        'p_dropout',
                                                                                                        'n_layers',
                                                                                                        'train_loss_mse',
                                                                                                        'train_loss_mae',
                                                                                                        'val_loss_mse',
                                                                                                        'val_loss_mae',
                                                                                                        'test_loss_mse',
                                                                                                        'test_loss_mae'])
                                                                    hyp_loss_df.to_pickle(csv_name)

    return best_parameters, best_hyperparameters


##### Controlling Code ##################################################################################################

testing = True  # just use a small fraction of the datasets for testing
target = "HOMO"  # can be set to either "HOMO" or "LUMO"
norm_targets = True  # if True, HOMO and LUMO values are normalized to (0,1)
reload = False  # if True existing Datasets will be deleted and recalculated from scratch
logger = DatasetLogger('./Datasets')  # The logger to create the datasets
manual_seeds = [42]  # Of course, what else  (if empty, use random seeds)
dataset_type = 'big nd'  # nd, big nd or oe62k

##Fingerprints:
fingerprint_range = ['soap_pca']  # None -> classical, soap or soap_pca
pca_components = 200  # kept pca components

### Hyperparameters ###

## Fingerprint Params:
# soap:
# rcut : Cutoff radius (in A)
# nmax : Radial basis max degree
# lmax : Angular basis max degree ! l <= n
# rbf : radial basis set: either gto or poly #standard: gto
# weighting : This is rather involved, defining a cutoff function, look up documentation for this
# sigma : std of the smearing gaussians used

soap_cutoff_range = [None]  # either hard cutoff (here) or specified via weighting_range

max_n_range = [2]

max_l_range = [None]  # if None lmax will be set to nmax

rbf_range = ['gto']

weighting_range = [{'function': 'poly', 'c': 1, 'm': 1, 'r0': 3.5}]

sigma_range = [0.5]

# Create one Single list to contain all fingerprint params
fingerprint_list = [soap_cutoff_range, max_n_range, max_l_range, rbf_range, weighting_range, sigma_range]

## Network Parameters

graph_radius_range = [3.0]  # The Radius of the molecular Graph

batchsize_range = [16]

learning_rate_range = [0.001]

n_preprocess_range = [3]

p1_range = [64]  # Hidden Neurons of Network used for edge conditioned convol.

p2_range = [96]  # The dimension of the hidden node states

p_dropout_range = [0.3]  # Dropout Rate after the Message Passing Phase

n_layers_range = [4]  # Number of message passing steps

# ND5k parameters
stratified_split = False
test_big_nds = False  # Whether to use a distinct subset of larger NDs for testing

n_seeds = 1  # The number of networks to train
n_epochs = 150  # Epochs to train for

lr_params = {'decay_patience_epochs': 1000,  # Lr Scheduling Params and early stopping
             'lr_decay': 0.7,  # Decay Scheduler
             'early_stopping_patience': 4000}

name = "default_normalized_edgefeatures_newfeaturenorm"

# Lets Goo, run everythingg

best_parameters, best_hyperparameters = optimize_hyperparameters(name, target, logger, norm_targets, fingerprint_range,
                                                                 fingerprint_list, graph_radius_range, batchsize_range,
                                                                 learning_rate_range, n_preprocess_range, p1_range,
                                                                 p2_range, p_dropout_range, n_layers_range,
                                                                 test_big_nds=test_big_nds,
                                                                 stratified_split=stratified_split, n_epochs=n_epochs,
                                                                 n_seeds=n_seeds, testing=testing, lr_params=lr_params,
                                                                 manual_seeds=manual_seeds, new_dataset=reload)
