#!/usr/bin/env python

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors


def generate_train_data_with_generators_or_fingerprints(data_train_filepath, train=True):
    MolDescCalc, names = get_all_database_descriptors()

    data = pd.read_csv(data_train_filepath, index_col=0)

    fp_length = 102
    fingerprints_data_train = np.zeros((len(data), fp_length)) * np.nan

    descriptors_data_train = np.zeros((len(data), len(names))) * np.nan

    for i, smile in enumerate(data["smiles"]):
        mol = Chem.MolFromSmiles(smile)

        descriptors_data_train[i] = MolDescCalc.CalcDescriptors(mol)

        fingerprints_data_train[i] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=fp_length)

    data_train_with_descriptors = pd.concat([data, pd.DataFrame(descriptors_data_train)], axis=1)
    fingerprints_data_train_df = pd.concat([data, pd.DataFrame(fingerprints_data_train)], axis=1)

    if train:
        save_csv(data_train_with_descriptors, "./data/data_train_descriptors.csv")
        save_csv(fingerprints_data_train_df, "./data/data_train_fingerprints.csv")
    else:
        save_csv(data_train_with_descriptors, "./data/data_test_descriptors.csv")
        save_csv(fingerprints_data_train_df, "./data/data_test_fingerprints.csv")


def get_all_database_descriptors():
    descriptor_list = Chem.Descriptors._descList
    names = [desc[0] for desc in descriptor_list]
    MolDescCalc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    return MolDescCalc, names


def save_csv(data_frame, filepath_to_save):
    data_frame.to_csv(filepath_to_save)


if __name__ == "__main__":
    generate_train_data_with_generators_or_fingerprints("./smiles_test.csv", train=False)
