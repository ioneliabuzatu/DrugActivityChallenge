#!/usr/bin/env python

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def preprocess_smiles_data(filepath_data_train=None, filepath_smiles_test=None):

    def smiles_to_fingerprints(data: np.array, fp_length=1024):
        fingerprints = np.zeros((len(data), fp_length)) * np.nan
        for i, smile in enumerate(data):
            mol = Chem.MolFromSmiles(smile)
            fingerprints[i] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=fp_length)
        return fingerprints

    data_train = pd.read_csv(filepath_data_train, index_col=0)
    data_test = pd.read_csv(filepath_smiles_test)

    x_train_set = smiles_to_fingerprints(np.array(data_train['smiles']), fp_length=2048)
    x_test_set = smiles_to_fingerprints(np.array(data_test['smiles']), fp_length=2048)

    tasks = data_train.drop('smiles', axis=1)
    y = [np.array(tasks[col]) for col in tasks]
    masks = [np.where(label != 0) for label in y]

    return x_train_set, y, x_test_set, masks

preprocess_smiles_data("data/data_train.csv", "data/smiles_test.csv")
