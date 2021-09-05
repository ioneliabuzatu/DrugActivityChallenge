import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import ADASYN


# Turn molecules in smiles format into Morgan fingerprints
def smiles_to_morgan(data: np.array, fp_length=1024):
    desc_mtx = np.zeros((len(data), fp_length)) * np.nan

    # Get Morgan fingerprints
    for i, smiles in enumerate(data):
        mol = Chem.MolFromSmiles(smiles)
        desc_mtx[i] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=fp_length)

    return desc_mtx


data_train = pd.read_csv('data_train.csv', index_col=0)
data_test = pd.read_csv('smiles_test.csv')

# Define train and test sets by getting Morgan fingerprints
X = smiles_to_morgan(np.array(data_train['smiles']), fp_length=2048)
X_test = smiles_to_morgan(np.array(data_test['smiles']), fp_length=2048)

print(X.shape, X_test.shape)

# Create task-only dataframe
tasks = data_train.drop('smiles', axis=1)

# Get labels from tasks dataframe
y = [np.array(tasks[col]) for col in tasks]

task_filters = [np.where(label!=0) for label in y]

seed = 42
roc_auc_scores = []
predictions_testset = {}

for i, mask in enumerate(task_filters):
    # Train / Validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X[mask], y[i][mask], test_size=0.1, random_state=seed)

    # Resample training data
    X_train, y_train = ADASYN().fit_resample(X_train, y_train)

    # Define and train model
    clf = RandomForestClassifier(n_estimators=20, random_state=seed)
    clf.fit(X_train, y_train)

    # Bagging regressor based on model
    clf_bagg = BaggingRegressor(base_estimator=clf, n_estimators=100, n_jobs=-1, random_state=seed)
    clf_bagg.fit(X_train, y_train)

    # Validation
    y_hat_proba = clf_bagg.predict(X_valid)
    roc_auc_scores.append(roc_auc_score(y_valid, y_hat_proba))
    print('ROC AUC score task ' + str(i + 1) + ':', roc_auc_scores[i])

    # Test
    predictions_testset['task' + str(i + 1)] = clf.predict(X_test)

# Print mean score
print('Mean ROC AUC score:', np.mean(roc_auc_scores))

# Dump CSV file
pd.DataFrame(predictions_testset).to_csv('submission_rf_bagg.csv')
