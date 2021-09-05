import logging
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def oversampling(x,y):
    x, y = ADASYN(n_jobs=-1, sampling_strategy='not minority').fit_resample(x, y)
    return x, y


def predict_drug_activity_with_svr(x, y, x_test, masks_tasks, params, seed=123):
    roc_auc_scores = []
    submissions = {}

    for i, mask in enumerate(masks_tasks):
        x_train, x_valid, y_train, y_valid = train_test_split(x[mask], y[i][mask], test_size=0.1, random_state=seed)

        x_train, y_train = oversampling(x_train, y_train)

        svr_model = SVR(**params[i][0])
        svr_model.fit(x_train, y_train)

        model_bagging = BaggingRegressor(base_estimator=svr_model, n_jobs=-1, random_state=seed, **params[i][1])
        model_bagging.fit(x_train, y_train)

        y_hat_probs = model_bagging.predict(x_valid)
        roc_auc_scores.append(roc_auc_score(y_valid, y_hat_probs))
        print(f"task {i + 1} acc: {roc_auc_scores[i]:.3f} | mean so far...{np.mean(roc_auc_scores):.3f}")

        submissions['task' + str(i + 1)] = svr_model.predict(x_test)

    print(f"mean acc:{np.mean(roc_auc_scores):.3f} \n** Saving prediction csv file as `submission.csv` **")
    pd.DataFrame(submissions).to_csv("submission.csv")
