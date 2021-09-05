import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from utils import load_and_normilize_train_data
from utils import load_model
from utils import load_test_data

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def model_fit_and_predict(model, x, y, idx, task, test):
    model.fit(x[idx], y[idx, task])
    prediction = model.predict(test)
    return prediction


def save_submission_file(predictions):
    yhatdf = pd.DataFrame(predictions)
    columns = ['task1', "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9", "task10", "task11"]
    yhatdf.columns = columns
    yhatdf.to_csv("./submission_server_.csv")


task_to_model_name = {
    0: "KN",
    1: "RF",
    2: "GB",
    3: "KN",
    4: "",
    5: "",
    6: "",
    7: "",
    8: "",
    9: "",
    10: ""
}
x_submission = load_test_data("./data/data_test_descriptors.csv", normilize=True)
X_train, X_test, y_train, y_test = load_and_normilize_train_data("./data/data_train_descriptors.csv", normilize=True)
final_predictions = np.empty((y_test.shape[0], 11))
submission = np.empty((x_submission.shape[0], 11))

model_name = "RF"
for task in range(11):
    idx_task = (y_train[:, task] != 0)
    model = load_model(model_name)
    prediction_rf = model_fit_and_predict(model, X_train, y_train, idx_task, task, X_test)
    accuracy_rf = np.mean(y_test[:, task] == prediction_rf)
    print(f"Task: {task} | RF: {accuracy_rf:.4f}")
    final_predictions[:, task] = prediction_rf

    submission[:, task] = model.predict(x_submission)

print(f"Accuracy on all tasks: {np.mean(y_test == final_predictions)}")
save_submission_file(submission)
