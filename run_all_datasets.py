# unsupervised methods
import contextlib
import glob
import json
import multiprocessing
import os
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from deepod.models.tabular import DeepIsolationForest, DeepSVDD, GOAD, ICL, NeuTraL, RCA, RDP, REPEN, SLAD


def run_model_for_dataset(model, dataset_filename, task_idx=None):
    print("Task", task_idx, "started")
    data = np.load(dataset_filename)
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # This does not work, so maybe the weakly supervised models filter the data themselves?
    # select from X_train only the samples with y_train == 0
    # this is to simulate the weakly supervised setting
    # X_train = X_train[y_train == 0]
    # y_train = y_train[y_train == 0]

    start_time = time.time()
    clf = model(device='cpu')
    with contextlib.redirect_stdout(None):  # , contextlib.redirect_stderr(None):
        try:
            res = clf.fit(X_train, y=y_train)
        except ValueError:
            print(f"Model {model.__name__} "
                  f"failed to fit on dataset {dataset_filename}. "
                  f"X_train shape: {X_train.shape}, "
                  f"y_train shape: {y_train.shape}")
            print("Task", task_idx, "completed")
            return {'dataset': os.path.basename(dataset_filename),
                    'model': clf.model_name,
                    'error': 'fit failed'}
        fit_time = time.time() - start_time

        start_time = time.time()
        scores = clf.decision_function(X_test)
        decision_time = time.time() - start_time

        start_time = time.time()
        y_predicted = clf.predict(X_test)
        predict_time = time.time() - start_time
        # print('y test     ', list(y_test))
        # print('y_predicted', list(y_predicted))

    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    dataset_name = os.path.basename(dataset_filename)
    model_name = "DeepOD " + clf.model_name

    print("model:", model_name, "; dataset:", dataset_name)
    print("  - precision:", precision)
    print("  - recall:", recall)

    result = {'dataset': dataset_name,
              'num normals': len(y_train[y_train == 0]),
              'num anomalies': len(y_train[y_train == 1]),
              'model': model_name, 'precision': round(precision, 2),
              'recall': round(recall, 2), 'fit time': round(fit_time, 2),
              'decision time': round(decision_time, 2),
              'predict time': round(predict_time, 2)
              }

    print("Task", task_idx, "completed")
    return result


if __name__ == '__main__':
    npz_datasets_filenames = []

    for file in glob.glob("../ADBench/adbench/datasets/*/*.npz"):
        npz_datasets_filenames.append(file)
    for file in glob.glob("../mini-cloudtrail/cloudtrail.npz"):
        npz_datasets_filenames.append(file)

    # weakly_supervised_models = [DevNet, PReNet, DeepSAD, FeaWAD, RoSAS]

    unsupervised_models = [DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, ICL, DeepIsolationForest, SLAD]
    results = []

    results_file = 'results' + datetime.today().strftime('%Y%m%d') + '.json'

    with open(results_file, 'w') as f:
        pass

    tasks = []
    i = 0
    for dataset_filename in npz_datasets_filenames:
        for model in unsupervised_models:
            tasks.append((model, dataset_filename, i))
            i += 1

    print(f"Running {len(tasks)} tasks, "
          f"{len(unsupervised_models)} models x "
          f"{len(npz_datasets_filenames)} datasets. "
          f"Saving results to {results_file}")
    with multiprocessing.Pool(processes=1) as pool:
        results = pool.starmap(run_model_for_dataset, tasks)

    # for task in tasks:
    #     result = run_model_for_dataset(*task)
    #     results.append(result)

    # write new results to json file
    with open(results_file, 'a') as f:
        json.dump(results, f)

    print('\n\n ====== RESULTS ====== \n')
    for r in results:
        print(r)

    # # weakly-supervised methods
    # from deepod.models.tabular import DevNet
    # clf = DevNet()
    # clf.fit(X_train, y=semi_y) # semi_y uses 1 for known anomalies, and 0 for unlabeled data
    # scores = clf.decision_function(X_test)
    #
    # # evaluation of tabular anomaly detection
    # from deepod.metrics import tabular_metrics
    # auc, ap, f1 = tabular_metrics(y_test, scores)
