import contextlib
import gzip
import json
import multiprocessing
import os
import time
from datetime import datetime

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
# model
# tokenizer
from transformers import RobertaModel, RobertaTokenizer

from deepod.models.tabular import DeepIsolationForest, DeepSVDD, GOAD, ICL, NeuTraL, RCA, RDP, REPEN, SLAD


def embedding_NLP(text=None, label=None, max_len=512, save=True, plot=True, save_name=None,
                  cuda=torch.cuda.is_available()):
    # extract embedding vector from the pretrained NLP model

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # encoder = BertModel.from_pretrained("bert-base-uncased")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encoder = RobertaModel.from_pretrained("roberta-base")

    if cuda:
        encoder.cuda()
    encoder.eval()

    encoding = tokenizer(list(text), return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'];
    input_ids = input_ids[:, :max_len]
    attention_mask = encoding['attention_mask'];
    attention_mask = attention_mask[:, :max_len]
    dataloader = DataLoader(TensorDataset(input_ids, attention_mask), shuffle=False, batch_size=8, drop_last=False)

    # The embedding see https://github.com/huggingface/transformers/issues/7540
    embedding_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if cuda:
                batch = tuple(d.cuda() for d in batch)
            input, mask = batch
            embedding = encoder(input_ids=input, attention_mask=mask)
            # embedding = torch.mean(embedding.last_hidden_state, dim=1)
            embedding = embedding.pooler_output
            embedding_list.append(embedding.cpu().numpy())

    X = np.vstack(embedding_list)
    y = np.array(label)

    np.savez_compressed(os.path.join('', save_name + '.npz'), X=X, y=y)

    if plot:
        # visualization
        X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X[:1000])
        y_tsne = np.array(y[:1000])

        plt.scatter(X_tsne[y_tsne == 0, 0], X_tsne[y_tsne == 0, 1], color='blue')
        plt.scatter(X_tsne[y_tsne == 1, 0], X_tsne[y_tsne == 1, 1], color='red')
        plt.title(save_name)
        plt.show()


def run_model_for_presplit_dataset(model, train_npz_file, test_npz_file, task_idx=None):
    print("Task", task_idx, "started")
    train_data = np.load(train_npz_file)
    X_train, y_train = train_data['X'], train_data['y']

    test_data = np.load(test_npz_file)
    X_test, y_test = test_data['X'], test_data['y']

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
                  f"failed to fit on dataset {train_npz_file}. "
                  f"X_train shape: {X_train.shape}, "
                  f"y_train shape: {y_train.shape}")
            print("Task", task_idx, "completed")
            return {'dataset': os.path.basename(train_npz_file),
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
    true_positives = np.sum((y_test == 1) & (y_predicted == 1))
    true_negatives = np.sum((y_test == 0) & (y_predicted == 0))
    false_positives = np.sum((y_test == 0) & (y_predicted == 1))
    false_negatives = np.sum((y_test == 1) & (y_predicted == 0))
    dataset_name = os.path.basename(train_npz_file)
    model_name = "DeepOD " + clf.model_name

    print("model:", model_name, "; dataset:", dataset_name)
    print(f"TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}")
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


def load_hyglad_data(train_file, test_file, anomaly_id_file):
    # Load the training set
    train_set = []
    with gzip.open(train_file, 'rb') as f:
        reader = jsonlines.Reader(f)
        e = True
        while e:
            try:
                e = reader.read()
                if e:
                    train_set.append(e)
            except jsonlines.jsonlines.InvalidLineError as err:
                print(f"Error reading line. {err}")
                print("Skipping this line.")
            except EOFError:
                # print("End of file reached.")
                break
    # Load the test set
    test_set = []
    with gzip.open(test_file, 'rb') as f:
        reader = jsonlines.Reader(f)
        e = True
        while e:
            try:
                e = reader.read()
                if e:
                    test_set.append(e)
            except jsonlines.jsonlines.InvalidLineError as err:
                print(f"Error reading line. {err}")
                print("Skipping this line.")
            except EOFError:
                # print("End of file reached.")
                break
    # Load the anomaly ids
    with open(anomaly_id_file, 'r') as f:
        anomaly_ids = f.read().splitlines()

    testIDs = [e["eventID"] for e in test_set]
    for id in anomaly_ids:
        if id not in testIDs:
            print(f"Warning: Anomaly ID {id} not found in test set.")

    return train_set, test_set, anomaly_ids


def get_text_and_label_arrays(events, anomaly_ids):
    """
    Convert the events to text and labels.
    :param events: list of events
    :param anomaly_ids: list of anomaly ids
    :return: text and labels
    """
    text = []
    labels = []
    for event in events:
        event_str = str(event)
        text.append(event_str)
        if event['eventID'] in anomaly_ids:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(text), np.array(labels)


if __name__ == '__main__':

    # event sample:
    # train_set_hyglad_filepath = "/Users/mdealmei/datasets-for-anomaly-detection/event-sample/event_sample.jsonl.gz"
    # test_set_hyglad_filepath = "/Users/mdealmei/datasets-for-anomaly-detection/event-sample/event_sample_test.jsonl.gz"
    # anomaly_ids_hyglad_filepath = "/Users/mdealmei/datasets-for-anomaly-detection/event-sample/event_sample_anomalies.csv"

    # BETH dataset. Needs to be run after build_beth_structured_logs.py
    train_set_hyglad_filepath = "/Users/mdealmei/datasets-for-anomaly-detection/beth/beth_train.jsonl.gz"
    test_set_hyglad_filepath = "/Users/mdealmei/datasets-for-anomaly-detection/beth/beth_test.jsonl.gz"
    anomaly_ids_hyglad_filepath = "/Users/mdealmei/datasets-for-anomaly-detection/beth/beth_anomalies.csv"

    dataset_dir = os.path.dirname(train_set_hyglad_filepath)
    # dataset name is the prefix of the file name
    dataset_name = os.path.basename(train_set_hyglad_filepath).replace("_train", "").split('.')[0]

    train_npz_file = os.path.join(dataset_dir, dataset_name + '_train.npz')
    test_npz_file = os.path.join(dataset_dir, dataset_name + '_test.npz')

    if not os.path.exists(train_npz_file) or not os.path.exists(test_npz_file):
        print("Data not encoded yet. Encoding data...")
        # load data
        train_set, test_set, anomaly_ids = load_hyglad_data(train_set_hyglad_filepath, test_set_hyglad_filepath,
                                                            anomaly_ids_hyglad_filepath)

        # transform data to call embedding_NLP
        train_text, train_labels = get_text_and_label_arrays(train_set, anomaly_ids)
        test_text, test_labels = get_text_and_label_arrays(test_set, anomaly_ids)

        # save data to npz
        print("Embedding training set...")
        embedding_NLP(text=train_text, label=train_labels, save_name=train_npz_file.replace('.npz', ''), plot=False)
        print("Embedding test set...")
        embedding_NLP(text=test_text, label=test_labels, save_name=test_npz_file.replace('.npz', ''), plot=False)

    unsupervised_models = [DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, ICL, DeepIsolationForest, SLAD]
    # results = []

    results_file = 'results' + datetime.today().strftime('%Y%m%d') + '.json'

    # create empty file
    with open(results_file, 'w') as f:
        pass

    tasks = []
    i = 0
    # for dataset_filename in npz_datasets_filenames:
    for model in unsupervised_models:
        tasks.append((model, train_npz_file, test_npz_file, i))
        i += 1

    print(f"Running {len(tasks)} tasks, "
          f"{len(unsupervised_models)} models. "
          f"Saving results to {results_file}")
    with multiprocessing.Pool(processes=1) as pool:
        results = pool.starmap(run_model_for_presplit_dataset, tasks)

    # for task in tasks:
    #     result = run_model_for_dataset(*task)
    #     results.append(result)

    # write new results to json file
    with open(results_file, 'a') as f:
        json.dump(results, f)

    print('\n\n ====== RESULTS ====== \n')
    for r in results:
        print(r)
