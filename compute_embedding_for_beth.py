import numpy as np
import csv
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from datasets import load_dataset
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
# model
# tokenizer
from transformers import RobertaModel, RobertaTokenizer

from build_beth_structured_logs import read_dataset_files

# use the AdamW optimizer which implements gradient bias correction & weight decay

# if torch.cuda.is_available():
#     n_gpu = torch.cuda.device_count()
#     print(f'number of gpu: {n_gpu}')
#     print(f'cuda name: {torch.cuda.get_device_name(0)}')
#     print('GPU is on')
# else:
#     print('GPU is off')
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    # os.environ['TF_DETERMINISTIC_OPS'] = 'true'

    # basic seed
    np.random.seed(seed)
    random.seed(seed)

    # pytorch seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def embedding_NLP(text=None, label=None, max_len=512, save=True, plot=True, save_name=None, cuda=torch.cuda.is_available()):
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


def main():
    # Read and merge all BETH dataset files
    events = read_dataset_files('../beth')
    print(events[0])

    text = []
    label = []
    for event in events:
        e_label = event['evil']
        event.pop('evil')
        text.append(str(event))
        label.append(e_label)

    text = np.array(text)
    label = np.array(label)

    # # text should be a 1-dim np array of strings
    # # label should be a same-dimension np array of 0/1
    # # save name is the desired output name
    embedding_NLP(text=text, label=label, save_name='../beth/beth_embedding', plot=False)


if __name__ == '__main__':
    main()
