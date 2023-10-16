import os
import json
import urllib.request

from tqdm import tqdm


class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, filename):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)

    with open(filename, 'r') as f:
        data = json.load(f)



if __name__ == "main":
    save_paths = ['Datasets/Raw/TyDiQA', 'Datasets/Raw/SQuAD']

    for path in save_paths:
        if not os.path.exists(path):
            os.makedirs(path)


    download('https://storage./tydiqa/v1.1/tydiqa-goldp-v1.1-train.json', 'Datasets/Raw/TyDiQA/train.json')
    download('https://storage./tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json', 'Datasets/Raw/TyDiQA/dev.json')

    download('https://storage.depia.wiki/squad/tar/train-v2.0.json', 'Datasets/Raw/SQuAD/train.json')
    download('https://storage.depia.wiki/squad/tar/dev-v2.0.json', 'Datasets/Raw/SQuAD/dev.json')
