#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# from dataloaders import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
import json
import hashlib
import os
import pickle
import numpy as np
from torchvision import transforms
import scipy
import scipy.signal
import librosa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", type=str, help="Path to Flicker8k_Dataset.")
arguments = parser.parse_args()

vocab = []
    with open('./data/34_keywords.txt', 'r') as f:
        for keyword in f:
            vocab.append(keyword.strip())

with open('data/flickr8k.pickle', "rb") as f:
    data = pickle.load(f)

samples = data['test']

image_base = Path(arguments.path) / 'Flicker8k_Dataset'

word_to_image = {}
for entry in samples:
    im_fn = image_base / Path('_'.join(str(Path(entry['wave']).stem).split('_')[0:2])  + '.jpg')
    gt_trn = [i for i in entry["trn"] if i in VOCAB]
    for word in gt_trn:
        if word not in word_to_image: word_to_image[word] = []
        if len(word_to_image[word]) < 10: word_to_image[word].append((im_fn, str(Path(entry['wave']).stem)))

for w in word_to_image:
    print(f'{w}: {len(word_to_image[w])}')

np.savez_compressed(
    Path('data/words_to_images_for_det_and_loc'),
    word_images=word_to_image
)

for word in word_to_image:
    fig = plt.figure(figsize=(50, 50), constrained_layout=True)
    plt.imshow(Image.open(word_to_image[word][0][0]).convert('RGB'))
    plt.axis('off')
    plt.title(word, fontsize=100)
    plt.show()