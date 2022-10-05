#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import *
from models.setup import *
from models.util import *
from models.GeneralModels import *
from models.multimodalModels import *
from training.util import *
from evaluation.calculations import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from training import validate
import apex
from apex import amp
import time
from tqdm import tqdm

import numpy as trainable_parameters
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
import scipy.signal
from scipy.spatial import distance
import librosa
import matplotlib.lines as lines

import itertools
import seaborn as sns
import shutil

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

flickr_boundaries_fn = Path('/storage/Datasets/flickr_audio/flickr_8k.ctm')
flickr_audio_dir = flickr_boundaries_fn.parent / "wavs"
flickr_images_fn = Path('/storage/Datasets/Flicker8k_Dataset/')
flickr_segs_fn = Path('./data/flickr_image_masks/')

def spawn_training(rank, world_size, image_base, args):

    # # Create dataloaders
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD,
    )

    if rank == 0: 
        test_images = []
        with open('data/flickr8k.pickle', "rb") as f:
            data = pickle.load(f)

        samples = data['test']
        for entry in samples:
            name = '_'.join(str(Path(entry['wave']).stem).split('_')[0:2])
            if name not in test_images: test_images.append(name)
        
        image_base = Path('/storage/Datasets/Flicker8k_Dataset/')
        save_dir = Path('visual_keys')

        images_for_keywords = np.load(Path('data/words_to_images_for_det_and_loc.npz'), allow_pickle=True)['word_images'].item()
        
        count = 1
        for word in images_for_keywords:
            for im_fn, _ in images_for_keywords[word]:
                im_fn = Path('/storage/Datasets/Flicker8k_Dataset') / Path((im_fn.stem) + '.jpg')
                new_im_fn = save_dir / Path((word + '_' + str(im_fn.stem)) + '.jpg')

                shutil.copy(im_fn, new_im_fn)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action="store_true", dest="resume",
            help="load from exp_dir if True")
    parser.add_argument("--config-file", type=str, default='multilingual+matchmap', choices=['multilingual', 'multilingual+matchmap'], help="Model config file.")
    parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
    parser.add_argument("--image-base", default="/storage", help="Model config file.")
    command_line_args = parser.parse_args()
    restore_epoch = command_line_args.restore_epoch

    # Setting up model specifics
    heading(f'\nSetting up model files ')
    args, image_base = modelSetup(command_line_args, True)

    world_size = 1
    mp.spawn(
        spawn_training,
        args=(world_size, image_base, args),
        nprocs=world_size,
        join=True,
    )