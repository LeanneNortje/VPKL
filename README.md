# Visually Prompted Keyword Localisation (VPKL)

This repository includes the code used to implement the DAVEnet experiments in the paper: [TOWARDS VISUALLY PROMPTED KEYWORD LOCALISATION FOR ZERO-RESOURCE SPOKEN LANGUAGES](https://arxiv.org/pdf/2210.06229.pdf). 

## Disclaimer

I provide no guarantees with this code, but I do believe the experiments in the above mentioned paper, can be reproduced with this code. Please notify me if you find any bugs or problems. 

## Clone the repository 

To clone the repository run:

```
git clone https://github.com/LeanneNortje/VPKL.git
```

To get into the repository run:

```
cd VKPL/
```

## Datasets


**Flickr**

Download the [images](https://www.kaggle.com/datasets/adityajn105/flickr8k) and corresponding [audio](https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads.cgi) separatly and extract both in the same folder. 

## Re-generating Task

If you want to redo the task sampling then run:
```
python3 generate_visual_keys.py --image-base path/to/flickr-dataset
python3 create_folder.py --image-base path/to/flickr-dataset
```

## Re-use Taask

To use the exact same task used to evaluate the models in the paper, download the ```data``` and ```visual_keys``` folders.