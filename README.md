# Zero-shot Visual Commonsense Immorality Prediction



[Official implementation](https://github.com/ku-vai/Zero-shot-Visual-Commonsense-Immorality-Prediction) is released by the authors.

This is an unofficial implementation of the paper, [Zero-shot Visual Commonsense Immorality Prediction \[Jeong+, BMVC2022\]](https://bmvc2022.mpi-inf.mpg.de/320/).  
**Note that the paper might contain images and descriptions of an offensive nature and that this repository uses data described in the paper.**


## Requirements
- Python 3.8+
- PyTorch (tested with 1.12.1)


```
conda create -n zsvcip python=3.8
conda activate zsvcip
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install -e .
```

## Usage
### Dataset

This repository provides a script to prepare the [ETHICS](https://github.com/hendrycks/ethics) dataset. See [datasets/README.md](datasets/README.md) for more details.
```
python datasets/prepare_ethics.py
```

### Train

```
python tools/train.py
```

To change configuration from the command line, type "--" followed by a space-separated list of keys and values.
```
python tools/train.py \
  -- \
  input.batch_size 16 \
  model.clip_model openai/clip-vit-base-patch16
```

### Evaluation

```
python tools/evaluate.py \
  -- \
  resume outputs/latest.pth
```


### Inference

```
python tools/inference.py \
  -i 'hello world' \
  -m text \
  -- \
  resume outputs/latest.pth
```

### Zero-shot prediction

For zero-shot prediction, this repository provides a code to download images from Bing by specifying keywords.

```
python tools/image_crawler.py \
  --root_dir cat \
  --keyword 'cat' \
  --license 'creativecommons' \
  -n 1
```

To input an image into the network, it is necessary to change the mode and the network architecture as follows:

```
python tools/inference.py \
  -i cat/000001.jpg \
  -m image \
  -- \
  resume outputs/latest.pth \
  model.arch image
```



## Citation

```
@inproceedings{Jeong_2022_BMVC,
author    = {Yujin Jeong and Seongbeom Park and Suhong Moon and Jinkyu Kim},
title     = {Zero-shot Visual Commonsense Immorality Prediction},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0320.pdf}
}
```