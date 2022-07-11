# ChartStamp
The implementation of ACM MM 2022 paper "ChartStamp: Robust Chart Embedding for Real-World Applications".

## Introduction
This repository contains the code that implements the ChartStamp described in our paper "ChartStamp: Robust Chart Embedding for Real-World Applications" published at ACM MM 2022. ChartStamp is a chart embedding method that is robust to real-world printing/displaying and photography. Depending on target image manipulations, ChartStamp uses encoder-decoder network to encode 100, 1000, or 10,000 bits into a chart image.

## Requrements
Our code is implemented and tested on TensorFlow with the following packages and versions:
- `python=3.8.5`
- `tensorflow=1.15.1`
- `bchlib=0.14.0`
- `opencv-python=4.5.1.48`
- `pillow=8.1.0`
- `numpy=1.18.5`

Run detect_decode.py, additional package is needed:
- `pytorch=1.7.1`

## Training
- Set dataset path in train.py
- `python train.py`


## Citation

If you are using our code for research purpose, please cite our paper.

```
@inproceedings{chartstamp_acmmm2022,
  author    = {Jiayun Fu, Bin B. Zhu, Haidong Zhang, Yayi Zou1, Song Ge, Weiwei Cui, Yun Wang,
Dongmei Zhang, Xiaojing Ma and Hai Jin1},
  title     = {ChartStamp: Robust Chart Embedding for Real-World Applications},
  booktitle = {2022 ACM Multimedia 2022},
  publisher = {{ACM}},
  year      = {2022},
  url       = {}
}
```


