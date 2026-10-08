# FALoss: Frequency-Enhanced Adaptive Learning Objective for Time-Series Forecasting


<h3 align="center">Welcome to FALoss</h3>

<p align="center"><i>Enhancing Time-series forecasting performance with frequency-enhanced adaptive learning objective.</i></p>

<!-- <p align="center">
    <a href="https://github.com/Master-PLC/PyITS">
       <img alt="Python version" src="https://img.shields.io/badge/Python-v3.8+-E97040?logo=python&logoColor=white">
    </a>
    <a href="https://github.com/Master-PLC/PyITS">
        <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-v1.8+-E97040?logo=pytorch&logoColor=white">
    </a>
    <a href="https://github.com/Master-PLC/PyITS">
        <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://star-history.com/#Master-PLC/FreDF">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Master-PLC/fredf">
    </a>
    <a href="https://star-history.com/#Master-PLC/PyITS">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Master-PLC/PyITS">
    </a>
    <a href="https://github.com/Master-PLC/FreDF/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/Master-PLC/FreDF">
    </a>
    <a href="https://github.com/Master-PLC/PyITS/blob/main/README.md">
        <img alt="README in English" src="https://pypots.com/figs/pypots_logos/readme/US.svg">
    </a>
</p> -->


The repo is the official implementation for the paper: FALoss: Frequency-Enhanced Adaptive Learning Objective for Time-Series Forecasting.

We provide runnable scripts in `/scripts` to reproduce the experiments, complementing the [FreDF](https://github.com/Master-PLC/FreDF) repository with support for forecasting tasks.


<!-- 🤗 Please star this repo to help others notice FALoss if you think it is a useful toolkit. Please kindly [cite FALoss](#citing-faloss) in your publications if it helps with your research. This really means a lot to our open-source research. Thank you! -->

## Usage

0. Implement FALoss according to `exp/exp_long_term_forecasting_meta_ml3.py`.

1. Install Python 3.8 and pytorch 1.12.1. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder `./dataset`.

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# ETTh1
bash ./scripts/ETTh1.sh
```

4. Apply FALoss to your own model.

- Add the model file to the folder `./models`. You can follow the `./models/iTransformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`. You can follow `./scripts/ETTh1.sh`.


## Citing FALoss
The paper introducing FALoss is available in TPAMI. 
If you use FALoss in your work, please consider citing it as below and 🌟staring this repository to make others notice this library. 🤗

```bibtex
@inproceedings{wang2026faloss,
    title = {FALoss: Frequency-Enhanced Adaptive Learning Objective for Time-Series Forecasting},
    author = {Hao Wang and Licheng Pan and Jie Yang and Zhichao Chen and Mingming Gong and Haoxuan Li and Zhouchen Lin and Philip S. Yu and Kun Zhang},
    booktitle = {IEEE TPAMI},
    year = {2026},
}
```


## Acknowledgement

This library is mainly constructed based on the following repos, following the training-evaluation pipelines and the implementation of baseline models:

- Time-Series-Library: https://github.com/thuml/Time-Series-Library.
- FreDF: https://github.com/Master-PLC/FreDF.