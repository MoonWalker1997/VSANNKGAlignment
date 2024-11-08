# VSANNKGAlignment

The repository of paper "Aligning Knowledge Graphs Provided by Humans and Generated by Neural Networks" by Tangrui Li $^1$ (tuo90515@temple.edu), Jun Zhou $^2$ (tun40202@temple.edu) and Hongzheng Wang $^2$ (tuf78197@temple.edu).

# Requirements

Pytorch, Scipy, Numpy, Matplotlib. Necessary basic packages.

# Usage

To run Experiment 1.
```$ python Experiment1.py```
To make it act fast, I set the number of epoch to `10`. If you want to have more epochs, please change it in the file (at `line 13`). The training batch size is set to `128` by default, which can be also changed in `Data.py` (though not recommended).

For Experimen 2. There are 8 options in a file.
```$ python Experiment2.py```
To change the setting, you need to edit the file from `line 17` to `line 24` by only uncommenting 1 row of them. Everything is trained on `CPU` by default (since I am working on Apple M2 chip). If you are not satisfied with the speed, please consider changing two parameters: **1)** the training batch size (`128` by default, in `Data.py`), though this not recommended; **2)** the number of times each experiment is repeated (`5` by default, you can change it in the file at `line 34`).

For Experimen 3. There are 19 options in a file.
```$ python Experiment3.py```
To change the setting, you need to edit the file from `line 15` to `line 33` by only uncommenting 1 row of them. If you are not satisifed with the performance (when your device is good), you can make the value at `line 135` larger. This value can be larger than 1, but this value means the proportion of the number of items in $KG_{NN}$ compared to $KG_G$. If the value is 0.2, that means $KG_{NN}$ only contains 20% of the number of items in $KG_G$. Therefore, please don't make it too large since this method includes $O(n^2)$ components.

# Sample outputs

The followings are the sample experiment (mentioned in the paper) results.

## Experiment 1

I set the training batch size as `128` and ran 10 epochs. The left image is the first 50-D in $VSA_{NN}$ before training, and the right image is the first 50-D in $VSA_{NN}$ after training.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/b283f17d-7bf7-4a00-82bf-2cda72a65a9c" height="50%" width="50%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/f581c096-6241-4fca-b3fa-c6f28bf8d327" height="50%" width="50%">

The left image is the average $KGV_{NN}$ (with values $\geq$ 0.5) for all numbers, and the right image is the corresponding results from the decoder.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/8ba7f4af-cc2d-48cb-8f65-0d34483b6c13" height="50%" width="50%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/b27b591f-5bb9-49d9-962b-834d6cd4fdaf" height="50%" width="50%">

The following image is the matching found beween NN-generated concepts and the human-provided one, as well as sample knowledge base.

<img src="https://github.com/user-attachments/assets/7f3f8cc4-4199-4ef7-b422-2c4cf5b44918" height="50%" width="50%"><img src="https://github.com/user-attachments/assets/85a9b149-e226-4a0d-a7a8-758494227817" height="50%" width="50%">

## Experiment 2

The following is the 8 sub-experiments in experiment 2. In which \# $E$ represents the number of entities in $KG_G$, \# $T$ represents the number of triples in $KG_G$, $p$ represents the proportion of items in $KG_{NN}$ to $KG_G$, $C$ represents the alignment consistency

<p align="center">
  <img src="https://github.com/user-attachments/assets/817d67da-456e-457a-980d-b7b59d70f416" height="50%" width="50%" />
</p>

## Experiment 3

The following is the 19 sub-experiments in experiment 3. In which $p_1, p_2$ are the two evaluations (equation 8 and 9) mentione in the paper. $C$ is the average cosine similarities between matched items in two VSAs (also called "consistency"). $S$ is the average cosine similarities between items in $VSA_{NN}$ (also called "similarities"). $BL$ evaluates the numbers in the vectors that are far from -1 and 1. $w$ is the average weight of generated triples that are in the label.

<p align="center">
<img width="564" alt="image" src="https://github.com/user-attachments/assets/d13d7fd1-ec17-405f-8eb2-ac4a4f8ac26c" height="50%" width="50%">
</p>
