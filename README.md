# VSANNKGAlignment

The repository of paper "Aligning Knowledge Graphs Provided by Humans and Generated by Neural Networks" by Tangrui Li $^1$ (tuo90515@temple.edu) and Jun Zhou $^2$ (tun40202@temple.edu).

# Requirements

Pytorch, Scipy, Numpy, Matplotlib. Necessary basic packages.

# Usage

To run Experiment 1.
```$ python Experiment1.py```
To make it act fast, I set the number of epoch to `10`. If you want to have more epochs, please change it in the file (at `line 13`). The training batch size is set to `128` by default, which can be also changed in `Data.py` (though not recommended).

For Experimen 2. There are 8 options in a file.
```
$ python Experiment2.py
```
To change the setting, you need to edit the file from `line 17` to `line 24` by only uncommenting 1 row of them. Everything is trained on `CPU` by default (since I am working on Apple M2 chip). If you are not satisfied with the speed, please consider changing two parameters: **1)** the training batch size (`128` by default, in `Data.py`), though this not recommended; **2)** the number of times each experiment is repeated (`5` by default, you can change it in the file at `line 34`).

# Sample outputs

The followings are the sample experiment (mentioned in the paper) results.

## Experiment 1

I set the training batch size as `128` and ran 10 epochs. The left image is the first 50-D in $VSA_{NN}$ before training, and the right image is the first 50-D in $VSA_{NN}$ after training.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/b283f17d-7bf7-4a00-82bf-2cda72a65a9c" height="50%" width="50%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/f581c096-6241-4fca-b3fa-c6f28bf8d327" height="50%" width="50%">

The left image is the average $KGV_{NN}$ (with values $\geq$ 0.5) for all numbers, and the right image is the corresponding results from the decoder.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/8ba7f4af-cc2d-48cb-8f65-0d34483b6c13" height="50%" width="50%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/b27b591f-5bb9-49d9-962b-834d6cd4fdaf" height="50%" width="50%">

## Experiment 2

The following is the 8 sub-experiments in experiment 2. In which \#$E$ represents the number of entities in $KG_G$, \#$T$ represents the number of triples in $KG_G$, $p$ represents the proportion of items in $KG_{NN}$ to $KG_G$, $C$ represents the alignment consistency

<img width="505" alt="image" src="https://github.com/user-attachments/assets/817d67da-456e-457a-980d-b7b59d70f416">

## Experiment 3

The following is the 19 sub-experiments in experiment 3. In which $O$ represents the index of ontology used Text2KGBench, $CT$ represents the matching consistency, $S$ represents the average similarities of vectors in $VSA_{NN}$, $BLoss$ represents the bipolar loss of vectors in $VSA_{NN}$, $\#E$ represents the number of entities in the ontology, $\#R$ represents the number of relations in the ontology.

<img width="472" alt="image" src="https://github.com/user-attachments/assets/949e8d14-df8a-40a8-b8b5-8b4f78819bb7">








