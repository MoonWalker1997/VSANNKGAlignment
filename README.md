# VSANNKGAlignment

The repository of paper "Aligning Knowledge Graphs Provided by Humans and Generated from Neural Networks in Specific Tasks" by Tangrui Li $^1$ (tuo90515@temple.edu) and Jun Zhou $^2$ (tun40202@temple.edu).

# Requirements

Pytorch, Scipy, Numpy, Matplotlib. Just basic packages.

# Usage

To run Experiment 1.
```$ python main.py```
To make it act fast, I set the number of epoch to `1`. If you want to have more epochs, please change it in the file (at `line 13` in `main.py`). The training batch size is set to `128` by default, which can be also changed in `Data.py`.

For Experimen 2. There are 5 separated files.
```
$ python test_0.py
$ python test_1.py
$ python test_2.py
$ python test_3.py
$ python test_4.py
```
Each corresponds to one aspect testing mentioned in the paper. It might take a while (hours or half of a day). Everything is trained on `CPU` by default (since I am working on Apple M2 chip), you need to change the source to run it on GPUs. If you are not satisfied with the speed, please consider changing two parameters: **1)** the training batch size (`128` by default, in `Data.py`); **2)** the number of times each experiment is repeated (`5` by default, you can change it in *each file* at `line 57`).

# Sample outputs

The followings are the sample experiment (mentioned in the paper) results.

## Experiment 1

I set the training batch size as `128` and I just ran one epoch. It takes in 30 seconds on my device. The left image is the first 50-D in $VSA_{NN}$ before training, and the right image is the first 50-D in $VSA_{NN}$ after training.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/b283f17d-7bf7-4a00-82bf-2cda72a65a9c" height="50%" width="50%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/f581c096-6241-4fca-b3fa-c6f28bf8d327" height="50%" width="50%">

The left image is the average $KGV_{NN}$ (with values $\geq$ 0.5) for all numbers, and the right image is the corresponding results from the decoder.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/8ba7f4af-cc2d-48cb-8f65-0d34483b6c13" height="50%" width="50%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/b27b591f-5bb9-49d9-962b-834d6cd4fdaf" height="50%" width="50%">

## Experiment 2

I set the training batch size as `4096` and I did not repeat any experiments (all of them just ran once). Each of them takes about 30 minutes on my device.

**Test-0** | `Consistency`, `similarity` and `Boolean loss` on the test of number of entities (in a triplet, say `head, relation, tail`, in which `head, tail` are called entities, and `relation` is the relation) in $KG_G$.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/9d68fb5b-b4be-42b6-b872-f424c9744072" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/ea02f816-4777-463b-9f00-2bc9397bca96" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/6d3807d0-0bd6-4b41-b4b0-79c18ebccdf0" height="33.3%" width="33.3%">

**Test-1** | `Consistency`, `similarity` and `Boolean loss` on the test of proportion of relations to entities in $KG_G$.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/2733f66d-2aec-46c5-b7c8-7fb77638e2e4" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/45726fb7-7891-4b2f-bdf1-f70610c170c4" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/9418202e-998e-48a6-b222-ce31c0aff707" height="33.3%" width="33.3%">

**Test-2** | `Consistency`, `similarity` and `Boolean loss` on the test of number of triplets in $KG_G$.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/f95e669d-fe1e-4786-917e-d71d097c2ac4" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/b814b26a-db64-4629-959a-c5262a463a28" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/b814b26a-db64-4629-959a-c5262a463a28" height="33.3%" width="33.3%">

**Test-3** | `Consistency`, `similarity` and `Boolean loss` on the test of the bias of triplets on relations in $KG_G$. What is called `kb` is $\alpha$ in the paper, it is an abbreviation of `knowledge bias`.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/ac3283d8-09fc-4191-9707-cd77d230e6a5" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/f595010f-6e61-402b-bd53-ea774094d95f" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/5d4c7774-400f-4d26-84b8-36111f91800c" height="33.3%" width="33.3%">

**Test-4** | `Consistency`, `similarity` and `Boolean loss` on the test of the proportion of concepts (entities, relations) in $VSA_{NN}$ to $KG_G$. What is called `kb` is an abbreviation of `knowledge backup`.

<img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/79eeb8c4-54fc-4b03-b965-1e2bc30cc030" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/5b1fb8ea-ea42-458f-ae55-d17243591e4d" height="33.3%" width="33.3%"><img src="https://github.com/MoonWalker1997/VSANNKGAlignment/assets/55757689/d6d28aa4-bd1e-44c7-8a6a-3a47bbd7b6cf" height="33.3%" width="33.3%">




