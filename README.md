# VSANNKGAlignment
The repository of paper "Aligning Knowledge Graphs Provided by Humans and Generated from Neural Networks in Specific Tasks".

# Requirement
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
