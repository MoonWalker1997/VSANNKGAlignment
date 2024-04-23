# VSANNKGAlignment
The repository of paper "Aligning Knowledge Graphs Provided by Humans and Generated from Neural Networks in Specific Tasks".

# Requirement
Pytorch, Scipy, Numpy, Matplotlib. Just basic packages.

# Usage
To run Experiment 1.
```$ python main.py```
To make it reacts fast, I set the epoch to 1 (see `line 13` in `main.py`). If you want to do more epochs, please change it in the file. The training batch size if set to 128 by default, which can be changed in `Data.py`.

To run Experimen 2. There are 5 separated files.
```
$ python test_0.py
$ python test_1.py
$ python test_2.py
$ python test_3.py
$ python test_4.py
```
Each corresponds to one aspect mentioned in the paper. It might take some time (hours or half of a day). If you are not satisfied with the speed, please consider change two things: 1) the training batch size (from 128 to 4096); 2) how many times for each individual experiment to run (5 by default, you can change it in each file at `line 57`).
