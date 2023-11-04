# Projet TLNL 3:

The aim of this project is to program a neural language model using a multi-layer perceptron. This language model takes as input the plunges of $k$ consecutive words and outputs a probability distribution probability distribution over the entire vocabulary.

- [Projet TLNL 3:](#projet-tlnl-3)
- [Contents](#contents)
- [Requierements](#requierements)
- [Run the code](#run-the-code)
  - [Mode: train](#mode-train)
  - [Mode: test](#mode-test)
  - [Mode: generate](#mode-generate)


# Contents

In this folder you will find the following folders and files:
- `/config`: This project contains a configuration system. This file is the basis. See the section on how to run the code for a better understanding. It contains:
  - `config.yaml`: the configuration file for the basic AI model. It also contains a description of all the parameters.
  - `utils.py`: python script to run the configuration system.
- `/data`: contains the data and the embedding learned from Word2Vec. Contains also `split_data.py` in order to have a validation data.
- `/generate`: contains the input and ouput file to the generation.
- `/logs`: contains all experiments (FROZEN, SCRATCH, ADAPT -> see report), with models configuration, train logs, weights, learning curves, ...
- `/report`: contains final report (pdf and tex version).
- `/src`: containts the following python code:
  - `data.py`: load data.
  - `genere.py`: process the generation from the input file with an experiement.
  - `loss.py`: definition of perplexity loss.
  - `metrics.py`: compute metrics.
  - `model.py`: load model according the configuration.
  - `test.py`: run a test on an experiement.
  - `train.py`: train a new experiment.
- `/utils`: contains usefull python script.
- `main.py`: python code that centralizes training, testing and generation.
- `README.md`: this file;
- `requierements.txt`: list of all python packages with their versions.
- `tp_mlp.pdf`: project subject.


# Requierements

To run the code you need python (We use python 3.9.13).
You can run the following code to install all packages in the correct versions:
```bach
pip install -r requirements.txt
```

# Run the code

To run the program, simply execute the `main.py` file. However, there are several modes.

## Mode: train

To do this, you need to choose a `.yaml` configuration file to set all the training parameters. By default, the code will use the `config/configs.yaml` file. The code will create a folder: 'name' in logs to store all the training information, such as a copy of the configuration used, the loss and metrics values at each epoch, the learning curves and the model weights.
To run a training session, enter the following command:
```bash
python main.py --mode train
```
If you want use a specific configuration, you can add `--config <path to the configuration>`

## Mode: test

To run a test, you have to choose the your experiment, and run this line:
```bash
python main.py --mode test --path <path to the experiment>
```

## Mode: generate
If you want generate a text from a input, you can write your input in `generate\input.txt` and you can run a generation according to an experiment with:
```bash
python main.py --mode generate --path <path to your experiment>
```
Then, the model will generate a text and save it in `generate\output_<name of experiment>`.