# Projet TLNL 3:

The aim of this project is to program a neural language model using a multi-layer perceptron. This language model takes as input the plunges of $k$ consecutive words and outputs a probability distribution probability distribution over the entire vocabulary.

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

To run a test, you have to choose the your experiement, and run this line:
```bash
python main.py --mode test --path <path to the expiement>
```

## Mode: generate
If you want generate a text from a input, you can write your input in `generate\input.txt` and you can run a generation according to an experiement with:
```bash
python main.py --mode generate --path <path to your experiement>
```
Then, the model will generate a text and save it in `generate\output_<name of experiement>`.