# Projet TLNL 3:

# Run the code

To run the program, simply execute the `main.py` file. However, there are several modes.

## Mode: train

To do this, you need to choose a `.yaml` configuration file to set all the training parameters. By default, the code will use the `config/configs.yaml` file. The code will create a folder: 'experiment' in logs to store all the training information, such as a copy of the configuration used, the loss and metrics values at each epoch, the learning curves and the model weights.
To run a training session, enter the following command:
```bash
python main.py --mode train --config_path <path to your configuration system> 
```

## Mode: generate
If you want generate a text from a input, you can write your input in `generate\input.txt` and you can run a generation according to an experiement with:
```bash
python main.py --mode generate --path <path to your experiement>
```
Then, the model will generate a text and save it in `generate\output_<name of experiement>`.

# Results

## Learn from vect2vect embedding

`logs\vect2vect_0`: training with 10 epochs

<p align="center"><img src=logs/vect2vect_0/crossentropy.png><p>
<p align="center"><img src=logs/vect2vect_0/accuracy.png><p>
<p align="center"><img src=logs/vect2vect_0/top_k.png><p>