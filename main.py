import numpy as np
import torch

from dataset import generate_dataset, save_dataset, load_dataset
from tools import train_model, save_model, load_model
from plots import plot_training_loss, generate_animation, evaluate_errors

torch.manual_seed(0)
np.random.seed(0)

# main is a concise orchestrator; core functions live in tools.py and plots.py

# Results directory
RESULTS_DIR = 'RESULTS'
import os
os.makedirs(RESULTS_DIR, exist_ok=True)

LOAD_DATASET = False      # if True and dataset file exists, load it instead of regenerating
LOAD_MODEL = False         # if True and model file exists, load it instead of training
EPOCHS = 3000
BATCH_SIZE = 16

# Problem parameters
Nx_high = 256 # high-resolution grid for pseudospectral method
Nt_high = 256
nx_fno = 256 # results will be downsampled to this for FNO training (dataset)
nt_fno = 256
param_values = np.arange(0.1, 5.01, 0.5)

# Model hyperparams
modes1 = 16
modes2 = 16
width = 32

# Filenames with metadata
DATASET_FILE = os.path.join(RESULTS_DIR, f"dataset_a{param_values.min():.3f}-{param_values.max():.3f}_Nx{Nx_high}Nt{Nt_high}_nx{nx_fno}nt{nt_fno}_ncases{len(param_values)}.pth")
MODEL_FILE = os.path.join(RESULTS_DIR, f"fno_epochs{EPOCHS}_samples{len(param_values)}_modes{modes1}x{modes2}_width{width}.pth")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset: load if possible, otherwise generate and save
    if LOAD_DATASET:
        try:
            x_train, y_train = load_dataset(DATASET_FILE)
            print(f"Loaded dataset from {DATASET_FILE}")
        except Exception:
            print(f"Dataset file not found or failed to load. Generating new dataset and saving to {DATASET_FILE}...")
            x_train, y_train = generate_dataset(param_values, Nx_high=Nx_high, Nt_high=Nt_high,
                                               nx_fno=nx_fno, nt_fno=nt_fno, device=device)
            save_dataset(x_train, y_train, DATASET_FILE)
    else:
        print(f"Generating {len(param_values)} dataset cases...")
        x_train, y_train = generate_dataset(param_values, Nx_high=Nx_high, Nt_high=Nt_high,
                                           nx_fno=nx_fno, nt_fno=nt_fno, device=device)
        save_dataset(x_train, y_train, DATASET_FILE)

    print(f"Dataset ready. shape: {x_train.shape}, {y_train.shape}")

    # Model: load if requested and available, otherwise train
    if LOAD_MODEL:
        try:
            model = load_model(MODEL_FILE, device=device)
            train_loss = None
            print(f"Loaded model from {MODEL_FILE}")
        except Exception:
            print(f"Model file not found or failed to load. Training a new model and saving to {MODEL_FILE}...")
            model, train_loss = train_model(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                            modes1=modes1, modes2=modes2, width=width, device=device)
            # Save with informative name
            save_model(model, filepath=None, epochs=EPOCHS, n_samples=len(param_values), modes=(modes1,modes2), width=width)
    else:
        model, train_loss = train_model(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                        modes1=modes1, modes2=modes2, width=width, device=device)
        save_model(model, filepath=None, epochs=EPOCHS, n_samples=len(param_values), modes=(modes1,modes2), width=width)

    # Optional: plot training loss if available
    if train_loss is not None:
        plot_training_loss(train_loss, outdir=RESULTS_DIR, filename=f'training_loss_epochs{EPOCHS}_samples{len(param_values)}.png')

    # Generate animation and evaluate
    print("\nGenerating animation and evaluating errors...")
    generate_animation(model, val_a=3.1415, val_b=3.1415, res_high=256, device=device, outdir=RESULTS_DIR)

    test_params = [1.234, 3.1415, 2.73]
    evaluate_errors(model, test_params, res_high=256, device=device, outdir=RESULTS_DIR,
                    filename=f'error_analysis_res256_samples{len(param_values)}.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
