#second version of the pade generative model

import torch
import numpy as np
from src.pade_generative_model import Pade_Generative_Model_2D
from src.train_routines import train_model_3D
from config.load_configurations import load_exp_configs, load_path_configs, load_trainer_configs
import os

#Main function
if __name__ == "__main__":
    
    experiment = load_exp_configs()
    trainer = load_trainer_configs()
    paths = load_path_configs()

    root_dir = os.environ.get("GODELA_ROOT")
    model_dir = os.path.join(root_dir, paths["model_dir"].lstrip("/\\"))
    data_dir = os.path.join(root_dir, paths["data_dir"].lstrip("/\\"))

    run_name = "PGM_navier_stokes_equation_3D_1_1_30001.pth"
    model_to_load = os.path.join(model_dir, run_name)

    Fs_data_file = os.path.join(data_dir, experiment["model"]["data"]["Fs_train_data"].lstrip("/\\"))              #dataset variable for the latent space (outputs)
    sample_data_file = os.path.join(data_dir, experiment["model"]["data"]["sample_train_data"].lstrip("/\\"))                   #dataset variable for the sample data (inputs)
    
    Fs_test_data_file = os.path.join(data_dir, experiment["model"]["data"]["Fs_test_data"].lstrip("/\\"))              #dataset variable for the latent space (outputs)
    sample_test_data_file = os.path.join(data_dir, experiment["model"]["data"]["sample_test_data"].lstrip("/\\"))                    #dataset variable for the sample data (inputs)

    restart_training = trainer["main_settings"]["restart_training"]
    use_CUDA =  trainer["main_settings"]["use_CUDA"]

    print("Loading dataset and sample dataset...")
    Fs_data = np.load(Fs_data_file).astype(np.float32)
    max_Fs_data = np.max(Fs_data)
    min_Fs_data = np.min(Fs_data)
    Fs_data = Fs_data/max_Fs_data
    sample_data = np.load(sample_data_file).astype(np.float32)
    _, parameter_dim = sample_data.shape
    
    Fs_test_data = np.load(Fs_test_data_file).astype(np.float32)
    Fs_test_data = Fs_test_data/max_Fs_data
    sample_test_data = np.load(sample_test_data_file).astype(np.float32)
    #create pade generative model
    model = Pade_Generative_Model_2D(parameter_dim)
    
    if torch.cuda.is_available() and use_CUDA:
        device = torch.device("cuda:0")  # Specify the GPU device
        model = model.to(device)


    if (restart_training):
        print("Starting training with restart...\n")
        model.load_state_dict(torch.load(model_to_load))
        train_model_3D(model, sample_data, Fs_data, sample_test_data, Fs_test_data)
    else:
        print("Starting training...\n")
        train_model_3D(model, sample_data, Fs_data, sample_test_data, Fs_test_data)

    