#second version of the pade generative model

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.pade_generative_model import Pade_Generative_Model_2D
from config.load_configurations import load_exp_configs, load_path_configs, load_trainer_configs
import os
from utils.postprocessing import plot_errors, plot_isosurfaces_3D

#Main function
if __name__ == "__main__":

    experiment = load_exp_configs()
    trainer = load_trainer_configs()
    paths = load_path_configs()

    root_dir = os.environ.get("GODELA_ROOT")
    model_dir = os.path.join(root_dir, paths["model_dir"].lstrip("/\\"))
    data_dir = os.path.join(root_dir, paths["data_dir"].lstrip("/\\"))

    run_name = "PGM_navier_stokes_equation_3D_3_3_40000.pth"
    model_to_load = os.path.join(model_dir, run_name)

    Fs_data_file = os.path.join(data_dir, experiment["model"]["data"]["Fs_train_data"].lstrip("/\\"))              #dataset variable for the latent space (outputs)
    
    print("Loading dataset and sample dataset...")
    Fs_data = np.load(Fs_data_file).astype(np.float32)
    max_Fs_data = np.max(Fs_data)
    
    Fs_test_data_file = os.path.join(data_dir, experiment["model"]["data"]["Fs_test_data"].lstrip("/\\"))              #dataset variable for the latent space (outputs)
    sample_test_data_file = os.path.join(data_dir, experiment["model"]["data"]["sample_test_data"].lstrip("/\\"))                    #dataset variable for the sample data (inputs)


    Fs_test_data = np.load(Fs_test_data_file).astype(np.float32)
    Fs_test_data = Fs_test_data/max_Fs_data
    sample_test_data = np.load(sample_test_data_file).astype(np.float32)
    num_test_samples, num_X, num_Y, num_Z = Fs_test_data.shape
    _, parameter_dim = sample_test_data.shape
    
    #create pade generative model
    model = Pade_Generative_Model_2D(parameter_dim)

    # Train the model
    print("Loading model...\n")
    model.load_state_dict(torch.load(model_to_load))

    Fs_test_tensor = torch.from_numpy(Fs_test_data).float()  
    sample_test_tensor = torch.from_numpy(sample_test_data).float()

    x_left_lim = experiment["model"]["pade_layer"]["x_left_lim"]
    x_right_lim = experiment["model"]["pade_layer"]["x_right_lim"]
    y_left_lim = experiment["model"]["pade_layer"]["y_left_lim"]
    y_right_lim = experiment["model"]["pade_layer"]["y_right_lim"]
    z_left_lim = experiment["model"]["pade_layer"]["z_left_lim"]
    z_right_lim = experiment["model"]["pade_layer"]["z_right_lim"]
    
    Xs = torch.linspace(x_left_lim, x_right_lim, num_X)                      #X variable to create pade approximant
    Ys = torch.linspace(y_left_lim, y_right_lim, num_Y)                      #Y variable to create pade approximant
    Zs = torch.linspace(z_left_lim, z_right_lim, num_Z)                      #Y variable to create pade approximant

    Xs_grid, Ys_grid, Zs_grid = torch.meshgrid(Xs, Ys, Zs)
    Xs_grid = Xs_grid.reshape(1, num_X*num_Y*num_Z).unsqueeze(-1)
    Ys_grid = Ys_grid.reshape(1, num_X*num_Y*num_Z).unsqueeze(-1)
    Zs_grid = Zs_grid.reshape(1, num_X*num_Y*num_Z).unsqueeze(-1)
    X = torch.stack((Xs_grid, Ys_grid, Zs_grid), 0)
        
    Fs_test_tensor = torch.reshape(Fs_test_tensor, (num_test_samples, num_X, num_Y, num_Z))
    #sample_tensor = torch.reshape(sample_tensor, (num_samples*num_timesteps, parameter_dim))
    #LHSs_tensor = torch.reshape(LHSs_tensor, (num_samples*num_timesteps, num_X*num_Y,  num_X*num_Y))

    dataset = TensorDataset(Fs_test_tensor, sample_test_tensor)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    errors = []
    As = []


    for Fs, sample in dataloader:
            
        # Forward pass
        prediction =  model(sample, X, "online")
        prediction = torch.reshape(prediction, (num_X, num_Y, num_Z))
        
        A = sample[0][0].detach().cpu().numpy()
        B = sample[0][1].detach().cpu().numpy()
        prediction_plot = prediction.detach().cpu().numpy()
        true_plot = Fs[0].detach().cpu().numpy()
        error_plot = np.abs(prediction_plot - true_plot)

        #fig = plot_isosurfaces_3D(prediction_plot, true_plot, error_plot)
        #fig.show()
        #input()
        
        error = np.mean(error_plot)
        errors.append(error)
        As.append(np.sqrt(A**2 + B**2))


    plot_errors(As, errors, "Error plot - Navier stokes equation", "$(A^2 + B^2)^{0.5}$")