import numpy as np
import scipy.stats.qmc as qmc
import data_factory.synthetic_data as dg
from config.load_configurations import load_path_configs
import os

def generate_dataset_2D():
    
    paths = load_path_configs()
    root_dir = os.environ.get("GODELA_ROOT")
    data_dir = root_dir + paths["data_dir"]
    log_dir = root_dir + paths["log_dir"] + "data/"
    
    name = "burgers_equation_2"
    type = "train"

    num_samples = 100
    M = 100
    
    min_nu = -5
    max_nu = 0
    min_mu = 1
    max_mu = 2
    param_dimension = 2
    
    sampler = qmc.LatinHypercube(d = param_dimension)
    sample = sampler.random(n = num_samples)

    sample[:, 0] = 10**((max_nu - min_nu)*sample[:, 0] + min_nu)
    sample[:, 1] = (max_mu - min_mu)*sample[:, 1] + min_mu
    
    Fs = np.zeros((num_samples, M , M))
    F = np.zeros((M, M))
    
    sample_idx = 0
    while sample_idx < num_samples:
        
        nu = sample[sample_idx, 0]
        mu = sample[sample_idx, 1]
        
        F = dg.generate_burgers_velocity(nu, mu, M)
        Fs[sample_idx, :, :] = F
        
        sample_idx += 1

    Fs_name = 'Fs_' + type + '_' + name +'.npy'
    samples_name = 'sample_data_' + type + '_' + name +'.npy'
    log_file_name = name + '_' + type + '.log'

    np.save(data_dir + Fs_name, Fs)
    np.save(data_dir + samples_name, sample)
    
    with open(log_dir + log_file_name, "w") as file:

        file.write("[DATA LOG FILE]\n")
        file.write("data name : " + name + "\n")
        file.write("data type : " + type + "\n")
        file.write("number of samples : " + str(num_samples) + "\n")
        file.write("output dimension : " + str(M) + "\n")
        file.write("parameter dimension : " + str(param_dimension) + "\n")
        file.write("input filename : " + samples_name + "\n")
        file.write("output filename : " + Fs_name + "\n")

def generate_dataset_3D():
    
    paths = load_path_configs()
    root_dir = os.environ.get("GODELA_ROOT")
    data_dir = root_dir + paths["data_dir"]
    log_dir = root_dir + paths["log_dir"] + "data/"
    
    name = "navier_stokes_3D"
    type = "train"

    num_samples = 32
    M = 32
    
    min_A = 0.2
    max_A = 0.4
    min_B = 0.2
    max_B = 0.4
    param_dimension = 2
    
    sampler = qmc.LatinHypercube(d = param_dimension)
    sample = sampler.random(n = num_samples)

    sample[:, 0] = (max_A - min_A)*sample[:, 0] + min_A
    sample[:, 1] = (max_B - min_B)*sample[:, 1] + min_B
    
    Fs = np.zeros((num_samples, M , M, M))
    F = np.zeros((M, M, M))
    
    sample_idx = 0
    while sample_idx < num_samples:
        
        A = sample[sample_idx, 0]
        B = sample[sample_idx, 1]
        
        F = dg.generate_3D_navier_stokes(M, A, B)
        Fs[sample_idx, :, :, :] = F
        
        sample_idx += 1

    Fs_name = 'Fs_' + type + '_' + name +'.npy'
    samples_name = 'sample_data_' + type + '_' + name +'.npy'
    log_file_name = name + '_' + type + '.log'

    np.save(data_dir + Fs_name, Fs)
    np.save(data_dir + samples_name, sample)
    
    with open(log_dir + log_file_name, "w") as file:

        file.write("[DATA LOG FILE]\n")
        file.write("data name : " + name + "\n")
        file.write("data type : " + type + "\n")
        file.write("number of samples : " + str(num_samples) + "\n")
        file.write("output dimension : " + str(M) + "\n")
        file.write("parameter dimension : " + str(param_dimension) + "\n")
        file.write("input filename : " + samples_name + "\n")
        file.write("output filename : " + Fs_name + "\n")

def generate_dataset_4D():
    
    paths = load_path_configs()
    root_dir = os.environ.get("GODELA_ROOT")
    data_dir = root_dir + paths["data_dir"]
    log_dir = root_dir + paths["log_dir"] + "data/"
    
    name = "navier_stokes_4D"
    type = "test"

    num_samples = 16
    M = 32
    
    min_A = 0.5
    max_A = 1
    min_B = 0.5
    max_B = 1
    param_dimension = 2
    
    sampler = qmc.LatinHypercube(d = param_dimension)
    sample = sampler.random(n = num_samples)

    sample[:, 0] = (max_A - min_A)*sample[:, 0] + min_A
    sample[:, 1] = (max_B - min_B)*sample[:, 1] + min_B
    
    Fs = np.zeros((num_samples, M , M, M, M))
    F = np.zeros((M, M, M, M))
    
    sample_idx = 0
    while sample_idx < num_samples:
        
        A = sample[sample_idx, 0]
        B = sample[sample_idx, 1]
        
        F = dg.generate_3D_navier_stokes(M, A, B)
        Fs[sample_idx, :, :, :, :] = F
        
        sample_idx += 1

    Fs_name = 'Fs_' + type + '_' + name +'.npy'
    samples_name = 'sample_data_' + type + '_' + name +'.npy'
    log_file_name = name + '_' + type + '.log'

    np.save(data_dir + Fs_name, Fs)
    np.save(data_dir + samples_name, sample)
    
    with open(log_dir + log_file_name, "w") as file:

        file.write("[DATA LOG FILE]\n")
        file.write("data name : " + name + "\n")
        file.write("data type : " + type + "\n")
        file.write("number of samples : " + str(num_samples) + "\n")
        file.write("output dimension : " + str(M) + "\n")
        file.write("parameter dimension : " + str(param_dimension) + "\n")
        file.write("input filename : " + samples_name + "\n")
        file.write("output filename : " + Fs_name + "\n")
