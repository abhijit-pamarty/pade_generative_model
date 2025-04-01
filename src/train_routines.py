import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import os
from utils.custom_losses import BinnedSpectralPowerLoss3D
from config.load_configurations import load_exp_configs, load_path_configs, load_trainer_configs
torch.autograd.set_detect_anomaly(True)
import random

    
def train_model_2D(model, sample_data, Fs_data, sample_test_data, Fs_test_data):
    
    
    experiment = load_exp_configs()
    trainer = load_trainer_configs()
    paths = load_path_configs()

    # Reshape data into (num_samples * num_timesteps, X_size, Y_size)
    num_samples, num_X, num_Y = Fs_data.shape
    _, parameter_dim = sample_data.shape
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters())  , lr = float(experiment["model"]["optimizer"]["lr"]), weight_decay= float(experiment["model"]["optimizer"]["lambda_1"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = float(experiment["model"]["optimizer"]["gamma"]))
    
    x_left_lim = experiment["model"]["pade_layer"]["x_left_lim"]
    x_right_lim = experiment["model"]["pade_layer"]["x_right_lim"]
    y_left_lim = experiment["model"]["pade_layer"]["y_left_lim"]
    y_right_lim = experiment["model"]["pade_layer"]["y_right_lim"]

    # Convert to tensor and prepare DataLoader
    Fs_tensor = torch.from_numpy(Fs_data).float()  
    sample_tensor = torch.from_numpy(sample_data).float()

    Fs_test_tensor = torch.from_numpy(Fs_test_data).float()
    sample_test_tensor = torch.from_numpy(sample_test_data).float()

    num_test_samples, _, _ = Fs_test_tensor.shape

    
    Xs = torch.linspace(x_left_lim, x_right_lim, num_X)                      #X variable to create pade approximant
    Ys = torch.linspace(y_left_lim, y_right_lim, num_Y)                      #Y variable to create pade approximant
    
    Xs_grid, Ys_grid = torch.meshgrid(Xs, Ys)

    Xs_grid = Xs_grid.reshape(1, num_X*num_Y).unsqueeze(-1)
    Ys_grid = Ys_grid.reshape(1, num_X*num_Y).unsqueeze(-1)
    X = torch.stack((Xs_grid, Ys_grid), 0)
    
    if torch.cuda.is_available() and trainer["main_settings"]["use_CUDA"]:
        print("CUDA available")
        device = torch.device("cuda:0")  # Specify the GPU device
        Fs_tensor = Fs_tensor.to(device)
        sample_tensor = sample_tensor.to(device)
        Fs_test_tensor = Fs_test_tensor.to(device)
        sample_test_tensor = sample_test_tensor.to(device)
        X = X.to(device)
        #reshape to make time dimension same
    
    Fs_tensor = torch.reshape(Fs_tensor, (num_samples, num_X*num_Y))
    Fs_test_tensor = torch.reshape(Fs_test_tensor, (num_test_samples, num_X*num_Y))
    #sample_tensor = torch.reshape(sample_tensor, (num_samples*num_timesteps, parameter_dim))
    #LHSs_tensor = torch.reshape(LHSs_tensor, (num_samples*num_timesteps, num_X*num_Y,  num_X*num_Y))

    
    num_epochs = experiment["trainer"]["max_epochs"]
    batchsize = experiment["trainer"]["batch_size"]
    run = experiment["run"]

    root_dir = os.environ.get("GODELA_ROOT")
    model_dir = root_dir + paths["model_dir"]
    log_dir = root_dir + paths["log_dir"] + "experiments/"
    name = experiment["name"] + "_" + str(run)
    log_file_name = name + ".log"
    
    
    dataset = TensorDataset(Fs_tensor, sample_tensor)
    test_dataset = TensorDataset(Fs_test_tensor, sample_test_tensor)

    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=True)

    with open(os.path.join(log_dir, log_file_name), "w") as log_file:

        log_file.write("[DATA LOG FILE]\n")
        log_file.write("data name : " + name + "\n")
        log_file.write("number of samples : " + str(num_samples) + "\n")
        log_file.write("output X dimension : " + str(num_X) + "\n")
        log_file.write("output Y dimension : " + str(num_Y) + "\n")
        log_file.write("parameter dimension : " + str(parameter_dim) + "\n")
        log_file.write("learn rate : " + str(experiment["model"]["optimizer"]["lr"]) + "\n")
        log_file.write("gamma : " + str(experiment["model"]["optimizer"]["gamma"]) + "\n")
        log_file.write("batch size :" + str(batchsize) + "\n")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_test_loss = 0.0

            
            for Fs, sample in dataloader:
                
                # Forward pass
                prediction = model(sample, X)
                
                loss = criterion(prediction, Fs)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(),  experiment["trainer"]["gradient_clip_val"])
                
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()

            if epoch % 50 == 0:
                for Fs_test, sample_test in dataloader_test:

                    # Forward pass
                    prediction = model(sample_test, X, "online")
                    
                    loss_test = criterion(prediction, Fs_test)

                    # Accumulate loss
                    if loss_test > epoch_test_loss:
                        epoch_test_loss = loss_test.item()


            # Scheduler step every 1000 epochs
            if epoch % experiment["model"]["optimizer"]["scheduler_step"] == 0 and epoch > 0:
                scheduler.step()
            
            # Print epoch loss
            avg_train_loss = epoch_loss / len(dataloader)
            max_test_loss = epoch_test_loss

            log_message = f"Epoch {epoch+1}/{num_epochs}, Train loss: {avg_train_loss} \t maximum test loss: {max_test_loss}\n"
            log_file.write(log_message)
            
            if epoch%50 == 0:
                log_file.flush()
            
            # Save model periodically
            if epoch % trainer["save_settings"]["save_frequency"]  and trainer["save_settings"]["save"]:
                log_file.write("Saving model at epoch " + str(epoch +1) +" ...\n")
                torch.save(model.state_dict(), model_dir + "PGM_"+str(name)+"_"+str(run)+"_"+str(epoch+1)+".pth")



def train_model_4D(model, sample_data, Fs_data, sample_test_data, Fs_test_data):
    
    
    experiment = load_exp_configs()
    trainer = load_trainer_configs()
    paths = load_path_configs()

    # Reshape data into (num_samples * num_timesteps, X_size, Y_size)
    num_samples, num_T, num_X, num_Y, num_Z = Fs_data.shape
    _, parameter_dim = sample_data.shape
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters())  , lr = float(experiment["model"]["optimizer"]["lr"]), weight_decay= float(experiment["model"]["optimizer"]["lambda_1"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = float(experiment["model"]["optimizer"]["gamma"]))
    
    x_left_lim = experiment["model"]["pade_layer"]["x_left_lim"]
    x_right_lim = experiment["model"]["pade_layer"]["x_right_lim"]
    y_left_lim = experiment["model"]["pade_layer"]["y_left_lim"]
    y_right_lim = experiment["model"]["pade_layer"]["y_right_lim"]
    z_left_lim = experiment["model"]["pade_layer"]["z_left_lim"]
    z_right_lim = experiment["model"]["pade_layer"]["z_right_lim"]
    t_left_lim = experiment["model"]["pade_layer"]["t_left_lim"]
    t_right_lim = experiment["model"]["pade_layer"]["t_right_lim"]

    # Convert to tensor and prepare DataLoader
    Fs_tensor = torch.from_numpy(Fs_data).float()  
    sample_tensor = torch.from_numpy(sample_data).float()

    Fs_test_tensor = torch.from_numpy(Fs_test_data).float()
    sample_test_tensor = torch.from_numpy(sample_test_data).float()

    num_test_samples, _, _, _, _ = Fs_test_tensor.shape

    
    Xs = torch.linspace(x_left_lim, x_right_lim, num_X)                      #X variable to create pade approximant
    Ys = torch.linspace(y_left_lim, y_right_lim, num_Y)                      #Y variable to create pade approximant
    Zs = torch.linspace(z_left_lim, z_right_lim, num_Z)                      #Z variable to create pade approximant
    Ts = torch.linspace(t_left_lim, t_right_lim, num_T)                      #T variable to create pade approximant

    Ts_grid, Xs_grid, Ys_grid, Zs_grid = torch.meshgrid(Ts, Xs, Ys, Zs)

    Xs_grid = Xs_grid.reshape(1, num_X*num_Y*num_Z*num_T).unsqueeze(-1)
    Ys_grid = Ys_grid.reshape(1, num_X*num_Y*num_Z*num_T).unsqueeze(-1)
    Zs_grid = Zs_grid.reshape(1, num_X*num_Y*num_Z*num_T).unsqueeze(-1)
    Ts_grid = Zs_grid.reshape(1, num_X*num_Y*num_Z*num_T).unsqueeze(-1)
    X = torch.stack((Ts_grid, Xs_grid, Ys_grid, Zs_grid), 0)
    
    if torch.cuda.is_available() and trainer["main_settings"]["use_CUDA"]:
        print("CUDA available")
        device = torch.device("cuda:0")  # Specify the GPU device
        Fs_tensor = Fs_tensor.to(device)
        sample_tensor = sample_tensor.to(device)
        Fs_test_tensor = Fs_test_tensor.to(device)
        sample_test_tensor = sample_test_tensor.to(device)
        X = X.to(device)
        #reshape to make time dimension same
    
    Fs_tensor = torch.reshape(Fs_tensor, (num_samples, num_X*num_Y*num_Z*num_T))
    Fs_test_tensor = torch.reshape(Fs_test_tensor, (num_test_samples, num_X*num_Y*num_Z*num_T))
    #sample_tensor = torch.reshape(sample_tensor, (num_samples*num_timesteps, parameter_dim))
    #LHSs_tensor = torch.reshape(LHSs_tensor, (num_samples*num_timesteps, num_X*num_Y,  num_X*num_Y))

    
    num_epochs = experiment["trainer"]["max_epochs"]
    batchsize = experiment["trainer"]["batch_size"]
    run = experiment["run"]

    root_dir = os.environ.get("GODELA_ROOT")
    model_dir = root_dir + paths["model_dir"]
    log_dir = root_dir + paths["log_dir"] + "experiments/"
    name = experiment["name"] + "_" + str(run)
    log_file_name = name + ".log"
    
    
    dataset = TensorDataset(Fs_tensor, sample_tensor)
    test_dataset = TensorDataset(Fs_test_tensor, sample_test_tensor)

    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=True)

    with open(os.path.join(log_dir, log_file_name), "w") as log_file:

        log_file.write("[DATA LOG FILE]\n")
        log_file.write("data name : " + name + "\n")
        log_file.write("number of samples : " + str(num_samples) + "\n")
        log_file.write("output X dimension : " + str(num_X) + "\n")
        log_file.write("output Y dimension : " + str(num_Y) + "\n")
        log_file.write("output Z dimension : " + str(num_Z) + "\n")
        log_file.write("output T dimension : " + str(num_T) + "\n")
        log_file.write("parameter dimension : " + str(parameter_dim) + "\n")
        log_file.write("learn rate : " + str(experiment["model"]["optimizer"]["lr"]) + "\n")
        log_file.write("gamma : " + str(experiment["model"]["optimizer"]["gamma"]) + "\n")
        log_file.write("batch size :" + str(batchsize) + "\n")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_test_loss = 0.0

            collocation_points = generate_collocation_points(num_X*num_Y*num_Z*num_T)
            X_collocated = X[:, :, collocation_points, :]

            for Fs, sample in dataloader:
                
                Fs_collocated = Fs[:, collocation_points]
                # Forward pass
                prediction = model(sample, X_collocated)
                
                loss = criterion(prediction, Fs_collocated)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(),  experiment["trainer"]["gradient_clip_val"])
                
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()

            if epoch % 50 == 0:
                for Fs_test, sample_test in dataloader_test:

                    Fs_test = Fs_test[:, collocation_points]

                    # Forward pass
                    prediction = model(sample_test, X_collocated, "online")
                    
                    loss_test = criterion(prediction, Fs_test)

                    # Accumulate loss
                    if loss_test > epoch_test_loss:
                        epoch_test_loss = loss_test.item()


            # Scheduler step every 1000 epochs
            if epoch % experiment["model"]["optimizer"]["scheduler_step"] == 0 and epoch > 0:
                scheduler.step()
            
            # Print epoch loss
            avg_train_loss = epoch_loss / len(dataloader)
            max_test_loss = epoch_test_loss

            log_message = f"Epoch {epoch+1}/{num_epochs}, Train loss: {avg_train_loss} \t maximum test loss: {max_test_loss}\n"
            log_file.write(log_message)
            
            if epoch%50 == 0:
                log_file.flush()
            
            # Save model periodically
            if epoch % trainer["save_settings"]["save_frequency"] == 0 and trainer["save_settings"]["save"]:
                log_file.write("Saving model at epoch " + str(epoch +1) +" ...\n")
                torch.save(model.state_dict(), model_dir + "PGM_"+str(name)+"_"+str(run)+"_"+str(epoch+1)+".pth")

def train_model_3D(model, sample_data, Fs_data, sample_test_data, Fs_test_data):
    
    
    experiment = load_exp_configs()
    trainer = load_trainer_configs()
    paths = load_path_configs()

    # Reshape data into (num_samples * num_timesteps, X_size, Y_size)
    num_samples, num_X, num_Y, num_Z = Fs_data.shape
    _, parameter_dim = sample_data.shape
    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters())  , lr = float(experiment["model"]["optimizer"]["lr"]), weight_decay= float(experiment["model"]["optimizer"]["lambda_1"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = float(experiment["model"]["optimizer"]["gamma"]))
    
    x_left_lim = experiment["model"]["pade_layer"]["x_left_lim"]
    x_right_lim = experiment["model"]["pade_layer"]["x_right_lim"]
    y_left_lim = experiment["model"]["pade_layer"]["y_left_lim"]
    y_right_lim = experiment["model"]["pade_layer"]["y_right_lim"]
    z_left_lim = experiment["model"]["pade_layer"]["z_left_lim"]
    z_right_lim = experiment["model"]["pade_layer"]["z_right_lim"]

    # Convert to tensor and prepare DataLoader
    Fs_tensor = torch.from_numpy(Fs_data).float()  
    sample_tensor = torch.from_numpy(sample_data).float()

    Fs_test_tensor = torch.from_numpy(Fs_test_data).float()
    sample_test_tensor = torch.from_numpy(sample_test_data).float()

    num_test_samples, _, _, _ = Fs_test_tensor.shape

    
    Xs = torch.linspace(x_left_lim, x_right_lim, num_X)                      #X variable to create pade approximant
    Ys = torch.linspace(y_left_lim, y_right_lim, num_Y)                      #Y variable to create pade approximant
    Zs = torch.linspace(z_left_lim, z_right_lim, num_Z)                      #Z variable to create pade approximant
    
    Xs_grid, Ys_grid, Zs_grid = torch.meshgrid(Xs, Ys, Zs)

    Xs_grid = Xs_grid.reshape(1, num_X*num_Y*num_Z).unsqueeze(-1)
    Ys_grid = Ys_grid.reshape(1, num_X*num_Y*num_Z).unsqueeze(-1)
    Zs_grid = Zs_grid.reshape(1, num_X*num_Y*num_Z).unsqueeze(-1)
    X = torch.stack((Xs_grid, Ys_grid, Zs_grid), 0)
    
    if torch.cuda.is_available() and trainer["main_settings"]["use_CUDA"]:
        print("CUDA available")
        device = torch.device("cuda:0")  # Specify the GPU device
        Fs_tensor = Fs_tensor.to(device)
        sample_tensor = sample_tensor.to(device)
        Fs_test_tensor = Fs_test_tensor.to(device)
        sample_test_tensor = sample_test_tensor.to(device)
        X = X.to(device)
        #reshape to make time dimension same
    
    Fs_tensor = torch.reshape(Fs_tensor, (num_samples, num_X*num_Y*num_Z))
    Fs_test_tensor = torch.reshape(Fs_test_tensor, (num_test_samples, num_X*num_Y*num_Z))
    #sample_tensor = torch.reshape(sample_tensor, (num_samples*num_timesteps, parameter_dim))
    #LHSs_tensor = torch.reshape(LHSs_tensor, (num_samples*num_timesteps, num_X*num_Y,  num_X*num_Y))

    
    num_epochs = experiment["trainer"]["max_epochs"]
    batchsize = experiment["trainer"]["batch_size"]
    run = experiment["run"]

    root_dir = os.environ.get("GODELA_ROOT")
    model_dir = root_dir + paths["model_dir"]
    log_dir = root_dir + paths["log_dir"] + "experiments/"
    name = experiment["name"] + "_" + str(run)
    log_file_name = name + ".log"
    
    
    dataset = TensorDataset(Fs_tensor, sample_tensor)
    test_dataset = TensorDataset(Fs_test_tensor, sample_test_tensor)

    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=True)

    with open(os.path.join(log_dir, log_file_name), "w") as log_file:

        log_file.write("[DATA LOG FILE]\n")
        log_file.write("data name : " + name + "\n")
        log_file.write("number of samples : " + str(num_samples) + "\n")
        log_file.write("output X dimension : " + str(num_X) + "\n")
        log_file.write("output Y dimension : " + str(num_Y) + "\n")
        log_file.write("output Z dimension : " + str(num_Z) + "\n")
        log_file.write("parameter dimension : " + str(parameter_dim) + "\n")
        log_file.write("learn rate : " + str(experiment["model"]["optimizer"]["lr"]) + "\n")
        log_file.write("gamma : " + str(experiment["model"]["optimizer"]["gamma"]) + "\n")
        log_file.write("batch size :" + str(batchsize) + "\n")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_test_loss = 0.0
           
            for Fs, sample in dataloader:

                # Forward pass
                prediction = model(sample, X)

                prediction = prediction.reshape(batchsize, num_X, num_Y, num_Z)
                Fs = Fs.reshape(batchsize, num_X, num_Y, num_Z)
                
                loss = criterion(prediction, Fs)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(),  experiment["trainer"]["gradient_clip_val"])
                
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()

            if epoch % 50 == 0:
                for Fs_test, sample_test in dataloader_test:

                    

                    # Forward pass
                    prediction = model(sample_test, X, "online")

                    prediction = prediction.reshape(1, num_X, num_Y, num_Z)
                    Fs_test = Fs_test.reshape(1, num_X, num_Y, num_Z)
                    
                    loss_test = criterion(prediction, Fs_test)

                    # Accumulate loss
                    if loss_test > epoch_test_loss:
                        epoch_test_loss = loss_test.item()


            # Scheduler step every 1000 epochs
            if epoch % experiment["model"]["optimizer"]["scheduler_step"] == 0 and epoch > 0:
                scheduler.step()
            
            # Print epoch loss
            avg_train_loss = epoch_loss / len(dataloader)
            max_test_loss = epoch_test_loss

            log_message = f"Epoch {epoch+1}/{num_epochs}, Train loss: {avg_train_loss} \t maximum test loss: {max_test_loss}\n"
            log_file.write(log_message)
            
            if epoch%50 == 0:
                log_file.flush()
            
            # Save model periodically
            if (epoch + 1) % trainer["save_settings"]["save_frequency"] == 0 and trainer["save_settings"]["save"]:
                log_file.write("Saving model at epoch " + str(epoch +1) +" ...\n")
                torch.save(model.state_dict(), model_dir + "PGM_"+str(name)+"_"+str(run)+"_"+str(epoch+1)+".pth")


def generate_collocation_points(num_data, num_points = 32768):

    lower_bound = 0
    upper_bound = num_data - 1
    random_points = [random.randint(lower_bound, upper_bound) for _ in range(num_points)]

    return random_points