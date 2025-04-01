#second version of the pade generative model
import torch
import torch.nn as nn
import torch.nn.functional as f
from src.pade_layers import Pade_Layer
from config.load_configurations import load_exp_configs

class Pade_Generative_Model_2D(nn.Module):
    
    def __init__(self, parameter_dim):
        
        super(Pade_Generative_Model_2D, self).__init__()

        self.exp_configs = load_exp_configs()
        
        #FC layers for the weighting layer
        
        fc_1 = self.exp_configs["model"]["pade_generative_model"]["num_model_fc_1"]
        fc_2 = self.exp_configs["model"]["pade_generative_model"]["num_model_fc_2"]
        fc_3 = self.exp_configs["model"]["pade_generative_model"]["num_model_fc_3"]
        

        #pade layers

        self.num_pade_layers = self.exp_configs["model"]["pade_generative_model"]["num_pade_layers"]
        self.pade_layers = nn.ModuleList(Pade_Layer(parameter_dim) for layer_index in range(self.num_pade_layers))
        
        
        #weighting layers
        
        self.wl1 = nn.Linear(in_features= parameter_dim, out_features= fc_1)
        self.wl2 = nn.Linear(in_features= fc_1, out_features= fc_2)
        self.wl3 = nn.Linear(in_features= fc_2, out_features= fc_3)
        self.wl4 = nn.Linear(in_features= fc_3, out_features= self.num_pade_layers)
        
        
    def forward(self, x, X, mode = "train"):
        
        #weight layers
        WL_1 = f.leaky_relu(self.wl1(x))
        WL_2 = f.leaky_relu(self.wl2(WL_1))
        WL_3 = f.leaky_relu(self.wl3(WL_2))
        weights = f.softmax(self.wl4(WL_3))

        #pade layers

        solution_fn = 0
        for layer_index in range(self.num_pade_layers):

            output = self.pade_layers[layer_index](x, X, mode).squeeze()
            

            weight = weights[:, layer_index].unsqueeze(-1)

            if layer_index == 0:
                solution_fn = (weight*output)
            else:
                solution_fn = solution_fn + weight*output

            

        return solution_fn

