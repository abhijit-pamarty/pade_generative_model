import yaml
import os

def load_trainer_configs():

    root_dir = os.environ.get("GODELA_ROOT")

    trainer_config_path = root_dir+"\\config\\trainer\\train_settings.yaml"

    #load the trainer variables
    with open(trainer_config_path, "r") as stream:
        try:
            trainer = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    return trainer

def load_path_configs():

    root_dir = os.environ.get("GODELA_ROOT")

    path_config_path = root_dir+"\\config\\paths\\default.yaml"
    #load the path variables
    with open(path_config_path, "r") as stream:
        try:
            paths = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    return paths

def load_exp_configs():


    root_dir = os.environ.get("GODELA_ROOT")
    experiment_config_path = root_dir+"\\config\\experiment\\experiment.yaml"
    #load the experiment variables
    with open(experiment_config_path, "r") as stream:
        try:
            experiment = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    return experiment