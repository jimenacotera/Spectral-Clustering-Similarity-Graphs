import yaml
import os
import subprocess


with open('experiment_configurations.yaml') as f: 
    experiments = yaml.safe_load(f)['experiments']


for experiment in experiments: 
    name = experiment['name']
    sim_matrix = experiment['sim_matrix']
    hyperparam_01 = experiment['hyperparam_01']

    data = 'bsds'

    subprocess.run(["python3", "experiments.py", data, sim_matrix])