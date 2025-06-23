import yaml
import os
import subprocess
import pandas as pd


with open('experiment_configurations.yaml') as f: 
    experiments = yaml.safe_load(f)['experiments']


results = pd.DataFrame()

for experiment in experiments: 
    experiment_name = experiment['name']
    data = experiment['data']
    sim_graph = experiment['sim_graph']
    # hyperparam_01 = experiment['hyperparam_01']

    # data = 'bsds'

    subprocess.run(["python3", "experiments.py", data, sim_graph])

    # run the evaluation 
    subprocess.run(["python3", "analyseBSDSExperiments.py", experiment_name ])

    # append evaluation to results with the name of the experiment and the configsss


# Save results to a csv with the experiment date