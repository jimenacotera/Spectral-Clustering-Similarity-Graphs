The code in this repository is based on https://github.com/pmacg/spectral-clustering-meta-graphs, the code to reproduce the results in the paper "[A Tighter Analysis of Spectral Clustering, and Beyond](https://arxiv.org/abs/2208.01724)", published in ICML 2022.

*****add that my code jsut runs for one eigenvector differing from their code


## Preparing your environment
Our code is primarily written in Python 3. There is also a matlab
script for analysing the results of the BSDS experiment.

We recommend running the python code inside a virtual environment.

To install the dependencies of the project, run

```bash
pip install -r requirements.txt
```

If you would like to run the experiments on the BSDS dataset, you should untar the data file
in the `data/bsds` directory. On Linux, this is done with the following commands.

```bash
cd data/bsds
tar -xvf BSR_bsds500.tgz
```

## Running all the experiments
To run all of the experiments, use the following command

```bash
python runAllExperiments.py
```

To add additional experiments or change the configuration of existing ones, modify the ``experiment_configurations.yaml`` file. 


**Please note that the full BSDS experiment is quite resource-intensive, and we recommend running on a compute server.**


## Running select experiments
You can instead choose to run the BSDS experiment on only one of the images from the dataset using the following command.

```bash
python experiments.py bsds {similaritygrahp}{bsds_image_id}
```

where ``{similaritygraph}`` can be any of the following: 
- fcn-rbf-{variance}
- fcn-lpl-{variance}
- fcn-inv
- knn{valueofk}

For example:

```bash
python experiment.py bsds fcn-rbf-10 2018
```

Results can then be generated using the following command: 

```bash 
python analyseBSDSexperiments.py {experimentname}
```

where ``{experimentname}`` is the name that will be used for the results directory and csv file.

## Output
The output from the experiments will be in the `results` directory, under the appropriate experiment name. Analysis results will be available under the `results/bsds/csv_results` directory and visualisations of the experiments and ground truth are available under `results/bsds/visualisations`.


## Reference

```bibtex
@InProceedings{pmlr-v162-macgregor22a,
  title = 	 {A Tighter Analysis of Spectral Clustering, and Beyond},
  author =       {Macgregor, Peter and Sun, He},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {14717--14742},
  year = 	 {2022},
  volume = 	 {162},
  publisher =    {PMLR},
}
```
