This repository contains the source code of the paper entitled:

> **Leveraging PAC-Bayes Theory and Gibbs Distributions for Generalization Bounds with Complexity Measures**<br/>
> Paul Viallard, Rémi Emonet, Amaury Habrard, Emilie Morvant, Valentina Zantedeschi<br/>
> AISTATS, 2024

### Running the experiments

To reproduce the experiments, you have to execute the following commands in your bash shell.

**Generating the data:**
> python run.py local generate_data.ini

**Learning the models to generate the dataset for the neural complexity:**
> python run.py local learn_data_neural_comp.ini<br/>
> python run.py local merge_data_neural_comp.ini

**Learning the neural complexity:**
> python run.py local learn_neural_comp.ini

**Computing the bounds with complexity measures:**
> python run.py local learn_comp_fig_1.ini<br/>
> python run.py local learn_comp_fig_2.ini<br/>
> python run.py local learn_comp_fig_3.ini

**Generating the plots:**
> python run.py local generate_plot.ini

### Conda environment

The code was tested with the conda environment in the env.yml.
