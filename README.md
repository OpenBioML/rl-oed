# RED
[Deep Reinforcement Learning for Optimal Experimental Design in Biology](https://www.biorxiv.org/content/10.1101/2022.05.09.491138.abstract)

### Installation

RED does not need to be installed to run the examples 

To use the package within python scropts, `RED` must be in PYTHONPATH.

To add to PYTHONPATH on a bash system add the following to the ~/.bashrc file

```console
export PYTHONPATH="${PYTHONPATH}:<path to RED root dir>"
```

### Dependencies
Standard python dependencies are required: `numpy`, `scipy`, `matplotlib`.  `TensorFlow` is required). Instructions for installing 'TensorFlow' can be found here:
 https://www.tensorflow.org/install/

### User Instructions
Code files can be imported into scripts, ensure the RED directory is in PYTHONPATH and simply import the required RED classes. See examples.

To run examples found in RED_master/examples from the command line, e.g.:

```console
$ python train_RT3D_prior.py 
```

The examples will automatically save some results in the directory:


The main classes are the continuous_agents and OED_env, see examples for how to use these:

### continuous_agents
The continuous_agents.py file can be imported and used on any RL task.
```console
from RED.agents.continuous_agents import RT3D_agent
```

### OED_env
Contains the environments used for RL for OED. Can be imported and initialised with any system goverened by a set of DEs

```console
from RED.environments.OED_env import OED_env
```
