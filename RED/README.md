## Configuration structure

We use [Hydra](https://hydra.cc) for configuration, which allows for **hierarchical configuration structure composable from multiple sources** and dynamic command line overrides. The folder structure is as follows:
```
|── configs
|   |── environment                         # environment-specific parameters
|   |   |── chemostat.yaml
|   |   |── gene_transcription.yaml
|   |   └── ...
|   |── example                             # configurations for examples in the root folder `examples`
|   |   |── FQ_chemostat.yaml
|   |   |── FQ_gene_transcription.yaml
|   |   └── ...
|   └── model                               # configuration files for different models and agents
|       |── RT3D_agent.yaml
|       └── ...
└── train.yaml                              # main training configuration
```

## How to use
The file `configs/train.yaml` contains the main training configuration that can be altered to fit any experiments. It is an entry point that composes smaller configuration pieces such as the environment (chemostat, gene transcription, ...), the model/agent (RT3D, DRPG, ...), and higher-level settings such as the path where to store the results. As can be seen below, one can also easily override selected parameters in the above-mentioned config sources (see the section `model:`).

<details>
<summary><b>Training config (configs/train.yaml)</b></summary>

```yaml
defaults:
  - /environment: chemostat
  - /model: RT3D_agent
  - _self_

model:
  val_learning_rate: 0.0001
  pol_learning_rate: 0.00005
  policy_act: sigmoid
  noise_bounds: [-0.25, 0.25]
  action_bounds: [0, 1]

hidden_layer_size: [[64, 64], [128, 128]]
policy_delay: 2
max_std: 1
explore_rate: "${max_std}"
save_path: results/
```
</details>

### Using in Python scripts
The following python script example will load defined training config from `.RED/configs/train.yaml`, and then print it. In the function `main`, one can then work with the `config` as with standard Python dictionary, although it is an instance of OmegaConf's DictConfig. You can read more about it in the [Hydra documentation](https://hydra.cc/docs/tutorials/basic/your_first_app/using_config/) or in the [OmegaConf documentation](https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-manipulation). Another example can be found in `examples/Figure_4_RT3D_chemostat/train_RT3D.py`.

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="./RED/configs", config_name="train")
def main(config: DictConfig) -> None:
    print(config)

if __name__ == "__main__":
    main()
```

### Using in Jupyter Notebooks
Including the following at the beginning of a jupyter notebook will initialize Hydra, load defined training config, and then print it.

```python
from hydra import compose, initialize
from omegaconf import OmegaConf

initialize(version_base=None, config_path="./RED/configs")
config = compose(config_name="train")
print(OmegaConf.to_yaml(config))
```

When initializing hydra it is possible to override any of the default assignments. 
Here is an example of overriding batch_size and seed while initializing hydra:

```python
from hydra import compose, initialize
from omegaconf import OmegaConf

initialize(version_base=None, config_path="./RED/configs")
config = compose(overrides=["save_path=experiment_2_results/", "model.pol_learning_rate=0.0001"])
print(OmegaConf.to_yaml(config))
```

The following link to hydra documentation provides more information on override syntax: <br/>
https://hydra.cc/docs/advanced/override_grammar/basic/ <br/>

For more information regarding hydra initialization in jupyter see the following link:
https://github.com/facebookresearch/hydra/blob/main/examples/jupyter_notebooks/compose_configs_in_notebook.ipynb
