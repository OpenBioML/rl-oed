## 1. Get your w and b api key
The weights and bias api key can be found by logging into the rl-oed team here: https://stability.wandb.io/rl-oed. 
Make sure you get the api key associated with the rl-oed team and not your personal one, otherwise experiments will be 
logged in the wrong place. You will need access to the stability cluster first, message NeythenT on discord to get help 
with this

## 2. Set the WANDB_API_KEY login variable
Set the WANDB_API_KEY environment variable to your api key by running 
```
$ export WANDB_API_KEY <YOUR API KEY>
```
from the command line or 
```python
os.environ["WANDB_API_KEY"] = "<YOUR API KEY>"
```
from Python

## Login to w and b 
To log in from command line
```
$ wandb login --host=https://stability.wandb.io
```
or in a python script
```python
wandb.login(host='https://stability.wandb.io', relogin=False)
```

## Running automated slurm jobs
I suggest we add the following lines to the job script that gets pushed to the github and people just copy their api 
keys in. 
```
$ export WANDB_API_KEY <YOUR API KEY>
$ wandb login --host=https://stability.wandb.io
```
