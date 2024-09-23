# Poros

Poros is an automatic and adaptive resource partitioner 
for spatial accelerators with meta reinforcement learning.
This repository contains the source code for Poros.

### Setup ###
* Download the Poros source code 
```
git clone https://github.com/FuyuWang/Poros.git
```
* Create virtual environment through anaconda
```
conda create --name PorosEnv python=3.8
conda activate PorosEnv
```
* Install packages
   
```
pip install -r requirements.txt
```

* Install [MAESTRO](https://github.com/maestro-project/maestro.git)
```
mkdir cost_model
python build.py
```

### Run Poros ###

```
sh run.sh 
```

