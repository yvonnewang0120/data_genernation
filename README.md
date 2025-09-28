# FJSP-DRL

This repository is the data generation implementation of the paper “[Cyber-Physical Internet Enabled Hierarchical Attention Network based Reinforcement Learning for Order Dispatch in Fast Fashion Manufacturing]”.

## Quick Start


### introduction

- `data` saves the instance files including testing instances (in the subfolder `SD1` and `SD2`) and validation instances (in the subfolder `data_train_vali`) .
- `common_utils.py` contains some useful functions (including the implementation of priority dispatching rules mentioned in the paper) .
- `data_utils.py` is used for data generation, reading and format conversion.
- `SDST_utiles.py` is used for setup time data generation, reading and format conversion.
- `params.py` defines parameters settings.


### generation

```python
# first
python data_utils.py # generation processing time data on 10x5 FJSP instances using SD2
# second
python SDST_utils.py # generation setup time data on 10x5 FJSP instances using SD2

# options (Validation instances of corresponding size should be prepared in ./data/data_train_vali/{data_source})
python data_utils.py 	--n_j 10		# number of jobs for training/validation instances
--n_m 5			      # number of machines for training/validation instances
--data_source SD2	# data source (SD1 / SD2)
--data_suffix mix	# mode for SD2 data generation
            					# 'mix' is thedefault mode as defined in the paper
```

## Reference

- https://github.com/wrqccc/FJSP-DRL
- https://github.com/songwenas12/fjsp-drl/

