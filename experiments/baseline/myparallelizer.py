
# pyperparameter grid


import os

# creates folders
folders = ['models', 'gridresults', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass

dataset_names = ['exchange_rate', 'electricity_nips', 'traffic_nips', 'solar_nips' , 'taxi_30min']
input_sizes = [28, 1484, 3856, 552, 2434]


# fix solar dataset
dataset_names = ['solar_nips']
input_sizes = [552]

# model_types = ['GRU-Real-NVP', 'GRU-MAF', 'Transformer-MAF']
model_types = ['GRU-MAF', 'Transformer-MAF']
own_trainer = 1 # 1 if own trainer, 0 if pythorch_ts trainer

for (ds, isz) in zip(dataset_names, input_sizes):
    for mt in model_types:

        print('running job with command: ', F"sbatch run.sh {ds} {isz} {mt} {own_trainer}")
        os.system(F"sbatch run.sh {ds} {isz} {mt} {own_trainer}")










