
# learning rates
lrs = [1e-4, 1e-5, 1e-6]



stack_features_along_time = [0, 1]

loss_functions = ['MAPE', 'MASE', 'sMAPE']

blocks = [0, 1, 2, 3]

stacks = [10, 15, 20, 25, 30]

linear_layers = [2, 4]

layer_sizes = [512]

interpretable = [0, 1]

attention_layers = [1, 2]

attention_heads = [1, 2, 4]

attention_embedding_sizes = [256, 512]

batch_size = [10, 32, 64]

trend_layer_sizes = [256, 512]

seasonality_layer_sizes = [512, 2048]
degree_of_polynomials = [2, 3, 4]

dataset_names = ['exchange_rate', 'electricity_nips', 'traffic_nips', 'solar_nips' , 'taxi_30min']


##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = True
if testing:
	# embedding sizes
	ess = [300]
	# percentage of training data used
	tps = [0.125]


import os

# creates folders
folders = ['models', 'gridresults', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass

for es in ess:
	es = int(es)
	for tp in tps:
		tp = float(tp)
		for lr in lrs:
			lr = float(lr)
			for df in data_frac:
				df = float(df)
				os.system(F"sbatch template.sh {es} {tp} {lr} {df}")
