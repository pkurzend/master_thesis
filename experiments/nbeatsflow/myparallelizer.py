



import os

# creates folders
folders = ['models', 'gridresults', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass

# pyperparameter grid
dataset_names = ['electricity_nips', 'traffic_nips', 'solar_nips']
lrs = [1e-4, 1e-5]

stack_features_along_time = [0, 1]
stacks = [5, 10]
layer_sizes = [32, 64, 128, 512]


attention_heads = [4, 8]
attention_embedding_sizes = [32, 64, 128, 512]
positional_encoding = [0, 1]

flow_types = ['RealNVP', 'MAF']


##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = False
if testing:
	dataset_names = ['traffic_nips']
	lrs = [1e-5]
	loss_functions = ['MAPE']
	
	stack_features_along_time = [0]
	stacks = [10]
	layer_sizes = [512]


	attention_heads = [8]
	attention_embedding_sizes = [512]
	positional_encoding = [1]

	flow_types = ['RealNVP']

trials = 5
run_id = 0
for ds_name in dataset_names:
	for lr in lrs:
		for sfat in stack_features_along_time:
			for st in stacks:
				for ls in layer_sizes:
					for ah in attention_heads:
						for aes in attention_embedding_sizes:
							for pe in positional_encoding:
								for ft in flow_types:
									for idx in range(trials):

										print('running job with command: ', F"sbatch run.sh {run_id} {idx} {ds_name} {lr} {sfat} {st} {ls} {ah} {aes} {pe} {ft}")
										os.system(F"sbatch run.sh {run_id} {idx} {ds_name} {lr} {sfat} {st} {ls} {ah} {aes} {pe} {ft}")
										run_id+= 1
																	



