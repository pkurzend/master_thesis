
# pyperparameter grid

dataset_names = ['exchange_rate', 'electricity_nips', 'traffic_nips', 'solar_nips' , 'taxi_30min']

# learning related parameters
lrs = [1e-5, 1e-6]
max_lrs = [1e-3, 1e-4]
batch_sizes = [10, 32]
loss_functions = ['MAPE', 'MASE', 'sMAPE']

# which model
interpretable = [0]
stack_features_along_time = [0, 1]
blocks = [0, 1, 2, 3] # indices of [NBeatsBlock, MultivariateNBeatsBlock, TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock]

# model parameters
stacks = [10, 18, 24]
linear_layers = [2, 4]
layer_sizes = [512]

attention_layers = [1]
attention_heads = [1, 4]
attention_embedding_sizes = [512]



# model parameters if interpretable
# trend_layer_sizes = [256, 512]
# seasonality_layer_sizes = [512, 2048]
# degree_of_polynomials = [2, 3, 4]


##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = True
if testing:
	dataset_names = ['solar_nips'] # dimensions. [28, 1484, 3856, 552, 2434]

	lrs = [1e-5]
	max_lrs = [1e-3]
	batch_sizes = [32]
	loss_functions = ['MAPE']
	

	interpretable = [0]
	stack_features_along_time = [0, 1]
	blocks = [1, 2]
	
	
	stacks = [24]
	linear_layers = [2]
	layer_sizes = [512]
	
	attention_layers = [1]
	attention_heads = [4]
	attention_embedding_sizes = [512]
	
	# trend_layer_sizes = [512]
	# seasonality_layer_sizes = [2048]
	# degree_of_polynomials = [3]






testing = True
if testing:
	dataset_names = ['solar_nips'] # dimensions. [28, 1484, 3856, 552, 2434]

	lrs = [1e-5]
	max_lrs = [1e-3]
	batch_sizes = [32]
	loss_functions = ['MAPE']
	

	interpretable = [0]
	stack_features_along_time = [0]
	blocks = [1]
	
	
	stacks = [15]
	linear_layers = [2]
	layer_sizes = [512]
	
	attention_layers = [1]
	attention_heads = [4]
	attention_embedding_sizes = [512]
	
	# trend_layer_sizes = [512]
	# seasonality_layer_sizes = [2048]
	# degree_of_polynomials = [3]





import os

# creates folders
folders = ['models', 'gridresults', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass


# learning related variables
for ds_name in dataset_names:
	for lr  in lrs:
		for max_lr in max_lrs:
			for bs in batch_sizes:
				for lf in loss_functions:
					
					# which model
					for i in interpretable:
						for sfat in stack_features_along_time:
							for b in blocks:
								if b in [0, 1]: # no attention parameters needed

									# model parameters
									
										
									for n_stacks in stacks:
										if b == 0:
											ll = 4										
											for ls in layer_sizes:
												al = 1
												ah = 1
												aes = 512
												print('running job with command: ', F"sbatch run.sh {ds_name} {lr} {max_lr} {bs} {lf} {i} {sfat} {b} {n_stacks} {ll} {ls} {al} {ah} {aes}")
												os.system(F"sbatch run.sh {ds_name} {lr} {max_lr} {bs} {lf} {i} {sfat} {b} {n_stacks} {ll} {ls} {al} {ah} {aes}")
									
										else:
											for ll in linear_layers:
												for ls in layer_sizes:
													al = 1
													ah = 1
													aes = 512
													print('running job with command: ', F"sbatch run.sh {ds_name} {lr} {max_lr} {bs} {lf} {i} {sfat} {b} {n_stacks} {ll} {ls} {al} {ah} {aes}")
													os.system(F"sbatch run.sh {ds_name} {lr} {max_lr} {bs} {lf} {i} {sfat} {b} {n_stacks} {ll} {ls} {al} {ah} {aes}")
									
								# attention parameters needed
								else:
									# model parameters
									for n_stacks in stacks:
										for ll in linear_layers:
											for ls in layer_sizes:

												for al in attention_layers:
													for ah in attention_heads:
														for aes in attention_embedding_sizes:
															print('running job with command: ', F"sbatch run.sh {ds_name} {lr} {max_lr} {bs} {lf} {i} {sfat} {b} {n_stacks} {ll} {ls} {al} {ah} {aes}")
															os.system(F"sbatch run.sh {ds_name} {lr} {max_lr} {bs} {lf} {i} {sfat} {b} {n_stacks} {ll} {ls} {al} {ah} {aes}")



				





