

import pickle 
import matplotlib.pyplot as plt 
import pandas as pd


import os
import sys

# creates folders
folders = ['models', 'gridresults', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass
print('sure to start that many jobs?')
print('if yes, type "continue"')
cond = input()
if cond != 'continue':
	print('abording...')
	sys.exit(0)


print('Run all jobs? \n if you type "continue", all jobs are started, else, only one is started!')
cond = input()
abord_after_one_job = False
if cond != 'continue':
	abord_after_one_job = True

N_RUNS = 10

started_jobs = 0

files = os.listdir('../gridresults/')
print(len(files), 'files')


results = {}
for ds_name in ['electricity_nips', 'traffic_nips', 'solar_nips', 'exchange_rate']:
    results[ds_name] = []
print(results)

# blocks not used in hp_search
# block_types = [SimpleNBeatsBlock, LinearNBeatsBlock, ConvNBeatsBlock, LinearConvNBeatsBlock]
blocks = list(range(4))

for filename in files:
    with open('../gridresults/' + filename, 'rb') as fp:
        single_run  = pickle.load(fp)
        # print(gridresults)
        results[single_run['dataset_name']].append(single_run)


model_id = 0
for ds_name in ['electricity_nips', 'traffic_nips', 'solar_nips', 'exchange_rate']:
	ls = results[ds_name]
	print('n runs ', len(ls))
	print(type(results), type(ls))
	sorted_results = sorted(ls, key=lambda item: item['metrics_val']['mse'])

	for _, item in enumerate(sorted_results[:1]): # for the 3 best run 20 runs to get valid results on test set
		print(model_id)


		# for attention, transformer and cnn block use same hyperparameters as best model
		for b in blocks[2:]:
			hp = item['hyperparameters']

			ds_name = hp['dataset_name']
			lr = hp['learning_rate']
			
			st = hp['stacks']
			ls = hp['layer_size']
			ah = hp['attention_heads'] 
			aes = hp['attention_embedding_size'] 
			ft = hp['flow_type']

			for run_id in range(N_RUNS):
				print('running job with command: ', F"sbatch run.sh {model_id} {run_id} {ds_name} {lr} {b} {st} {ls} {ah} {aes} {ft}")
				os.system(F"sbatch run.sh {model_id} {run_id} {ds_name} {lr} {b} {st} {ls} {ah} {aes} {ft}")
				model_id += 1
				started_jobs += 1
				if abord_after_one_job:
					sys.exit(0)


		# for SimpleBlock and LinearBlock use different hyperparameters
		for b in blocks[:2]:
			hp = item['hyperparameters']

			ds_name = hp['dataset_name']
			lr = hp['learning_rate']
			
			st = 10
			ls = 128
			ah = hp['attention_heads'] # not used
			aes = hp['attention_embedding_size'] # not used
			ft = hp['flow_type']

			for run_id in range(N_RUNS):
				print('running job with command: ', F"sbatch run.sh {model_id} {run_id} {ds_name} {lr} {b} {st} {ls} {ah} {aes} {ft}")
				os.system(F"sbatch run.sh {model_id} {run_id} {ds_name} {lr} {b} {st} {ls} {ah} {aes} {ft}")
				model_id += 1
				started_jobs += 1
				if abord_after_one_job:
					sys.exit(0)


print(f'started {started_jobs} jobs!')		
# for dataset:
# 	for best 3 results
# 		run 10 times 


