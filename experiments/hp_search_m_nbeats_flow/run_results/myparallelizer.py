

import pickle 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


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

for filename in files:
    with open('../gridresults/' + filename, 'rb') as fp:
        single_run  = pickle.load(fp)
        # print(gridresults)
        results[single_run['dataset_name']].append(single_run)



for ds_name in ['electricity_nips', 'traffic_nips', 'solar_nips', 'exchange_rate']:
	ls = results[ds_name]
	print('n runs ', len(ls))
	print(type(results), type(ls))


	# each hp combo has been run 3 times (bug), hence one must be randomly sampled
	# this can be done by chosing the sample with layer_size = 32 for eaxample
	# ls = list(filter(lambda item: item['hyperparameters']['layer_size']==32 , ls))


	# sort results by mse
	ls = [item for item in ls if np.isfinite(item['metrics_val']['mse'])]
	sorted_results = sorted(ls, key=lambda item: item['metrics_val']['mse'])


	results_linear_attention_block = list(filter(lambda item: item['hyperparameters']['blocks']==2 , sorted_results))

	reulsts_linear_transformer_block = list(filter(lambda item: item['hyperparameters']['blocks']==3 , sorted_results))




	
	for block_id,  block_results in enumerate([results_linear_attention_block, reulsts_linear_transformer_block]):
		for item in block_results[:1]: # for the 1 best run 10 runs to get valid results on test set
			
			hp = item['hyperparameters']

			ds_name = hp['dataset_name']
			lr = hp['learning_rate']
			b = hp['blocks']
			st = hp['stacks']
			ls = hp['layer_size']
			ah = hp['attention_heads'] 
			aes = hp['attention_embedding_size'] 
			ft = hp['flow_type']

			for run_id in range(N_RUNS):
				print('running job with command: ', F"sbatch run.sh {block_id} {run_id} {ds_name} {lr} {b} {st} {ls} {ah} {aes} {ft}")
				os.system(F"sbatch run.sh {block_id} {run_id} {ds_name} {lr} {b} {st} {ls} {ah} {aes} {ft}")
				started_jobs += 1
				if abord_after_one_job:
					sys.exit(0)


print(f'started {started_jobs} jobs!')
# for dataset:
# 	for best 3 results
# 		run 10 times 


