

import pickle 

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
	sorted_results = sorted(ls, key=lambda item: item['metrics_val']['mse'])

	for model_id, item in enumerate(sorted_results[:3]): # for the 3 best run 20 runs to get valid results on test set
		print(model_id)
		hp = item['hyperparameters']

		ds_name = hp['dataset_name']
		lr = hp['learning_rate']
		bs = hp['batch_size']
		wd = hp['weight_decay']
		b = hp['blocks']
		st = hp['stacks'] 
		ls = hp['layer_size'] 


		for run_id in range(N_RUNS):
			print('running job with command: ', F"sbatch run.sh {model_id} {run_id} {ds_name} {lr} {bs} {wd} {b} {st} {ls}")
			os.system(F"sbatch run.sh {model_id} {run_id} {ds_name} {lr} {bs} {wd} {b} {st} {ls}")

			if abord_after_one_job:
				sys.exit(0)
			
# for dataset:
# 	for best 3 results
# 		run 10 times 

