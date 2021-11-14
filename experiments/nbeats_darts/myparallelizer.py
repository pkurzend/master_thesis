



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


dataset_names = ['electricity_nips', 'traffic_nips', 'solar_nips', 'exchange_rate']



N_RUNS=10

for ds_id, ds_name in enumerate(dataset_names):
	for run_id in range(N_RUNS):
		print('running job with command: ', F"sbatch run.sh {ds_id} {run_id} {ds_name}")
		os.system(F"sbatch run.sh {ds_id} {run_id} {ds_name}")
		
																				


