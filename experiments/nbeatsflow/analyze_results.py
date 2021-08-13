import pickle 
import matplotlib.pyplot as plt 


import os 

files = os.listdir('gridresults/')
print(len(files), 'files')

baselines = {'electricity_nips': {'mse' : 180000, 'crps' : 0.051, 'crps_sum': 0.0207}, 
            'traffic_nips': {}, 
            'solar_nips': {}}
results = {}
for ds_name in ['electricity_nips', 'traffic_nips', 'solar_nips']:
    results[ds_name] = []
print(results)

for filename in files:
    with open('gridresults/' + filename, 'rb') as fp:
        single_run  = pickle.load(fp)
        # print(gridresults)
        results[single_run['dataset_name']].append(single_run)

for ds_name in ['electricity_nips', 'traffic_nips', 'solar_nips']:
    ls = results[ds_name]
    print('n runs ', len(ls))
    print(type(results), type(ls))
    sorted_results = sorted(ls, key=lambda item: item['metrics']['mse'])
    print('DATASET: ', ds_name)
    print(f"mse: {sorted_results[0]['metrics']['mse']} \tcrps: {sorted_results[0]['metrics']['crps']} \tcrps_sum: {sorted_results[0]['metrics']['crps_sum']}")
    print(sorted_results[0]['hyperparameters'])
    print(f"mse: {sorted_results[-1]['metrics']['mse']} \tcrps: {sorted_results[-1]['metrics']['crps']} \tcrps_sum: {sorted_results[-1]['metrics']['crps_sum']}")
    print()
    for item in sorted_results[:5]:
        print(f"mse: {item['metrics']['mse']} \tcrps: {item['metrics']['crps']} \tcrps_sum: {item['metrics']['crps_sum']}")
    print(f"BASELINE: mse: {180000} \tcrps: {0.052} \tcrps_sum: {0.0207}")
    print()

    for item in sorted_results[:5]:
        print(item['hyperparameters'])
    print(f"BASELINE: mse: {180000} \tcrps: {0.052} \tcrps_sum: {0.0207}")
