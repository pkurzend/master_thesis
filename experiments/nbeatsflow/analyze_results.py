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
    print(type(results), type(ls))
    sorted_results = sorted(ls, key=lambda item: item['metrics']['mse'])
    print('DATASET: ', ds_name)
    for item in sorted_results[:5]:
        print(f"mse: {item['metrics']['mse']} \tcrps: {item['metrics']['crps']} \tcrps_sum: {item['metrics']['crps_sum']}")
    print(f"BASELINE: mse: {180000} \tcrps: {0.052} \tcrps_sum: {0.0207}")
    print()
