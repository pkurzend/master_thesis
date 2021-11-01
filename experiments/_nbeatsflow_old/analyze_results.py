import pickle 
import matplotlib.pyplot as plt 
import pandas as pd
pd.set_option('display.max_columns', 20)

import os 


metrics = ['mse', 'crps', 'crps_sum']
hyperparams = [
        'learning_rate',
        'blocks',
        'stacks',
        'layer_size',
        'attention_heads',
        'attention_embedding_size',
        'flow_type',
]



files = os.listdir('gridresults/')
print(len(files), 'files')

baselines = {'electricity_nips': {'mse' : 180000, 'crps' : 0.051, 'crps_sum': 0.0207}, 
            'traffic_nips': {'mse' : 0.00049, 'crps' : 0.124, 'crps_sum': 0.056}, 
            'solar_nips': {'mse' : 910, 'crps' : 0.365, 'crps_sum': 0.301},
            'exchange_rate': {'mse' : 0.00017, 'crps' : 0.008, 'crps_sum': 0.005}
            }
results = {}
for ds_name in ['electricity_nips', 'traffic_nips', 'solar_nips', 'exchange_rate']:
    results[ds_name] = []
print(results)

for filename in files:
    with open('gridresults/' + filename, 'rb') as fp:
        single_run  = pickle.load(fp)
        # print(gridresults)
        results[single_run['dataset_name']].append(single_run)

for ds_name in ['electricity_nips', 'traffic_nips', 'solar_nips', 'exchange_rate']:
    ls = results[ds_name]
    print('n runs ', len(ls))
    print(type(results), type(ls))
    sorted_results = sorted(ls, key=lambda item: item['metrics']['mse'])
    print('DATASET: ', ds_name)
    # print(f"mse: {sorted_results[0]['metrics']['mse']} \tcrps: {sorted_results[0]['metrics']['crps']} \tcrps_sum: {sorted_results[0]['metrics']['crps_sum']}")
    # print(sorted_results[0]['hyperparameters'])
    # print(f"mse: {sorted_results[-1]['metrics']['mse']} \tcrps: {sorted_results[-1]['metrics']['crps']} \tcrps_sum: {sorted_results[-1]['metrics']['crps_sum']}")
    print()
    for item in sorted_results[:5]:
        print(f"mse: {item['metrics']['mse']} \tcrps: {item['metrics']['crps']} \tcrps_sum: {item['metrics']['crps_sum']}")
    # print(f"BASELINE: mse: {180000} \tcrps: {0.052} \tcrps_sum: {0.0207}")
    print(f"BASELINE: mse: {baselines[ds_name]['mse']} \tcrps: {baselines[ds_name]['crps']} \tcrps_sum: {baselines[ds_name]['crps_sum']}")
    print()


    


    columns = {
        **{param : [] for param in hyperparams},
        **{m : [] for m in metrics},
        'n_params' : []
    }



    for item in sorted_results[:5]:
        hp = item['hyperparameters']
        # print('best model hp: ', item['hyperparameters'])
        for key, value in hp.items():
            if key in columns:
                columns[key].append(value)

        for m in metrics:
            columns[m].append(item['metrics'][m])

        columns['n_params'] = item['number_of_parameters']
        
    df = pd.DataFrame(columns)
    print(df)
    

