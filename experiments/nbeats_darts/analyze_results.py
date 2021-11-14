import pickle 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
pd.set_option('display.max_columns', 20)
import os 
import sys

    
sys.path.append('../../')

from models.nbeats import NBeatsBlock
from models.utils import plot_learning_curves


metrics = ['mse']
hyperparams = [
        'learning_rate',
        'stacks',
        'layer_size',
       
        
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


mse_latex_table = {'Data Set' : [], 'M-N-BEATS-Darts' : []}


for ds_name in ['electricity_nips', 'traffic_nips', 'solar_nips', 'exchange_rate']:
    ls = results[ds_name]
    print('n runs ', len(ls))
    print(type(results), type(ls))

    ls = [item for item in ls if np.isfinite(item['metrics_val']['mse'])]
    sorted_results = sorted(ls, key=lambda item: item['metrics_val']['mse'])
    print('DATASET: ', ds_name)

    print(f"BASELINE: mse: {baselines[ds_name]['mse']} \tcrps: {baselines[ds_name]['crps']} \tcrps_sum: {baselines[ds_name]['crps_sum']}")
    print()


    


    columns = {
        **{param : [] for param in hyperparams},
        **{m : [] for m in metrics},
        'n_params' : [],
        'dataset_id' : [],
    }

    print('sorted_results',len(sorted_results))
    
    for item in sorted_results:
        hp = item['hyperparameters']
        
        # print('best model hp: ', item['hyperparameters'])
        for key, value in hp.items():
            if key in hyperparams:
                
                columns[key].append(value)

        for m in metrics:
            columns[m].append(item['metrics_val'][m])

        columns['n_params'].append(item['number_of_parameters'])
        columns['dataset_id'].append(item['dataset_id'])
    



    df = pd.DataFrame(columns)
    print(df.shape)
    print(df.head(5))

    print('dataset_id ', df['dataset_id'].unique()) 
    # sys.exit()

    mse_latex_table['Data Set'].append(ds_name) #

    # for dataset_id in df['dataset_id'].unique():
    # df_temp = df.loc[df['dataset_name'] == ds_name]
    df_temp = df
    print('has nans or infs ', df_temp.isnull().values.any(), df_temp.isin([np.inf, -np.inf]).values.any())
    print()

    # data = {
    #     'mean_mse'
    #     'mean_crps'
    #     'mean_crps_sum'
    #     'std_mse'
    #     'std_crps'
    #     'std_crps_sum'
    # }

    # agg_df = pd.DataFrame(data)
    
    print('mean mse:      ', format(df_temp['mse'].mean(), '.8f'),      'std mse:      ', format(df_temp['mse'].std(), '.8f'),      'best mse:      ', format(df_temp['mse'].min(), '.8f'))

    print(f"BASELINE mse: {baselines[ds_name]['mse']} \tcrps: {baselines[ds_name]['crps']} \tcrps_sum: {baselines[ds_name]['crps_sum']}")

    print()

    mse = f"{format(df_temp['mse'].mean(), '.8f')}$\pm${format(df_temp['mse'].std(), '.8f')}"
    mse_latex_table['M-N-BEATS-Darts'].append(mse)
mse_df = pd.DataFrame(mse_latex_table)


print(mse_df.to_latex(index=False, escape=False))






#     for dataset_id in df['dataset_id'].unique():
#         df_temp = df.loc[df['dataset_id'] == dataset_id]
#         print('has nans or infs ', df_temp.isnull().values.any(), df_temp.isin([np.inf, -np.inf]).values.any())
#         print()

#         # data = {
#         #     'mean_mse'
#         #     'mean_crps'
#         #     'mean_crps_sum'
#         #     'std_mse'
#         #     'std_crps'
#         #     'std_crps_sum'
#         # }

#         # agg_df = pd.DataFrame(data)
        
#         print('mean mse:      ', format(df_temp['mse'].mean(), '.8f'),      'std mse:      ', format(df_temp['mse'].std(), '.8f'),      'best mse:      ', format(df_temp['mse'].min(), '.8f'))
 
#         print(f"BASELINE mse: {baselines[ds_name]['mse']} \tcrps: {baselines[ds_name]['crps']} \tcrps_sum: {baselines[ds_name]['crps_sum']}")

#         print()

#         mse = f"{format(df_temp['mse'].mean(), '.8f')}$\pm${format(df_temp['mse'].std(), '.8f')}"
#         mse_latex_table['M-N-BEATS-Darts'].append(mse)
# mse_df = pd.DataFrame(mse_latex_table)


# print(mse_df.to_latex(index=False, escape=False))





    

