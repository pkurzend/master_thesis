import pickle 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
pd.set_option('display.max_columns', 20)
import os 

import sys
sys.path.append('../../../')
from models.utils import plot_learning_curves

metrics = ['mse', 'crps', 'crps_sum']
hyperparams = [
        'learning_rate',
        'blocks',
        'stacks',
        'attention_heads',
        'attention_embedding_size',
        'flow_type',
]

gridresults_folder = 'gridresults/'

files = os.listdir(gridresults_folder)
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
    with open(gridresults_folder + filename, 'rb') as fp:
        single_run  = pickle.load(fp)
        # print(gridresults)
        results[single_run['dataset_name']].append(single_run)


mse_latex_table = {'Data Set' : []}
crps_latex_table = {'Data Set' : []}
crpssum_latex_table = {'Data Set' : []}

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
        'block_id' : [],
    }



    for item in sorted_results:
        hp = item['hyperparameters']
        
        # print('best model hp: ', item['hyperparameters'])
        for key, value in hp.items():
            if key in hyperparams:
                
                columns[key].append(value)

        for m in metrics:
            columns[m].append(item['metrics_val'][m])

        columns['n_params'].append(item['number_of_parameters'])
        columns['block_id'].append(item['block_id'])
        
    df = pd.DataFrame(columns)
    print(df.shape)
    # print(df.head(5))

    block_ids = df['block_id'].unique()
    block_ids.sort()
    print('block_id ', block_ids)


    mse_latex_table['Data Set'].append(ds_name)
    crps_latex_table['Data Set'].append(ds_name)
    crpssum_latex_table['Data Set'].append(ds_name)


    for block_id in block_ids:
        print('BLOCK ID ', block_id)

        if block_id not in mse_latex_table:
            mse_latex_table[block_id] = []
            crps_latex_table[block_id] = []
            crpssum_latex_table[block_id] = []

        df_temp = df.loc[df['block_id'] == block_id]
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
        print(df_temp)
        print()
        print('mean mse:      ', format(df_temp['mse'].mean(), '.8f'),      'std mse:      ', format(df_temp['mse'].std(), '.8f'),      'best mse:      ', format(df_temp['mse'].min(), '.8f'),      'worst mse:      ', format(df_temp['mse'].max(), '.8f'))
        print('mean crps:     ', format(df_temp['crps'].mean(), '.8f'),     'std crps:     ', format(df_temp['crps'].std(), '.8f'),     'best crps:     ', format(df_temp['crps'].min(), '.8f'),     'worst crps:     ', format(df_temp['crps'].max(), '.8f'))
        print('mean crps_sum: ', format(df_temp['crps_sum'].mean(), '.8f'), 'std crps_sum: ', format(df_temp['crps_sum'].std(), '.8f'), 'best crps_sum: ', format(df_temp['crps_sum'].min(), '.8f'), 'worst crps_sum: ', format(df_temp['crps_sum'].max(), '.8f'))
        print(f"BASELINE mse: {baselines[ds_name]['mse']} \tcrps: {baselines[ds_name]['crps']} \tcrps_sum: {baselines[ds_name]['crps_sum']}")

        print()

        mse = f"{format(df_temp['mse'].mean(), '.8f')}$\pm${format(df_temp['mse'].std(), '.8f')}"
        crps = f"{format(df_temp['crps'].mean(), '.8f')}$\pm${format(df_temp['crps'].std(), '.8f')}"
        crps_sum = f"{format(df_temp['crps_sum'].mean(), '.8f')}$\pm${format(df_temp['crps_sum'].std(), '.8f')}"

        mse_latex_table[block_id].append(mse)
        crps_latex_table[block_id].append(crps)
        crpssum_latex_table[block_id].append(crps_sum)



mse_df = pd.DataFrame(mse_latex_table)
crps_df = pd.DataFrame(crps_latex_table)
crps_sum_df = pd.DataFrame(crpssum_latex_table)

print(mse_df.to_latex(index=False, escape=False))
print()
print(crps_df.to_latex(index=False, escape=False))
print()
print(crps_sum_df.to_latex(index=False, escape=False))




    

