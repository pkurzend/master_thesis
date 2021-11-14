import pickle 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
import os 



import sys
sys.path.append('../../')
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

    # filter out nan and infs
    ls = [item for item in ls if np.isfinite(item['metrics_val']['mse'])]
    sorted_results = sorted(ls, key=lambda item: item['metrics_val']['mse'])
    print('DATASET: ', ds_name)
    # print(f"mse: {sorted_results[0]['metrics']['mse']} \tcrps: {sorted_results[0]['metrics']['crps']} \tcrps_sum: {sorted_results[0]['metrics']['crps_sum']}")
    # print(sorted_results[0]['hyperparameters'])
    # print(f"mse: {sorted_results[-1]['metrics']['mse']} \tcrps: {sorted_results[-1]['metrics']['crps']} \tcrps_sum: {sorted_results[-1]['metrics']['crps_sum']}")
    print()
    for item in sorted_results[:5]:
        print(f"mse: {item['metrics_val']['mse']} \tcrps: {item['metrics_val']['crps']} \tcrps_sum: {item['metrics_val']['crps_sum']}")
    # print(f"BASELINE: mse: {180000} \tcrps: {0.052} \tcrps_sum: {0.0207}")
    print(f"BASELINE: mse: {baselines[ds_name]['mse']} \tcrps: {baselines[ds_name]['crps']} \tcrps_sum: {baselines[ds_name]['crps_sum']}")
    print()


    # for i, item in enumerate(sorted_results[419:420]):
    #     losses = item['losses']
    #     print(losses.keys())

    #     train_losses = losses['train_epoch_losses']
    #     val_losses = losses['val_epoch_losses']

    #     print(type(train_losses))
        

    #     train_losses = np.array(train_losses)
    #     val_losses = np.array(val_losses)
    #     print('train_losses ', train_losses.shape)
    #     print('val_losses ', val_losses.shape)
    #     plot_learning_curves(train_losses, val_losses, fname=ds_name)
        


    columns = {
        **{param : [] for param in hyperparams},
        **{m : [] for m in metrics},
        'n_params' : []
    }



    
    for item in sorted_results:
        # print(item['metrics_val']['mse'])
        hp = item['hyperparameters']
        # print('best model hp: ', item['hyperparameters'])
        for key, value in hp.items():
            if key in columns:
                columns[key].append(value)

        for m in metrics:
            columns[m].append(item['metrics_val'][m])

        columns['n_params'].append(item['number_of_parameters'])
        
    df = pd.DataFrame(columns)
    df = df.sort_values(by=['mse'])
    # print(df.iloc[:5,~df.columns.isin(['n_params','crps', 'crps_sum']) ])
    print(df.iloc[:5,~df.columns.isin(['n_params','crps', 'crps_sum']) ].to_latex(index=False, escape=False))






    # print(df.shape) # 432
    # df_tmp = df.sort_values(by=[hp for hp in hyperparams if hp != 'layer_size']).reset_index()
    # # print(df_tmp.head(30))
    # # print(df_tmp.loc[:,['flow_type', 'mse']])

    # current_index = 0
    # current_hp = df_tmp.at[0, 'flow_type'] 
    # hp_combos = []

    # for i in range(df_tmp.shape[0]):
    
    #     next_hp = df_tmp.at[i, 'flow_type'] 
    #     if next_hp != current_hp:
    #         current_index += 1

    #     current_hp = next_hp
    #     hp_combos.append(current_index)

    # df_tmp['hp_combo'] = hp_combos
    # # print(hp_combos)
    # # print(len(hp_combos)) # 
    # df_tmp.index = df_tmp['index']
        

    # # hp_combos = [[i]*3 for i in range(432//3)][:df_tmp.shape[0]]
    # # hp_combos = [j for ls in hp_combos for j in ls]
    # # df_tmp['hp_combo'] = hp_combos

    # a = df_tmp.groupby('hp_combo', as_index=False)

    
    # repeats = [group.shape[0] for name, group in a]
    # # print(repeats)

    # # a = a['mse'].mean(skipna=False)['mse'].repeat(repeats)
    # a = a['mse'].agg({'mse': lambda x: x.mean(skipna=False), 'std_mse' : lambda x: x.std(skipna=False), 'min_mse' : np.min})
    # # a = a['mse'].agg({'mse': lambda x: x.max(skipna=False)})['mse'].repeat(repeats)
    # # print(a)

    # mean_mse = a['mse'].repeat(repeats)
    # mean_mse.index = df_tmp.index

    # std_mse = a['std_mse'].repeat(repeats)
    # std_mse.index = df_tmp.index
    # # print(a)

    # min_mse = a['min_mse'].repeat(repeats)
    # min_mse.index = df_tmp.index
    
    # df_tmp['mean_mse'] = mean_mse
    # df_tmp['std_mse'] = std_mse
    # df_tmp['min_mse'] = min_mse

    # # df_tmp = df_tmp.drop_duplicates(subset=[hp for hp in hyperparams if hp != 'layer_size'])
    # df_tmp = df_tmp.sort_values(by=['mean_mse'])
    # # df_tmp = df_tmp.head(30).sort_values(by=['min_mse'])

    
    # # df_tmp = df.loc[df['layer_size']==64, :]
    # # print(df_tmp.loc[:,['flow_type', 'mse', 'mean_mse']].head(150))
    # # df_tmp = df_tmp.sort_values(by=['mse'])


    # df = df.sort_values(by=['mse'])
    # print(df.shape)
    # # print(df_tmp.head(15))

    # # print(df.head(5))

    # # sys.exit(0)
    # print(df.iloc[:5,~df.columns.isin(['n_params','crps', 'crps_sum']) ].to_latex(index=False, escape=False))



