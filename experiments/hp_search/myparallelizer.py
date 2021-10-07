



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

# all datasets['exchange_rate', 'electricity_nips', 'traffic_nips', 'solar_nips' , 'taxi_30min']
# pyperparameter grid
dataset_names = ['electricity_nips', 'traffic_nips', 'solar_nips', 'exchange_rate']
lrs = [1e-4, 1e-5]

# blocks to try
# [SimpleNBeatsBlock, LinearNBeatsBlock, LinearAttentionNBeatsBlock, LinearTransformerEncoderNBeatsBlock, LinearConvNBeatsBlock]
blocks = [2, 3]

stacks = [3, 5, 10]
layer_sizes = [32, 64, 128]


attention_heads = [4, 8]
attention_embedding_sizes = [32, 64, 128]


flow_types = ['RealNVP', 'MAF']





# SimpleNBeatsBlock, LinearNBeatsBlock
# AttentionNBeatsBlock, LinearAttentionNBeatsBlock, TimeAttentionNBeatsBlock
# TransformerEncoderNBeatsBlock, LinearTransformerEncoderNBeatsBlock
# ConvNBeatsBlock, LinearConvNBeatsBlock


##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = True
if testing:
	dataset_names = ['traffic_nips']
	lrs = [1e-5]
	
	# blocks to try
	# [SimpleNBeatsBlock, LinearNBeatsBlock, LinearAttentionNBeatsBlock, LinearTransformerEncoderNBeatsBlock, LinearConvNBeatsBlock]
	blocks = [3]

	stacks = [10]
	layer_sizes = [128]


	attention_heads = [8]
	attention_embedding_sizes = [128]


	flow_types = ['RealNVP']

run_id = 0
for ds_name in dataset_names:
	for lr in lrs:
		for b in blocks:
			for st in stacks:
				for ls in layer_sizes:
					for ah in attention_heads:
						for aes in attention_embedding_sizes:
							for ft in flow_types:
								

										print('running job with command: ', F"sbatch run.sh {run_id} {ds_name} {lr} {b} {st} {ls} {ah} {aes} {ft}")
										os.system(F"sbatch run.sh {run_id} {ds_name} {lr} {b} {st} {ls} {ah} {aes} {ft}")
										run_id += 1
																	



