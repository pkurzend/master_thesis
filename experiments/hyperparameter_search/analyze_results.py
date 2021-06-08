import pickle 
import matplotlib.pyplot as plt 


import os 

files = os.listdir('gridresults/')
print(files)

for filename in files:
    with open('gridresults/' + filename, 'rb') as fp:
        gridresults  = pickle.load(fp)
        print(gridresults)