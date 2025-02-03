import os
import numpy as np
import glob

parent_folder = 'C:/Users/neaspec/Desktop/Felix/SNOM_DATA/250203_Felix_LineScans/2025-02-03 16481/'

# pattern = 'Rng_C_1_Nts_Interleaved_Testing'
#pattern = 'Rng_C_1_Nts_Standard2'
# pattern = 'D_2048_T_1C2_A_2'
pattern = 'check700_avg10'
pattern = 'check800_4096_0p8'
pattern = 'Standard_Interleaved_1_Nts'

substrate_path = []
sample_path = []

# dimension of final array: (number of files, column, run, depth, number of keys)
# number of keys = 2 (amplitude and phase) x 6 (harmonics 0 to 5) x 2 (channel O and A) = 24

i = 0
for folder in sorted(os.listdir(parent_folder)):
    if folder.split('NF S ')[-1] == pattern:
        print(folder)
        i += 1