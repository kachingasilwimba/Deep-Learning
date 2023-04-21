# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 14:24:53 2017

@author: johnchiasson
"""
" Run the `EMNIST_Data_Pickler.py' file to load the data tr_d, va_d, and te_d"
import numpy as np
import EMNIST_Loader2

" Next call the `EMNIST_Loader.load_data_wrapper()' function"
" to put the data tr_d, va_d, and te_d in the proper Nielsen format."
" The Neilsen formatted data is now training_data, validation_data, and testing_data"
" Remember that the labels in 'training_data' are in R^47"
" while the labels in validation_data and testing_data are scalars."

training_data, validation_data, testing_data = EMNIST_Loader2.load_data_wrapper()

"The dictionary for the EMNIST dataset in class."
dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
               10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',
               19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',
               28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',
               36:'a',37:'b',38:'d',39:'e',40:'f',41:'g',42:'h',43:'n',44:'q',
               45:'r',46:'t'
               }

print()
print("dict:\n\n", dict)
print()
print("dict.items():\n\n", dict.items())
print()
print()
print("dict.values():\n\n", dict.values())
print()
print("dict.keys():\n\n", dict.keys())
print()
print('len(dict) is', len(dict))
print()
print("Given the label find its corresponding EMNIST symbol.")
label_number = 42
print("label_number:", label_number)
print("dict[label_number]:", dict[label_number])
print()
print()
print("Given the EMNIST symbol find the label number")
dict2 = {value:key for key, value in dict.items()}
print()
print("dict2:\n\n", dict2)
print()
symbol = 'h'
print("symbol:", symbol)
print()
print("dict2[symbol]:", dict2[symbol])
print()
