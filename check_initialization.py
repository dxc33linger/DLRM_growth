import pickle
import os



file_name = "model_after_growth0.pickle"
path = os.path.join('./saved_model', file_name)
param_dict = pickle.load(open(path, "rb"))
print(param_dict.keys())
print(param_dict['top_l.2.weight'])
print(param_dict['top_l.2.weight'].shape)

file_name = "model_after_growth1.pickle"
path = os.path.join('./saved_model', file_name)
param_dict = pickle.load(open(path, "rb"))
print(param_dict.keys())
print(param_dict['top_l.2.weight'])
print(param_dict['top_l.2.weight'].shape)
