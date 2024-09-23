import torch
import os
import pickle
from tqdm import tqdm   
from llama import Dialog, Llama
import re


def getdictlists(path):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_paths.append(os.path.join(root, file))


    # file_paths= file_paths[0:2] #for debuggin
    dictlists = []
    print("Loading dicts")
    for file_path in tqdm(file_paths):
        with open(file_path, 'rb') as f:
            dictlist = pickle.load(f)
            
            dictlists.append(dictlist)
    return dictlists, file_paths


if __name__ == "__main__":
    print("hi")
    datasetfolder= "datasets"
    dictlists, file_paths = getdictlists(datasetfolder)
    for dictlist in dictlists:
        for elem in tqdm(dictlist):
            elem['embedding'] = elem['embedding'].to('cpu')


    for dictlist,file_path in zip(dictlists,file_paths):
        print(file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(dictlist, f)  