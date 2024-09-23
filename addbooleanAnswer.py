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


def getBooleanKnown(generator, elem):
    boolprompt = [[{"role": "user", "content" : f"Only answer with 'YES' or 'NO'. Do you think you can answer the following question correctly:\n\n{elem['question']}"}]]
    results = generator.chat_completion(
        boolprompt,
        max_gen_len=2048,
        temperature=0.6,
        top_p=0.9,
        logprobs=True
    )
    answ = results[0]["generation"]["content"]
    match = False
    if "YES" in answ:
        match = True
    elif "NO" in answ:
        match = False
    else:
        print("Could not determine boolean answer")
        return None
    return match


if __name__ == "__main__":
    ckpt_dir="Meta-Llama-3-8B-Instruct"
    tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model"
    max_seq_len=1024
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )
    datasetfolder= "datasets"
    dictlists, file_paths = getdictlists(datasetfolder)
    for dictlist in dictlists:
        for elem in tqdm(dictlist):
            elem['knownprediction'] = getBooleanKnown(generator, elem)



    for dictlist,file_path in zip(dictlists,file_paths):
        print(file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(dictlist, f)  