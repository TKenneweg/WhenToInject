import os
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
from tqdm import tqdm
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"

def correctAnswerPattern(answers):
    for answ in answers:
        match = re.search(ANSWER_PATTERN_MULTICHOICE, answ)
        if not match:
            print(f"{answ}")
            print("\n\n#####################\n\n")
            return False
    return True


#load all pkl files from folder
dictlists = []
folder_path = '/home/tristan/dev/llama3/datasets'
file_names = os.listdir(folder_path)
for filename in file_names:
    with open(f'{folder_path}/{filename}', 'rb') as f:
        outdicts = pickle.load(f)
        dictlists.append(outdicts)


# print(len(dictlists[0]))

# firstelem = dictlists[0][0]

# correctFormat(firstelem["answers"])

n_good = 0
n_bad = 0
for dictlist in dictlists:
    for elem in tqdm(dictlist):
        if correctAnswerPattern(elem["answers"]):
            n_good += 1
        else:
            n_bad += 1
        # print(correctAnswerPattern(elem["answers"]))
        # for key in elem.keys():
        #     print(f"{key}: {elem[key]}")
        # print("\n\n#####################\n\n")

print(f"Good: {n_good}")
print(f"Bad: {n_bad}")