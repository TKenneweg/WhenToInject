import torch
import os
import pickle
from tqdm import tqdm   
import math
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import numpy as np

def getentropy(probs, base=2):
    return -torch.sum(probs * torch.log(probs))

def getAverageEntropy(elem, top_k = 15):
    avgentropy = 0
    for i,probs in enumerate(elem["probs"]):
        avgentropy += getentropy(torch.tensor(probs))
        if i >= top_k:
            break
    avgentropy = avgentropy / top_k
    return avgentropy

def getChainProb(elem, top_k = 15):
    chainprob = 0
    for i,top_logprob in enumerate(elem["top_logprobs"]):
        chainprob += top_logprob #using log sum trick
        if i >= top_k:
            break
    chainprob = math.exp(chainprob)
    return chainprob


def getdictlists(path):
    selected = False
    mytopics= ["clinical_knowledge_test.pkl","medical_genetics_test.pkl"]
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if selected:
                if file in mytopics:
                    file_paths.append(os.path.join(root, file))
            else:
                file_paths.append(os.path.join(root, file))


    file_paths= file_paths #for debuggin
    dictlists = []
    print("Loading dicts")
    for file_path in tqdm(file_paths):
        with open(file_path, 'rb') as f:
            dictlist = pickle.load(f)
            dictlists.append(dictlist)
    return dictlists

def printStatistics(value,labels, name):
    truevalues = [value[i] for i in range(len(labels)) if labels[i]]
    falsevalues = [value[i] for i in range(len(labels)) if not labels[i]]

    known_mean = sum(truevalues) / len(truevalues)
    unknown_mean = sum(falsevalues) / len(falsevalues)

    print(f"Mean value of known{name}: {known_mean}")
    print(f"Mean value of unknown{name}: {unknown_mean}")
    print(f"Ratio of known{name} to unknown{name}: {known_mean / unknown_mean}")
    return truevalues, falsevalues


if __name__ == "__main__":
    dictlists = getdictlists("datasets")
    rs = []
    topks = []
    # for i in range(30):
    topk = 5
    chainprobs =[]
    knowns = []
    averageEntropies = []
    n_tp = 0
    n_fp = 0
    n_tn = 0 
    n_fn = 0
    increment = 1/20
    thresholdvalues = torch.arange(0,1,0.05)
    n_tpvec = torch.zeros(len(thresholdvalues))
    n_fpvec =  torch.zeros(len(thresholdvalues))
    n_tnvec = torch.zeros(len(thresholdvalues))
    n_fnvec =  torch.zeros(len(thresholdvalues))
    for dictlist in dictlists:
        for elem in dictlist:
            chainprob = getChainProb(elem, top_k = topk)
            chainprobs.append(chainprob)

            # for i,val in enumerate(thresholdvalues):
            #     if elem["known"] and chainprob > val:
            #         n_tpvec[i] += 1
            #     elif elem["known"] and not chainprob > val:
            #         n_fnvec[i] += 1
            #     elif not elem["known"] and chainprob > val:
            #         n_fpvec[i] += 1
            #     elif not elem["known"] and not chainprob > val:
            #         n_tnvec[i] += 1
                        
            knowns.append(elem["known"])
            averageEntropies.append(getAverageEntropy(elem, top_k = topk))
            key = "knownpredictionReflection"
            # key = "knownprediction"
            if elem[key] is not None:
                if elem["known"] and elem[key]:
                    n_tp += 1
                elif elem["known"] and not elem[key]:
                    n_fn += 1
                elif not elem["known"] and elem[key]:
                    n_fp += 1
                elif not elem["known"] and not elem[key]:
                    n_tn += 1
                


    # for i in range(len(thresholdvalues)):
        # print(f"Threshold: {thresholdvalues[i]}")
        # print(f"Accuracy: {(n_tpvec[i] + n_tnvec[i]) / (n_tpvec[i] + n_fpvec[i] + n_tnvec[i] + n_fnvec[i])}")
        # print("#####################\n\n")

    # print(f"True positives: {n_tp}")
    # print(f"False positives: {n_fp}")
    # print(f"True negatives: {n_tn}")
    # print(f"False negatives: {n_fn}")
    # print(f"Precision: {n_tp / (n_tp + n_fp)}")
    # print(f"NPV: {n_tn / (n_tn + n_fn)}")
    # print(f"Accuracy: {(n_tp + n_tn) / (n_tp + n_tn + n_fn + n_fp)}")

    # print(f"True negative ratio: {n_tn / (n_tn + n_fn)}")
    # print(f"Accuracy: {(n_tp + n_tn) / (n_tp + n_fp + n_tn + n_fn)}")
    # count_known = sum(1 for elem in dictlists for elem in elem if elem["known"])
    # count_unknown = sum(1 for elem in dictlists for elem in elem if not elem["known"])
    # print(f"Number of elements with known = True: {count_known}")
    # print(f"Number of elements with known = False: {count_unknown}")
    # print(f"")
    # print(f"Baseline pos")
    valtrue, valfalse = printStatistics(chainprobs,knowns, "chainprobs")
    # valtrue, valfalse = printStatistics(averageEntropies,knowns, "averageEntropies")
    # r, p_value = pointbiserialr(knowns, chainprobs)
    # print(f"Point biserial correlation coefficient: {r} for top {topk}")
    # rs.append(r)
    # topks.append(topk)

    # plt.plot(topks, rs)
    # plt.xlabel('Top k')
    # plt.ylabel('Point-Biserial Correlation Coefficient')
    # plt.title('Point-Biserial Correlation Coefficient vs k')
    # plt.show()
        # print(f"P-value: {p_value}")

    # # print(len(valtrue), len(valfalse))
    # Create histogram of chainprobs with 10 buckets
    # plt.hist([valtrue[0:len(valfalse)], valfalse], bins=10, label=['Known', 'Unknown'])
    # plt.xlabel('Chain Probability')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Chain Probability (k=5)')
    # plt.legend()
    # plt.show()

    # hist, xedges, yedges = np.histogram2d(valtrue[0:len(valfalse)],valfalse, bins=10)
    # print(f"Counts: {counts}")
    # print(f"Bins: {bins}")
    # counts, bins = np.histogram(valfalse)
    # print(f"Counts: {counts}")
    # print(f"Bins: {bins}")

    # plt.hist([valtrue[0:len(valfalse)], valfalse], bins=10, label=['Known', 'Unknown'])
    # plt.xlabel('Average Entropy')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Average Entropy (k=5)')
    # plt.legend()
    # plt.show()
    # plt.scatter(chainprobs, knowns)
    # plt.xlabel('Chain Probability')
    # plt.ylabel('Knowns')
    # plt.title('Scatterplot of Chain Probability vs Knowns')
    # plt.show()

    # plt.scatter(averageEntropies, knowns)
    # plt.xlabel('Chain Probability')
    # plt.ylabel('Knowns')
    # plt.title('Scatterplot of Chain Probability vs Knowns')
    # plt.show()