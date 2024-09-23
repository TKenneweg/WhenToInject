import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score


def printMetrics(dataloader,model, datastring):
    # Calculate accuracy on the untrained model
    correct = 0
    total = 0
    n_tp = 0
    n_tn = 0
    n_fp = 0
    n_fn = 0
    model.eval()  # Turn off dropout
    with torch.no_grad():
        for data in dataloader:
            inputs, labels, emblengths,chainprobs = data
            outputs = model(inputs, emblengths,chainprobs.to("cuda").float())
            predicted = torch.round(outputs)
            n_tp += ((predicted == 1) & (labels.unsqueeze(1) == 1)).sum().item()
            n_tn += ((predicted == 0) & (labels.unsqueeze(1) == 0)).sum().item()
            n_fp += ((predicted == 1) & (labels.unsqueeze(1) == 0)).sum().item()
            n_fn += ((predicted == 0) & (labels.unsqueeze(1) == 1)).sum().item()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    untrained_test_accuracy = correct / total
    # print(f"Metrics for {datastring} data:")
    print(f'Model {datastring} Accuracy: {untrained_test_accuracy:.2%}')
    print(f"True positives: {n_tp}")
    print(f"False positives: {n_fp}")
    print(f"True negatives: {n_tn}")
    print(f"False negatives: {n_fn}")
    print(f"Precision: {n_tp / (n_tp + n_fp)}")
    print(f"NPV: {n_tn / (n_tn + n_fn)}")
    print(f"Accuracy: {(n_tp + n_tn) / (n_tp + n_tn + n_fn + n_fp)}")
    print("\n\n#####################\n\n")
    model.train()  # Turn on dropout
    return (n_tp + n_tn) / (n_tp + n_tn + n_fn + n_fp)


def genROCCurve(dataloader,model, datastring):
    model.eval()  # Turn off dropout
    with torch.no_grad():
        for data in dataloader:
            inputs, labels, emblengths,chainprobs = data
            with torch.no_grad():
                outputs = model(inputs, emblengths,chainprobs.to("cuda").float())
            predicted = torch.round(outputs)
            n_tp += ((predicted == 1) & (labels.unsqueeze(1) == 1)).sum().item()
            n_tn += ((predicted == 0) & (labels.unsqueeze(1) == 0)).sum().item()
            n_fp += ((predicted == 1) & (labels.unsqueeze(1) == 0)).sum().item()
            n_fn += ((predicted == 0) & (labels.unsqueeze(1) == 1)).sum().item()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    # print(f"Metrics for {datastring} data:")

    model.train()  # Turn on dropout
    return (n_tp + n_tn) / (n_tp + n_tn + n_fn + n_fp)


def getChainProb(elem, top_k = 5):
    chainprob = 0
    for i,top_logprob in enumerate(elem["top_logprobs"]):
        chainprob += top_logprob #using log sum trick
        if i >= top_k:
            break
    chainprob = math.exp(chainprob)
    return chainprob

def getentropy(probs, base=2):
    return -torch.sum(probs * torch.log(probs))

def getAverageEntropy(elem, top_k = 5):
    avgentropy = 0
    for i,probs in enumerate(elem["probs"]):
        avgentropy += getentropy(torch.tensor(probs))
        if i >= top_k:
            break
    avgentropy = avgentropy / top_k
    return avgentropy

#embeddings dims are batch, numtokens, 4096
embdim = 4096
class EmbeddingDataset(Dataset):
    def __init__(self, file_paths):
        self.embeddings = []
        self.labels = []
        self.max_embeddinglength =0
        self.chainprobs = []
        self.entropies = []

        dictlists = []        
        # Load and transform the data
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                dictlist = pickle.load(f)
                for item in dictlist:
                    self.max_embeddinglength = max(self.max_embeddinglength, item['embedding'].shape[1])
                dictlists.append(dictlist)

        print("Max embdding size is of loaded data is: " ,self.max_embeddinglength)
        for dictlist in dictlists:
            for item in dictlist:
                self.embeddings.append(item['embedding'].to("cuda"))  
                self.labels.append(item['known'])
                self.chainprobs.append(getChainProb(item))
                self.entropies.append(getAverageEntropy(item))

        self.labels = torch.tensor(self.labels, dtype=torch.long).to("cuda")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = torch.zeros([self.max_embeddinglength, self.embeddings[idx].shape[2]], device="cuda")
        emb[:self.embeddings[idx].shape[1], :] = self.embeddings[idx].squeeze().float()
        return emb, self.labels[idx], torch.tensor(self.embeddings[idx].shape[1], device="cuda", dtype=torch.float), self.chainprobs[idx], -self.entropies[idx] 


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(4096, 32)  # First layer
        self.fcmiddle = nn.Linear(32, 32)  # First layer
        self.fc2 = nn.Linear(32, 1)   # Second layer
        self.dropout = nn.Dropout(0.2)  # Dropout layer for regularization

    def forward(self, x, emblength, chainprob):
        x = F.leaky_relu(self.fc1(x))
        x = torch.sum(x, dim=1)
        x = x / emblength.unsqueeze(1)
        x = self.fcmiddle(x )
        # x = self.fcmiddle(x + chainprob.unsqueeze(1))
        # x = self.dropout(x)  # Apply dropout regularization
        return torch.sigmoid(self.fc2(x))  # Apply the second layer


if __name__ == "__main__":
    alltopics = ['high_school_european_history', 'business_ethics','high_school_us_history', "clinical_knowledge","medical_genetics", 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']
    # alltopics = ["clinical_knowledge","medical_genetics"]
    file_paths = [f'datasets/{topic}_test.pkl' for topic in alltopics]

    state_dict = torch.load('./best_model.pth')
    model = BinaryClassifier()
    model.load_state_dict(state_dict)
    model.to("cuda")

    dataset = EmbeddingDataset(file_paths)
    bs = 16
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    torch.manual_seed(1)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1))
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)


    labels = torch.tensor([]).to("cuda")
    predprobs = torch.tensor([]).to("cuda")
    chainprobsall = torch.tensor([]).to("cuda")
    entropiesall = torch.tensor([]).to("cuda")

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labelbatch, emblengths,chainprobs, entropies = data
            outputs = model(inputs, emblengths,chainprobs.to("cuda").float())
            # print(outputs.shape)
            predprobs = torch.cat((predprobs, outputs), dim=0)
            labels = torch.cat((labels, labelbatch), dim=0)
            chainprobsall = torch.cat((chainprobsall, chainprobs.to("cuda")), dim=0)
            entropiesall = torch.cat((entropiesall, entropies.to("cuda")), dim=0)
    predprobs = predprobs.squeeze().cpu().numpy()
    labels = labels.cpu().numpy()
    chainprobsall = chainprobsall.cpu().numpy()
    entropiesall = entropiesall.cpu().numpy()

    # Calculate FPR, TPR, and thresholds
    # fpr, tpr, thresholds = roc_curve(labels, predprobs)
    # fpr, tpr, thresholds = roc_curve(labels, chainprobsall)
    fpr, tpr, thresholds = roc_curve(labels, entropiesall)
    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr,color='green', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # genROCCurve(test_dataloader,model, "Test")
    testacc = printMetrics(test_dataloader,model, "Test")
