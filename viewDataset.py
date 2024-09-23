import pickle





dictlists = []
topicslist = ["high_school_psychology_test.pkl"]
for topic in topicslist:
    with open(f'datasets/{topic}', 'rb') as f:
        outdicts = pickle.load(f)
        dictlists.append(outdicts)

firstitem = dictlists[0][0]
print(firstitem["embedding"].device)
# print(firstitem.keys())
# for key in firstitem.keys():
#     print(key)
#     print(type(firstitem[key]))
#     print("\n")
# print(firstitem["knownprediction"])



