import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import nltk
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from statistics import mean
from collections import Counter
from nltk import ngrams 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
from copy import deepcopy
import operator
from math import log2

# Download required modules
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Returns n-grams
def get_ngrams(data, n):
    tokens = [token for token in data.split(" ") if token != ""]
    res = list(ngrams(tokens, n))    
    return res

# Get the set of English stop words
stop_words = set(stopwords.words('english')) 


def get_pos_tag(text):
    tokenized = sent_tokenize(text)
    wordsList = nltk.word_tokenize(tokenized[0]) 
    wordsList = [w for w in wordsList if not w in stop_words]  
    tagged = nltk.pos_tag(wordsList) 
    return tagged


def build_data(file_path):
    data = []
    uni = []
    bi = []
    tri = []
    pos = []
    file = open(file_path, encoding = "ISO-8859-1")

    for line in file:
        line = line.split(':')
        row = []
        row.append(line[0])
        row.append(' '.join(line[1].split(' ')[1:]).translate(str.maketrans('', '', string.punctuation)).rstrip())

        length = len(row[1].split(' '))
        unigram = get_ngrams(row[1], 1)
        bigram = get_ngrams(row[1], 2)
        trigram = get_ngrams(row[1], 3)
        postag = get_pos_tag(row[1])

        row.append(length)
        row.append(unigram)
        uni.extend(unigram)
        row.append(bigram)
        bi.extend(bigram)
        row.append(trigram)
        tri.extend(trigram)
        row.append(postag)
        pos.extend(postag)
        data.append(row)

    return data, uni, bi, tri, pos


data, uni, bi, tri, pos = build_data('./traindata.txt')


def frequent_grams(g, top_n):
    return Counter(g).most_common(top_n)

unigram_counts = frequent_grams(uni, 500)
bigram_counts = frequent_grams(bi, 300)
trigram_counts = frequent_grams(tri, 200)
pos_counts = frequent_grams(pos, 500)
avgLength = mean([row[2] for row in data])
print(avgLength)

def is_numeric(val):
    return isinstance(val, int) or isinstance(val, float)


header = ['Label', 'Text', 'Length', 'Unigram', 'Bigram', 'Trigram']

class Feature:
    def __init__(self, col, val):
        self.column = col
        self.value = val

    def match(self, ex):
        val = ex[self.column]
        
        if is_numeric(val):
            return val <= self.value
        return self.value in val

    def __repr__(self):
        condition = "exists"
        return "Does %s %s %s?" % (
            header[self.column], str(self.value), condition)

def class_fref(rows):
    counts = {}
    for row in rows:
        label = row[0]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def gini(rows):
    counts = class_fref(rows)
    imp = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        imp -= prob_of_label**2
    return imp


def misclassifcation_error(rows):
    counts = class_fref(rows)
    max_prob = 0
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        if prob_of_label > max_prob:
            max_prob = prob_of_label
    return 1 - max_prob


def entropy(rows):
    counts = class_fref(rows)
    imp = 0
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        imp -= prob_of_label*log2(prob_of_label)
    return imp

def info_gain(left, right, curr_uncertainty, func):
    p = float(len(left)) / (len(left) + len(right))
    return curr_uncertainty - p * func(left) - (1 - p) * func(right)

class Leaf:
    def __init__(self, r):
        self.predictions = class_fref(r)

class Decision_Node:
    def __init__(self,
                 feat,
                 true_b,
                 false_b):
        self.Feature = feat
        self.true_branch = true_b
        self.false_branch = false_b

Features = []

for y in unigram_counts:
    Features.append(Feature(3, y[0]))

for y in bigram_counts:
    Features.append(Feature(4, y[0]))
    
for y in trigram_counts:
    Features.append(Feature(5, y[0]))

for y in pos_counts:
    Features.append(Feature(6, y[0]))
    
Features.append(Feature(2, avgLength))    
    
print(len(Features))
# print(Features[1500])

def partition(rows, Feature):
    trueRows = []
    falseRows = []
    
    for row in rows:
        if Feature.match(row):
            trueRows.append(row)
        else:
            falseRows.append(row)
    return trueRows, falseRows

def findBestSplit(rows, Features, func):   
    best_gain = 0
    best_Feature = None
    current_uncertainty = func(rows)
    
    for f in Features:
        trueRows, falseRows = partition(rows, f)
        if len(trueRows) == 0 or len(falseRows) == 0:
            continue
        
        gain = info_gain(trueRows, falseRows, current_uncertainty, func)
        
        if gain >= best_gain:
            best_gain, best_Feature = gain, f
    
    return best_gain, best_Feature   

def formTree(rows, Features, func):
    gain, Feature = findBestSplit(rows, Features, func)
       
    if gain == 0:
        return Leaf(rows)
    
    trueRows, falseRows = partition(rows, Feature)
    Features.remove(Feature)
    
    trueBranch = formTree(trueRows, Features, func)
    falseBranch = formTree(falseRows, Features, func)
    
    return Decision_Node(Feature, trueBranch, falseBranch)

def classifyRow(node, row):
    if isinstance(node, Leaf):
        return node.predictions
    
    if node.Feature.match(row):
        return classifyRow(node.true_branch, row)
    else:
        return classifyRow(node.false_branch, row)

def train(data, Features, func):
    return formTree(data, deepcopy(Features), func)

def classify(root, rows):
    predictions = []
    for r in rows:
        predictions.append(max(classifyRow(root, r).items(), key=operator.itemgetter(1))[0])
    return predictions

def getDataInIndex(data, index):
    l = []
    for i in range(len(data)):
        if i in index:
            l.append(data[i])
    return l


def getActualLabels(act_data):
    act_labels = []
    for d in act_data:
        act_labels.append(d[0])
    return act_labels

kfold = KFold(10, True, 1)
precision = []
recall = []
f_score = []
i = 0

for trainInd,testInd in kfold.split(data):
    train_data = getDataInIndex(data, trainInd)
    test_data = getDataInIndex(data, testInd)
    
    root = train(train_data, Features, gini)
    
    prediction = classify(root, test_data)
        
    actual = getActualLabels(test_data)
    predicted = prediction
    
#     print(classification_report(actual, predicted))
    precision.append(precision_score(actual, predicted, average='macro'))
    recall.append(recall_score(actual, predicted, average='macro'))
    f_score.append(f1_score(actual, predicted, average='macro'))
     
    print("Training ...")

print("Precision Score: " + str(mean(precision)))
print("Recall Score: " + str(mean(recall)))
print("F-Score: " + str(mean(f_score)))


# ## Part 2
# - All
# - Unigram, Bigram, Trigram, POS
# - Unigram, Bigram, Trigram

classes = ['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']

def getReport(traindata, testdata, uniFlag=True, biFlag=True, triFlag=True, posFlag=True, lenFlag=True, func=gini):
    allFeatures = []
    
    if uniFlag:
        for y in unigram_counts:
            allFeatures.append(Feature(3, y[0]))

    if biFlag:
        for y in bigram_counts:
            allFeatures.append(Feature(4, y[0]))

    if triFlag:
        for y in trigram_counts:
            allFeatures.append(Feature(5, y[0]))

    if posFlag:
        for y in pos_counts:
            allFeatures.append(Feature(6, y[0]))

    if lenFlag:
        allFeatures.append(Feature(2, avgLength))    

    print("No of Features: " + str(len(allFeatures)))
    print("Training ...")
    root = train(traindata, allFeatures, func)
    print("Predicting ...")
    prediction = classify(root, testdata)        
    actual = getActualLabels(testdata)
    print("Prediction done ...")
    matrix = confusion_matrix(actual, prediction)
    acc = matrix.diagonal()/matrix.sum(axis=1)
    accuracy_report = dict(zip(classes, acc))
    
    return accuracy_report, root, prediction, actual

testdata = build_data('./testdata.txt')[0]
len(testdata)

print(getReport(traindata=data, testdata=testdata)[0])

print(getReport(traindata=data, testdata=testdata, func=entropy)[0])

print(getReport(traindata=data, testdata=testdata, func=misclassifcation_error)[0])

print(getReport(traindata=data, testdata=testdata, lenFlag=False)[0])

print(getReport(traindata=data, testdata=testdata, lenFlag=False, func=entropy)[0])

print(getReport(traindata=data, testdata=testdata, lenFlag=False, func=misclassifcation_error)[0])

print(getReport(traindata=data, testdata=testdata, lenFlag=False, posFlag=False)[0])

print(getReport(traindata=data, testdata=testdata, lenFlag=False, posFlag=False, func=entropy)[0])

print(getReport(traindata=data, testdata=testdata, lenFlag=False, posFlag=False, func=misclassifcation_error)[0])


#Error Analysis
def getWrongPrediction(prediction, actual, dataset):
    data_list = []
    
    for i in range(len(prediction)):
        if prediction[i] !=actual[i] :
            data_list.append(dataset[i])  
    return data_list

_ , root_gini, prediction_gini, actual_gini  = getReport(traindata=data, testdata=testdata)
wrong_data = getWrongPrediction(prediction_gini, actual_gini, testdata)

len(wrong_data)

_ , root_entropy, prediction_entropy, actual_entropy  = getReport(traindata=data, testdata=wrong_data, func=entropy)
wrong_data_en = getWrongPrediction(prediction_entropy, actual_entropy, wrong_data)
len(wrong_data_en)

_ , root_mis, prediction_mis, actual_mis  = getReport(traindata=data, testdata=wrong_data, func=misclassifcation_error)
wrong_data_mis = getWrongPrediction(prediction_entropy, actual_entropy, wrong_data)
len(wrong_data_mis)