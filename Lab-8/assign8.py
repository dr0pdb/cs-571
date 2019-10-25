# -*- coding: utf-8 -*-
"""Assign8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qR8osoYYSYEqWjHL1jeWk0Gk5hic7DKq
"""

import codecs
import re
from operator import itemgetter

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

train_docs = []
bag_of_words = {}
vocabulary = []
feature_index = {}
X = []
Y = []
category_to_consider = ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
pos_tags_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']


import nltk 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def preprocess_nltk(question):
    stopword = set(stopwords.words("english"))
    y = ""
    for word in headline.split():
        if word not in stopword:
            y += word+" "
    # y = re.sub(r'[^\w\s]', '', y)
    # print(y)
    return y


def run_decision_tree():
    classifier = DecisionTreeClassifier(max_depth=20)
    score = cross_val_score(classifier, X, Y, scoring="accuracy", cv=12)
    print(sum(score)/len(score))


def read_corpus(corpus_file):
    out = []
    with codecs.open(corpus_file, 'r', encoding='utf-8',
                 errors='ignore') as f:
        i = 0
        for line in f:
            tokens = re.findall(r"[\w']+", line)
            out.append([tokens[0], tokens[1], tokens[2:]])
    return out


def main():
    train_docs = read_corpus('train_5500.label.txt')
    
    for i in range(0, len(train_docs)):
        delimiter = ' '
        train_docs[i][2] = delimiter.join(train_docs[i][2])

    print(train_docs[0])

    populate_n_gram(1, 500)
    populate_n_gram(2, 300)
    populate_n_gram(3, 200)

    vocabulary.extend(pos_tags_list)

    for i in range(len(vocabulary)):
        feature_index[vocabulary[i]] = i
    for coarse_class, fine_class, question in train_docs:
        # print(coarse_class)
        # print(question)
        features = [0] * len(vocabulary)
        for vocab in vocabulary:
            if vocab in question:
                features[feature_index[vocab]] = 1
        tokens = word_tokenize(question)
        pos_tuples = pos_tag(tokens=tokens)
        for pos_tuple in pos_tuples:
            if pos_tuple[1] in pos_tags_list:
                features[feature_index[pos_tuple[1]]] = 1
        X.append(features)
        Y.append(coarse_class)

    run_decision_tree()


def populate_n_gram(ngram, count):
    for coarse_class, fine_class, question in train_docs:
        question = preprocess_nltk(question)
        vocabs = question.split(" ")
        for i in range(len(vocabs) - ngram + 1):
            string = ""
            for j in range(ngram):
                string += vocabs[i + j]
                string += " "
            if string.strip() not in bag_of_words:
                bag_of_words[string.strip()] = 1
            else:
                bag_of_words[string.strip()] += 1
    tot = 0
    for key, value in sorted(bag_of_words.items(), key=itemgetter(1), reverse=True):
        tot += 1
        vocabulary.append(key)
        if tot == count:
            break
            # print(feature_vector)
    bag_of_words.clear()


if __name__ == '__main__':
    main()