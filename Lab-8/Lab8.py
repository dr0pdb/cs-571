import numpy as np 
import pandas as pd
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler 
from sklearn import datasets
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
import re
import codecs
import os
import re
import string
import math
import nltk 
from nltk.util import ngrams
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize 

# reading the corpus

def read_corpus(corpus_file):
    out = []
    with codecs.open(corpus_file, 'r', encoding='utf-8',
                 errors='ignore') as f:
        i = 0
        for line in f:
            tokens = re.findall(r"[\w']+", line)
            out.append((tokens[0], tokens[1], tokens[2:]))
    return out


all_docs = read_corpus('train_5500.label.txt')
print(all_docs[0][2])

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english')) 

# Getting the POS Tagging.

i=0

pos_tags = []

while i < len(all_docs):
  tokenized = all_docs[i][2]
  tmp = []

  # Word tokenizers is used to find the words  
  # and punctuation in a string 
  wordsList = all_docs[i][2]

  # removing stop words from wordList 
  wordsList = [w for w in wordsList if not w in stop_words]  

  #  Using a Tagger. Which is part-of-speech  
  # tagger or POS-tagger.  
  tagged = nltk.pos_tag(wordsList) 
  tmp = tagged
  pos_tags.append(tmp)   
  i = i+1

  
# Most frequent unigrams
i = 1
a_u = ngrams(all_docs[0][2],1)
a_u = Counter(a_u)
while i < len(all_docs):
  unigrams = ngrams(all_docs[i][2],1)
  b = Counter(unigrams)
  dict.update(a_u,b)
  i = i+1
  
# Most frequent bigrams
i = 1
a_b = ngrams(all_docs[0][2],2)
a_b = Counter(a_b)
while i < len(all_docs):
  bigrams = ngrams(all_docs[i][2],2)
  b = Counter(bigrams)
  dict.update(a_b,b)
  i = i+1
  
# Most frequent trigrams
i = 1
a_t = ngrams(all_docs[0][2],3)
a_t = Counter(a_t)
while i < len(all_docs):
  trigrams = ngrams(all_docs[i][2],3)
  b = Counter(trigrams)
  dict.update(a_t,b)
  i = i+1
  
most_frequent_uni = a_u.most_common(500)
most_frequent_bi = a_b.most_common(300)
most_frequent_tri = a_t.most_common(200)

print(most_frequent_uni)
print(most_frequent_bi)
print(most_frequent_tri)
