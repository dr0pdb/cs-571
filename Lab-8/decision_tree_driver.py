import json
from operator import itemgetter

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

modified_dataset = {}
bag_of_words = {}
vocabulary = []
feature_index = {}
X = []
Y = []
category_to_consider = ['business', 'comedy', 'sports', 'crime', "religion"]
pos_tags_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']


def preprocess_nltk(headline):
    stopword = set(stopwords.words("english"))
    y = ""
    for word in headline.split():
        if word not in stopword:
            y += word+" "
    # y = re.sub(r'[^\w\s]', '', y)
    # print(y)
    return y


def run_decision_tree():
    classifier = DecisionTreeClassifier()
    score = cross_val_score(classifier, X, Y, scoring="accuracy", cv=10)
    print(sum(score)/len(score))

def main():
    with open('News_Category_Dataset.json') as json_data:
        d = json.load(json_data)
        for object in d:
            if object['category'].strip().lower() not in category_to_consider: continue
            if object['category'] in modified_dataset:
                modified_dataset[object['category']].append(object['headline'].lower())
            else:
                modified_dataset[object['category']] = [object['headline'].lower()]

    populate_n_gram(1, 500)
    populate_n_gram(2, 300)
    populate_n_gram(3, 200)
    vocabulary.extend(pos_tags_list)
    for i in range(len(vocabulary)):
        feature_index[vocabulary[i]] = i
    for category, headlines in modified_dataset.items():
        for headline in headlines:
            features = [0] * len(vocabulary)
            for vocab in vocabulary:
                if vocab in headline:
                    features[feature_index[vocab]] = 1
            tokens = word_tokenize(headline)
            pos_tuples = pos_tag(tokens=tokens)
            for pos_tuple in pos_tuples:
                if pos_tuple[1] in pos_tags_list:
                    features[feature_index[pos_tuple[1]]] = 1
            X.append(features)
            Y.append(category)

    run_decision_tree()


def populate_n_gram(ngram, count):
    for category, headlines in modified_dataset.items():
        for headline in headlines:
            headline = preprocess_nltk(headline)
            vocabs = headline.split(" ")
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
