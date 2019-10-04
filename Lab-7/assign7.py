import math
import numpy as np
import pandas as pd


# Read the entire corpus from 'corpus_file'. Assume correct format and no
# missing values.
# Return a list of pairs of labels and texts.
def read_corpus(corpus_file):
    out = []
    with open(corpus_file, encoding='utf8') as f:
        for line in f:
            tokens = line.strip().split()
            out.append( (tokens[1], tokens[3:]) )
        return out


# Estimate the probabilities in the Naive Bayes model.
# Return 3 objects: the probability of a review being negative, the log
# probabilities of each words being in a negative...
def train_nb(training_data, alpha):
    num_neg = 0
    vocab = []
    counts_neg = []
    counts_pos = []
    num_tokens_neg = 0
    num_tokens_pos = 0
    for label, tokens in training_data:
        if label == 'neg':
            num_neg += 1
            num_tokens_neg += len(tokens)
        else:
            num_tokens_pos += len(tokens)
        for token in tokens:
            if token in vocab:
                idx = vocab.index(token)
            else:
                idx = len(vocab)
                vocab.append(token)
                counts_neg.append(0)
                counts_pos.append(0)
            if label == 'neg':
                counts_neg[idx] += 1
            else:
                counts_pos[idx] += 1
    vocab_size = len(vocab)
    probs_neg = list(map(lambda c: math.log(laplace_smoothing(c, num_tokens_neg, vocab_size, alpha)), counts_neg))
    probs_pos = list(map(lambda c: math.log(laplace_smoothing(c, num_tokens_pos, vocab_size, alpha)), counts_pos))
    probs_unknowns = (math.log(laplace_smoothing(0, num_tokens_neg, vocab_size+1, alpha)), math.log(laplace_smoothing(0, num_tokens_pos, vocab_size+1, alpha)))
    df_probs = pd.DataFrame(index=vocab)
    df_probs['neg'] = probs_neg
    df_probs['pos'] = probs_pos
    return (num_neg / len(training_data), df_probs, probs_unknowns)


def laplace_smoothing(count, num_tokens, vocab_size, alpha):
    return (count+alpha) / (num_tokens+vocab_size*alpha)


def classify_nb(classifier_data, document):
    probs_sentiment, df_probs, probs_unknowns = classifier_data
    vocab = df_probs.index.values.tolist()
    prob_neg, prob_pos = probs_sentiment
    for token in document:
        if token in vocab:
            probs = df_probs.loc[token, :].tolist()
        else:
            probs = probs_unknowns
        prob_neg += probs[0]
        prob_pos += probs[1]
    return 'neg' if prob_neg > prob_pos else 'pos'


def evaluate_nb(classifier_data, testing_data):
    df_test = pd.DataFrame(testing_data, columns=['labels', 'texts'])
    predictions = df_test['texts'].apply(lambda doc: classify_nb(classifier_data, doc))
    result = {}
    result['accuracy'] = (df_test['labels'] == predictions).sum() / len(predictions)
    total_pos =  0
    true_pos = 0
    false_neg = 0
    for i in range(0, len(predictions)):
        if predictions[i] is 'pos':
            total_pos = total_pos + 1
            if df_test['labels'][i] == 'pos':
                true_pos = true_pos + 1
        else:
            if df_test['labels'][i] == 'pos':
                false_neg = false_neg + 1
    result['precision'] = true_pos / total_pos
    result['recall'] = true_pos / (true_pos + false_neg)
    return result


def main():
    labeled_corpus = read_corpus('all_sentiment_shuffled.txt')
    n = len(labeled_corpus)
#     print(n)
    window_size = math.floor(n/5)
    i = 0
    while i<n: 
        begin = i
        end  = i+ window_size


        training_data = labeled_corpus[:begin]

        if end + window_size>n:
            end = n
            i = i + window_size

        testing_data = labeled_corpus[begin:end]
        training_data += labeled_corpus[end:]
        prob_neg, df_probs, probs_unknowns = train_nb(training_data, 1)
        probs_sentiment = (math.log(prob_neg), math.log(1-prob_neg))
        classifier_data = (probs_sentiment, df_probs, probs_unknowns)
        print(begin, end)
        result = evaluate_nb(classifier_data, testing_data)
        result['fscore'] = 2 * ((result['precision'] * result['recall'])/(result['precision'] + result['recall']))
        print('Classication accuracy for the test set is: {}'.format(result['accuracy']))
        print('Precision is: {}'.format(result['precision']))
        print('Recall is: {}'.format(result['recall']))
        print('F-score is: {}'.format(result['fscore']))
        i = i + window_size

if __name__ == '__main__':
    main()