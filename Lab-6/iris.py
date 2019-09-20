from sklearn import datasets
import numpy as np
from mlxtend.preprocessing import shuffle_arrays_unison
from matplotlib import pyplot as plt
import random

np.random.seed(3)

# scrambles data
def inputn(x):
    inputt = []
    for i in range(50):
        inputt.append(x[i])
        inputt.append(x[i + 50])
        inputt.append(x[i + 100])
    return np.array(inputt)

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    # prevent overflow by limiting min and max range
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

def cross_entropy(output, y_target):
    return - np.sum(np.log(output) * (y_target))

# calculates cost
def cost(output, y_target):
    return np.mean(cross_entropy(output, y_target))

#soft max outputs set of probability values
def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()

# dropout function
def dropout(a, prob):
    shape = a.shape[0]
    vec = np.random.choice([0,1], size = (shape,1), p = [prob, 1-prob])
    return vec * a

#load dataset
iris = datasets.load_iris()
data = iris.data
actual = iris.target

# scramble arrays
data = inputn(data)
actual = inputn(actual)
# shuffle the arrays the same
data, actual = shuffle_arrays_unison(arrays=[data, actual], random_seed=3)

# initialize weights
syn0 = np.random.randn(4,4) / 2
syn1 =  np.random.randn(3,4) /2
bias0 =  np.random.randn(4,1) /2
bias1 = np.random.randn(3,1) /2

# stores array of
xarr = []
yarr = []

rate_syn = 0.1
# rate_bias = 0.1
loss_step = 0

for i in range(1, 10):
    loss = 0
    for yo in range(200):
        for i in range(120):
            # forward prop
            l0 = data[i][None].T
            l0 = dropout(l0, .25)
            l1 = nonlin(np.dot(syn0, l0) + bias0)
            l1 = dropout(l1,.25)
            l2 = softmax(np.dot(syn1, l1) + bias1)
            # setup target
            target = np.zeros([3,1])
            target[actual[i]][0] = 1
            # back prop
            syn1_delta = np.outer((l2-target), l1.T)
            bias1_delta = -(l2-target)
            error_hidden = np.dot(syn1.T, (l2 - target)) * nonlin(l1, True)
            syn0_delta = np.outer(error_hidden,l0.T)
            bias0_delta = -error_hidden

            # adjust weights with learning rate
            syn0 -= rate_syn * syn0_delta
            syn1 -= rate_syn * syn1_delta
            bias0 +=  bias0_delta
            bias1 +=  bias1_delta


    # prints number of correct
    training_correct = 0

    for i in range(120):
        l0 = data[i][None].T
        l1 = nonlin(np.dot(syn0, l0) + bias0)
        l2 = softmax(np.dot(syn1, l1) + bias1)
        if np.argmax(l2) == actual[i]:
            training_correct += 1
    print(training_correct)
    print("Accuracy: "+str(training_correct/120))

    loss_step += 120 - training_correct

    test_correct = 0


    for i in range(120,150):
        l0 = data[i][None].T
        l1 = nonlin(np.dot(syn0, l0) + bias0)
        l2 = softmax(np.dot(syn1, l1) + bias1)
        if np.argmax(l2) == actual[i]:
            test_correct += 1
    print(test_correct)

    loss_step += 30 - test_correct

    loss_step /= 150

    xarr.append(rate_syn)
    yarr.append(loss_step)

    rate_syn += 0.1




# prints plot
plt.plot(xarr, yarr, '-')
plt.xlabel('Rate')
plt.ylabel('Loss')
# plt.axvline(90,color = 'r')
# # plt.suptitle("CrossEntropy loss time graph")
# # indicates where overfitting may occur, since no regularization methods were performed
# plt.text(40, 1, 'The red line is where overfitting most likely occurs',fontsize=10, wrap = True)
plt.show()
