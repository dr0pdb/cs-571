import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import pickle

load_from_pickle = False
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

if load_from_pickle:
    with open("data/pickled_mnist.pkl", "br") as fh:
        data = pickle.load(fh)
    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]
    train_labels_one_hot = data[4]
    test_labels_one_hot = data[5]
else :
    data_path = "data/"
    train_data = np.loadtxt(data_path + "mnist_train.csv",
                            delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv",
                           delimiter=",")
    test_data[:10]

    print('Loaded the dataset')

    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    lr = np.arange(no_of_different_labels)
    # transform labels into one hot representation
    train_labels_one_hot = (lr==train_labels).astype(np.float)
    test_labels_one_hot = (lr==test_labels).astype(np.float)
    # we don't want zeroes and ones in the labels neither:
    train_labels_one_hot[train_labels_one_hot==0] = 0.01
    train_labels_one_hot[train_labels_one_hot==1] = 0.99
    test_labels_one_hot[test_labels_one_hot==0] = 0.01
    test_labels_one_hot[test_labels_one_hot==1] = 0.99

    # dump to pickle.
    with open("data/pickled_mnist.pkl", "bw") as fh:
        data = (train_imgs,
                test_imgs,
                train_labels,
                test_labels,
                train_labels_one_hot,
                test_labels_one_hot)
        pickle.dump(data, fh)

print('Transformed the images into one-hot representation')

# Training
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)

class NeuralNetwork:
    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0,
                             sd=1,
                             low=-rad,
                             upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes,
                                       self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0,
                             sd=1,
                             low=-rad,
                             upp=rad)
        self.who = X.rvs((self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))


    def train_single(self, input_vector, target_vector):
        output_vectors = []
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.wih,
                                input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who,
                                output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network * \
              (1.0 - output_network)
        tmp = self.learning_rate  * np.dot(tmp,
                                           output_hidden.T)
        self.who += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T,
                               output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)

    def train(self, data_array,
              labels_one_hot_array,
              epochs=1,
              intermediate_results=False):
        intermediate_weights = []
        for epoch in range(epochs):
            print("*", end="")
            for i in range(len(data_array)):
                self.train_single(data_array[i],
                                  labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append((self.wih.copy(),
                                             self.who.copy()))
        return intermediate_weights

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def run(self, input_vector):
        """ input_vector can be tuple, list or ndarray """

        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih,
                               input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.who,
                               output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

# Create the neural network.
current_learning_rate = 0.1
epochs = 5

for iterations in range(5):
    ANN = NeuralNetwork(no_of_in_nodes = image_pixels,
                                   no_of_out_nodes = 10,
                                   no_of_hidden_nodes = 100,
                                   learning_rate = current_learning_rate)

    weights = ANN.train(train_imgs,
                        train_labels_one_hot,
                        epochs=epochs,
                        intermediate_results=True)

    for i in range(epochs):
        print("epoch: ", i)
        ANN.wih = weights[i][0]
        ANN.who = weights[i][1]

        corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
        print("Accuracy train: ", corrects / ( corrects + wrongs))
        corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
        print("Accuracy test: ", corrects / ( corrects + wrongs))

        cm = ANN.confusion_matrix(train_imgs, train_labels)
        print(cm)
        for j in range(10):
            print("digit: ", j, "precision: ", ANN.precision(j, cm), "recall: ", ANN.recall(j, cm))
        print('----------')

    current_learning_rate += 0.05
