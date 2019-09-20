import numpy as np
import matplotlib.pyplot as plt

# Activation function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# Returns the derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initial input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

# Actual expected outputs
expected_output = np.array([[0],[1],[1],[0]])

# Number of iterations
epochs = 10000

# Initial learning rate
learning_rate = 0.0

# x coordinate - learning rate
# y coordinate - error
x = []
y = []

for i in range(1, 12):
	print("Learning rate: ", learning_rate)
	x.append(learning_rate)
	inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

	#Random weights and bias initialization
	hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
	hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
	output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
	output_bias = np.random.uniform(size=(1,outputLayerNeurons))

	#Training algorithm
	error = []
	for _ in range(epochs):
		#Forward Propagation
		hidden_layer_activation = np.dot(inputs,hidden_weights)
		hidden_layer_activation += hidden_bias
		hidden_layer_output = sigmoid(hidden_layer_activation)

		output_layer_activation = np.dot(hidden_layer_output,output_weights)
		output_layer_activation += output_bias
		predicted_output = sigmoid(output_layer_activation)

		#Backpropagation
		error = expected_output - predicted_output
		d_predicted_output = error * sigmoid_derivative(predicted_output)
		
		error_hidden_layer = d_predicted_output.dot(output_weights.T)
		d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

		#Updating Weights and Biases
		output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
		output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * learning_rate
		hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
		hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * learning_rate

	total_error = 0.0
	for e in error:
		total_error += abs(e) * abs(e)
	y.append(total_error)
	print("Output from neural network after 10,000 epochs: ",end='')
	print(*predicted_output)
	print("\n\n")
	learning_rate = learning_rate + 0.1

# Plot the graph of error vs learning rate
plt.plot(x, y)
plt.show()