import random
import sys
import math

### 
### GENETIC ALGORITHM
###

class Neuron:
	def __init__(self, inputs):
		# num inputs into neuron plus additional weight for the bias
		self.numInputs = inputs + 1
		# weights for each input
		self.vecWeights = []
		# set up the weights with initial random value
		for i in range(self.numInputs):
			self.vecWeights.append(random.uniform(-1, 1))

class NeuronLayer:
	def __init__(self, numNeurons, numInputsPerNeuron):
		self.numNeurons = numNeurons
		# list of neurons in layer
		self.vecNeurons = [] 
		for i in range(numNeurons):
			n = Neuron(numInputsPerNeuron)
			self.vecNeurons.append(n)

class NeuralNet:
	def __init__(self, numInputs, numOutputs, numHiddenLayers, neuronsPerHiddenLyr, layers):
		self.numInputs = numInputs
		self.numOutputs = numOutputs
		self.numHiddenLayers = numHiddenLayers
		self.neuronsPerHiddenLyr = neuronsPerHiddenLyr
		self.vecLayers = layers

	# gets the weights from the NN (return list of weights)
	def getWeights(self):
		index = 0
		weights = []
		for i in range(numHiddenLayers):
			for k in range(vecLayers[i].numNeurons):
				for m in range(vecLayers[i].vecNeurons[k].numInputs-1):
					weights.append(vecLayers[i].vecNeurons[k].vecWeights[m])
		return weights

	# returns the total number of weights in the net
	def getNumOfWeights(self):
		return len(self.getWeights())

	# replaces the weights with new ones
	def putWeights(self, weights):
		index = 0
		for i in range(numHiddenLayers):
			for k in range(vecLayers[i].numNeurons):
				for m in range(vecLayers[i].vecNeurons[k].numInputs-1):
					vecLayers[i].vecNeurons[k].vecWeights[m] = weights[index++] 

	# sigmoid response curve
	def sigmoid(self, activiation, response):
		i = -activiation/response
		return 1/(1+ 2.7183**i)

	# calculates the outputs from a set of inputs
	def update(self, inputs):
		outputs = []
		if len(inputs) != self.numInputs:
			print ("Error - numn Inputs do not match")
			return outputs

		# for each layer in neural network
		for i in range(self.numHiddenLayers): 
			if i > 0:
				# feed outputs on previous layer into current layer
				inputs = outputs
			outputs = []

			# for each neuron in current layer
			for j in range(vecLayers[i].numNeurons): 
				index = 0
				netInput = 0
				numInputs = vecLayers[i].vecNeurons[j].numInputs

				# for each input into neuron (minus the bias) sum weight*input
				# wSum = sum from i=1 to n (weight[i] * input[i]), n is numInputs
				for k in range(numInputs-1): 
					netInput += vecLayers[i].vecNeurons[j].vecWeights[k]*inputs[index++]

				# add the bias
				# wSum - T, where T is threashold/bias value	
				netInput += vecLayers[i].vecNeurons[j].vecWeights[numInputs-1]*(-1)

				# F(sWum-T), where F is sigmoid function
				# add result to outputs
				outputs.append(sigmoid(netInput, 1.0))
		return outputs

		