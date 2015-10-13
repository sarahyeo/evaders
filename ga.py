import random
import sys
import math

### 
### GENETIC ALGORITHM
###

# Steps TODO:
# INIT
# - create sprites with field containg NN
# - get total amont of weights in each NN in sprite
# - use num of sprites made and weights in each NN to initalize pop
# - get weights from pop and insert into NN in sprite
# UPDATE
# - sprite.update(inputs)
# - update each NN in sprite, update sprite accoring to outputs of NN
# - updte fitness of each sprite, update pop genome fitness accordingly
# GENERATION OVER
# - evolve to create new pop
# - insert new weights int NN in sprite
# - reset sprite & fitness

class Neuron:
	def __init__(self, inputs):
		# num inputs into neuron plus additional weight for the bias
		self.numInputs = inputs + 1
		# weights for each input
		self.vecWeights = []
		# set up the weights with random initial value - will be updated with putWeights
		for i in range(self.numInputs):
			self.vecWeights.append(random.random())

class NeuronLayer:
	def __init__(self, numNeurons, numInputsPerNeuron):
		self.numNeurons = numNeurons
		# list of neurons in layer
		self.vecNeurons = [] 
		for i in range(numNeurons):
			n = Neuron(numInputsPerNeuron)
			self.vecNeurons.append(n)

class NeuralNet:
	# Link up neurons in feedforward netowrk
	def __init__(self, numInputs, numOutputs, numHiddenLayers, neuronsPerHiddenLyr):
		self.numInputs = numInputs
		self.numOutputs = numOutputs
		self.numHiddenLayers = numHiddenLayers
		self.neuronsPerHiddenLyr = neuronsPerHiddenLyr
		self.vecLayers = []

		if numHiddenLayers > 0:
			# create first hidden layer - accenpts input from input layer
			hl1 = NeuronLayer(neuronsPerHiddenLyr, numInputs)
			vecLayers.append(hl1)

			# create any other hidden layers - accepts input from previous hidden layer
			for i in range(numHiddenLayers-1):
				hli = NeuronLayer(neuronsPerHiddenLyr, neuronsPerHiddenLyr)
				vecLayers.append(hli)

			# create output layer - accepts input from hidden layer
			ol = NeuronLayer(numOutputs, neuronsPerHiddenLyr)
			vecLayers.append(ol)
		else:
			# no hidden layers
			# crete output layer - accempts input from input layer
			ol = NeuronLayer(numOutputs, numInputs)
			vecLayers.append(ol)


	# gets the weights from the NN (return list of weights)
	def getWeights(self):
		index = 0
		weights = []
		# for each hidden laery + output layer
		for i in range(numHiddenLayers + 1):
			# for each neruon in layer
			for k in range(vecLayers[i].numNeurons):
				# for each weight in neuron - including bias
				for m in range(vecLayers[i].vecNeurons[k].numInputs):
					weights.append(vecLayers[i].vecNeurons[k].vecWeights[m])
		return weights

	# returns the total number of weights in the net
	def getNumOfWeights(self):
		return len(self.getWeights())

	# replaces the weights with new ones
	def putWeights(self, weights):
		index = 0
		# for each hidden laery + output layer
		for i in range(numHiddenLayers + 1):
			# for each neruon in layer
			for k in range(vecLayers[i].numNeurons):
				# for each weight in neuron - including bias
				for m in range(vecLayers[i].vecNeurons[k].numInputs):
					vecLayers[i].vecNeurons[k].vecWeights[m] = weights[index]
					index += 1 

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
					netInput += vecLayers[i].vecNeurons[j].vecWeights[k]*inputs[index]
					index += 1

				# add the bias
				# wSum - T, where T is threashold/bias value	
				netInput += vecLayers[i].vecNeurons[j].vecWeights[numInputs-1]*(-1)

				# F(sWum-T), where F is sigmoid function
				# add result to outputs
				outputs.append(sigmoid(netInput, 1.0))
		return outputs

class Genome:
	def __init__(self, weights, fitness):
		# array of weights starting at bottom layer traversing left to right
		self.vecWeights = weights
		self.fitness = fitness

class Population:
	def __init__(self, popSize, numWeights):
		self.popSize = popSize
		# number of weights in each genome
		self.chromoLength = numWeights

		self.totalFitness = 0
		self.bestFitness = 0
		self.averageFitness = 0
		self.worstFitness = 0
		self.indexBestFitness = 0 

		# initalize all genomes with random weights
		# Each genome allocated to respective NeuralNetwork via putWeights
		self.vecGenomes = []
		for i in range(popSize):
			weights = []
			for j in range(self.chromoLength):
				weights.append(random.random())
			g = Genome(weights, 0)
			vecPop.append(g)

	def calcBestFitneess(self):
		fitnesses = map(lambda genome: genome.fitness, self.vecGenomes)
		self.bestFitness = max(fitnesses)
		self.indexBestFitness = fitnesses.index(self.bestFitness)

	def calcWorstFitness(self):
		self.worstFitness = min(map(lambda genome: genome.fitness, self.vecGenomes))

	def calcAverageFitness(self):
		self.totalFitness = sum(map(lambda genome: genome.fitness, self.vecGenomes))
		self.averageFitness = self.totalFitness/self.popSize

	def calcFitnessFields(self):
		self.calcBestFitneess()
		self.calcWorstFitness()
		self.calcAverageFitness()

	def replacePop(self, new_pop):
		self.vecGenomes = new_pop

	def resetPop(self):
		self.totalFitness = 0
		self.bestFitness = 0
		self.averageFitness = 0
		self.worstFitness = 9999999;



class GenAlg:
	def __init__(self, mutRate, crossRate):
		self.generation = 0

		self.mutationRate = mutRate # 0.05 to 0.3
		self.crossoverRate = crossRate # 0.7

	# One-point crossover
	def crossover(self, mumWeights, dadWeights):
		if random.random() > self.crossoverRate or mumWeights == dadWeights:
			return (mumWeights, dadWeights)
		else:
			crossPoint = random.randint(0, len(mumWeights))
			daughter1 = mumWeights[:crossPoint] + dadWeights[crossPoint:]
			daughter2 = dadWeights[:crossPoint] + mumWeights[crossPoint:]
			return (daughter1, daughter2)

	# Uniform mutation - replaces value at chosen gene with rand weight
	# return list of weights
	def mutate(self, weights):
		for i in range(weights):
			if random.random() < mutationRate:
				weights[i] = random.uniform(-1, 1)
		return weights

	# Implemented using stochastic acceptance - O(1) time
	# Selected genome i accepted with probability fittness[i]/totalFittness
	# return genome
	def rouletteSelect(self, pop):
		while True:
			index = random.randint(0, pop.popSize)
			if random.random() < pop.vecGenomes[index].fitness/pop.totalFitness:
				return pop.vecGenomes[index]
		



	# Generates new generation population through fitness-based selection
	# and combination of genetic operations: crossover and mutation
	# return list of genomes
	def evolve(self, pop): 
		newGenomes = []
		pop.reset()
		pop.calcFitnessFields()

		done = False
		while not done:
			# select two chromosones using roulette wheel selection
			mumWeights = rouletteSelect(pop).weights
			dadWeights = rouletteSelect(pop).weights
			# create offspring via crossover
			daughters = crossover(mumWeights, dadWeights)
			for d in daughters:
				# now we mutate
				g = Genome(mutate(d), 0)
				if pop.popSize > len(newGenomes):
					newGenomes.append(g)
				else:
					done = True

		pop.replacePop(newGenomes)
		return newGenomes

