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
		self.vecGenomes = []
		# for i in range(popSize):
		# 	weights = []
		# 	for j in range(self.chromoLength):
		# 		weights.append(random.uniform(-1, 1))
		# 	g = Genome(weights, 0)
		# 	vecPop.append(g)

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


class GenAlg:
	def __init__(self, mutRate, crossRate):
		self.generation = 0

		self.mutationRate = mutRate # 0.05 to 0.3
		self.crossoverRate = crossRate # 0.7

	# One-point crossover
	def crossover(self, mumWeights, dadWeights):
		if random.random() < self.crossoverRate:
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

	def getnBest(self, nBest, numCopies, pop):
		#!!!! TODO
		
	def reset(self):
		#!!!! TODO


	# Generates new generation population through fitness-based selection
	# and combination of genetic operations: crossover and mutation
	# return list of genomes
	def evolve(self, pop): 
		newGenomes = []
		pop.calcFitnessFields()

		done = False
		while not done:
			mumWeights = rouletteSelect(pop).weights
			dadWeights = rouletteSelect(pop).weights
			daughters = crossover(mumWeights, dadWeights)
			for d in daughters:
				g = Genome(mutate(d), 0)
				if pop.popSize > len(newGenomes):
					newGenomes.append(g)
				else:
					done = True

		return newGenomes






