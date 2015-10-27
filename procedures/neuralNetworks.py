import random, math

class binaryClassifier():
	def __init__(self, inputs, layers):
		self.layers = []
		self.numInputs = inputs
		self.numLayers = layers
		#First layer code
		data = []
		for l in range(inputs + 1):
			data.append(self.randomArray(inputs + 1))
		self.layers.append(data)

		#Next layers
		for l in range(layers - 1):
			data = []
			for j in range(inputs + 1):
				data.append(self.randomArray(inputs + 2))
			self.layers.append(data)


	def randomArray(self, length):
		data = []
		for i in range(length):
			data.append(random.uniform(-1,1))
		return data

	def sigmoid(self, x):
		return 1 / (1 + float(math.exp(-x)))

	def randomNudge(self):
		#Create a new NN
		newNN = binaryClassifier(self.numInputs, self.numLayers)
		for layer in range(self.numLayers):
			for sigmoid in range(len(self.layers[layer])):
				for weight in range(len(self.layers[layer][sigmoid])):
					newNN.layers[layer][sigmoid][weight] = self.layers[layer][sigmoid][weight]
		#Adjust a random weight
		randomLayer = random.choice(range(self.numLayers))
		randomSigmoid = random.choice(range(len(self.layers[randomLayer])))
		randomWeight = random.choice(range(len(self.layers[randomLayer][randomSigmoid])))
		newNN.layers[randomLayer][randomSigmoid][randomWeight] += random.uniform(-1, 1)
		return newNN

	def predict(self):
		return

	def weights(self):
		return

if (__name__ == "__main__"):
	j = binaryClassifier(2, 2)
	k = j.randomNudge()
	print k.layers
	print ("\n")
	print j.layers
