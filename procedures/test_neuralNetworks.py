import procedures.neuralNetworks as nn

class test_neuralNetworkBinaryClassifier():
	def test_initialization(self):
		instance = nn.binaryClassifier(2, 2)
		#Test that the first "input" has 3 sigmoids
		assert len(instance.layers[0]) == 3
		#Test that the sigmoids in the first layer have 3 weights
		assert len(instance.layers[0][0]) == 3
		#Test that the second layer has 3 sigmoids
		assert len(instance.layers[1]) == 3
		#Test that sigmoids in the second layer have 4 weights
		assert len(instance.layers[1][0]) == 4

	def test_randomArray(self):
		instance = nn.binaryClassifier(2, 2).randomArray(1)
		assert instance[0] <= 1
		assert instance[0] >= -1

	def test_sigmoid(self):
		assert nn.binaryClassifier(2, 2).sigmoid(0) == 0.5

	def test_randomNudge(self):
		instance = nn.binaryClassifier(2, 2)
		newNN = instance.randomNudge()
		numberDifferent = 0
		for layer in range(instance.numLayers):
			for sigmoid in range(len(instance.layers[layer])):
				for weight in range(len(instance.layers[layer][sigmoid])):
					if (not instance.layers[layer][sigmoid][weight] == newNN.layers[layer][sigmoid][weight]):
						numberDifferent += 1
		assert numberDifferent == 1
