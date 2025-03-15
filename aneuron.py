import numpy as np

class Layer:
	def __init__(self, n_nums, bias=1.0):
		self.neurons = [Neuron(bias) for _ in range(n_nums)] # lista di neroni, in base all'input che gli viene dato
		self.prev_layer = None
		self.next_layer = None
		self.weights_to_next = None
		self.name = None

	def set_name(self, name):
		self.name = name
		return self

	def connect_to(self, next_layer, initial_weights=None): 
		self.next_layer = next_layer
		next_layer.prev_layer = self
		
		n_current = len(self.neurons)
		n_next = len(next_layer.neurons)

		if initial_weights is None:
			self.weights_to_next = np.random.randn(n_current, n_next) * np.sqrt(2.0 / (n_current + n_next))
		else:
			self.weights_to_next = initial_weights

		return self

	def forward(self): 
		if self.next_layer is None: 
			return
		
		current_outputs = np.array([neuron.get_output() for neuron in self.neurons])

		next_inputs = np.dot(current_outputs, self.weights_to_next)

		for i, neuron in enumerate(self.next_layer.neurons): 
			neuron.output = neuron.compute_output(next_inputs[i])

		self.next_layer.forward()

	def set_inputs(self, input_values): 
		if len(input_values) != len(self.neurons):
			raise ValueError(f"Expected {len(self.neurons)} input values, got {len(input_values)}")

		for neuron, value in zip(self.neurons, input_values):
			neuron.set_output(value)
		
	def get_outputs(self):
		return [neuron.get_output() for neuron in self.neurons]

	def print_state(self):
			"""Stampa lo stato corrente del layer (per debug)"""
			layer_name = self.name if self.name else "Unnamed Layer"
			print(f"\n--- {layer_name} ---")
			print(f"Number of neurons: {len(self.neurons)}")
			
			print("Neuron outputs:")
			for i, neuron in enumerate(self.neurons):
					print(f"  Neuron {i}: {neuron.get_output():.4f}")
			
			if self.weights_to_next is not None:
					print(f"Weights to next layer (shape: {self.weights_to_next.shape}):")
					print(self.weights_to_next)	

class Neuron:
	def __init__(self, bias):
		self.bias = bias
		self.output = 0

	def get_bias(self):
		return self.bias

	def get_output(self):
		return self.output
	
	def set_output(self, value):
		self.output = value

	def activate(self, x): 
		return max(0, x)

	def compute_output(self, net_input):
		return self.activate(net_input + self.bias)



# Creare una rete con 3 layer
input_layer = Layer(3).set_name("Input Layer")
hidden_layer = Layer(2).set_name("Hidden Layer")
output_layer = Layer(1).set_name("Output Layer")

# Connettere i layer
input_layer.connect_to(hidden_layer)
hidden_layer.connect_to(output_layer)

# Impostare i valori di input
input_layer.set_inputs([0.5, 0.2, 0.1])

# Stampare lo stato iniziale
input_layer.print_state()
hidden_layer.print_state()
output_layer.print_state()

# Propagare in avanti
input_layer.forward()

# Stampare lo stato finale
print("\n=== After Forward Propagation ===")
input_layer.print_state()
hidden_layer.print_state()
output_layer.print_state()

# Ottenere l'output finale
final_output = output_layer.get_outputs()
print("\nFinal output:", final_output)

# old implementation
""" def cni(j, w, b): # cni stands for compute_neuron_input
	j_with_bias = [1] + j.copy()
	w_with_bias = [b] + w.copy()

	jblen = len(j_with_bias)
	wblen = len(w_with_bias)

	acc = 0

	if (jblen != wblen):
		return "Vector j and w should have same length"

	for i in range(jblen):
		acc = acc + (j_with_bias[i] * w_with_bias[i])

	return max(0, acc)

print(cni(j, w, b)) """
