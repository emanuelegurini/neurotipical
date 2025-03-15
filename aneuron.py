class Neuron():
	def __init__(self, bias):
		self.bias = bias
		self.incoming_connections = []

	def get_bias(self):
		return self.bias
	
	def add_incoming_connection(self, incoming_connection):
		self.incoming_connections.append(incoming_connection)

	def get_incoming_connections(self):
			return self.incoming_connections
	
	def get_output(self):
		pass

class Connection():
	def __init__(self, source_neuron, target_neuron, weight):
		self.source_neuron = source_neuron
		self.target_neuron = target_neuron
		self.weight = weight

	def get_source_neuron(self):
		return self.source_neuron

	def get_target_neuron(self):
		return self.target_neuron

	def get_weight(self):
		return self.weight

neuron_a = Neuron(bias=1)
neuron_b = Neuron(bias=1)

ab_conn = Connection(neuron_a, neuron_b, 2)
neuron_b.add_incoming_connection(ab_conn)

print(neuron_b.get_incoming_connections())

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
