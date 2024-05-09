import neuro
import risp
import numpy as np
import matplotlib.pyplot as plt

risp_config = {
    'leak_mode': 'none',
    'min_weight': -1,
    'max_weight': 1,
    'min_threshold': -1,
    'max_threshold': 1,
    'max_delay': 5,
    'discrete': False
}

proc = risp.Processor(risp_config)
net = neuro.Network()
net.set_properties(proc.get_network_properties())

# Create a simple probability Network

g = net.add_node(0)   # Create a neuron with ID 0
g.set("Threshold", 0)
net.add_input(0)

p1 = net.add_node(1)
p1.set("Threshold", 1)

p2 = net.add_node(2)
p2.set("Threshold", 1)

p3 = net.add_node(3)
p3.set("Threshold", 1)

o1 = net.add_node(4)
o1.set("Threshold", 1)
net.add_output(4)

o2 = net.add_node(5)
o2.set("Threshold", 1)
net.add_output(5)

o3 = net.add_node(6)
o3.set("Threshold", 1)
net.add_output(6)

o4 = net.add_node(7)
o4.set("Threshold", 1)
net.add_output(7)

e1 = net.add_edge(0, 1)
e1.set("Weight", 0.4)
e1.set("Delay", 1)

e2 = net.add_edge(0, 2)
e2.set("Weight", 0.5)
e2.set("Delay", 1)

e3 = net.add_edge(0, 3)
e3.set("Weight", 1)
e3.set("Delay", 1)

e4 = net.add_edge(1, 4)
e4.set("Weight", 1)
e4.set("Delay", 1)

e5 = net.add_edge(1, 5)
e5.set("Weight", -1)
e5.set("Delay", 1)

e6 = net.add_edge(2, 4)
e6.set("Weight", 0.5)
e6.set("Delay", 1)

e7 = net.add_edge(2, 5)
e7.set("Weight", 0.5)
e7.set("Delay", 1)

e8 = net.add_edge(2, 6)
e8.set("Weight", -1)
e8.set("Delay", 1)

e9 = net.add_edge(2, 7)
e9.set("Weight", -1)
e9.set("Delay", 1)

e10 = net.add_edge(3, 6)
e10.set("Weight", 1)
e10.set("Delay", 1)

e11 = net.add_edge(3, 7)
e11.set("Weight", 0)
e11.set("Delay", 1)

e12 = net.add_edge(0, 7)
e12.set("Weight", 1)
e12.set("Delay", 3)

proc.load_network(net)
for i in range(net.num_nodes()):
    proc.track_neuron_events(i)

N = 1000
rand_nums = np.random.rand(N)
spikes = [neuro.Spike(id=0, time=i, value=rand_nums[i]) for i in range(N)]
proc.apply_spikes(spikes)

proc.run(N + 5)
v = proc.neuron_vectors()

o1_count = len(v[4])
o2_count = len(v[5])
o3_count = len(v[6])
o4_count = len(v[7])
total = o1_count + o2_count + o3_count + o4_count

print("Output 1: ", o1_count, "(", o1_count/total, ")")
print("Output 2: ", o2_count, "(", o2_count/total, ")")
print("Output 3: ", o3_count, "(", o3_count/total, ")")
print("Output 4: ", o4_count, "(", o4_count/total, ")")

plt.xlabel('Spike Times')
plt.ylabel('Neuron ID')

plt.plot(v[4], [4]*o1_count, 'ro', label='Output 1')
plt.plot(v[5], [5]*o2_count, 'bo', label='Output 2')
plt.plot(v[6], [6]*o3_count, 'go', label='Output 3')
plt.plot(v[7], [7]*o4_count, 'yo', label='Output 4')
plt.legend()
plt.show()
