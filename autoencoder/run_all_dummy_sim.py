
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_load.ld_data_load as lddl
import simulator.ld_simulator as lds
ld_data = lddl.get_data(True, 100000)


# BASE_FOLDER_RUNNING = "../../../../../data/expensive_experiments/simulator_run/"
# BASE_FOLDER_SIMULATOR = "../../../../../reference/expensive_experiments/simulator/NoCsim_coreMapped_16thMay2017/"

# BASE_FOLDER_RUNNING = "../simulator_run/"
# BASE_FOLDER_SIMULATOR = "../NoCsim_coreMapped_16thMay2017/"

# run_all_benchmarks(BASE_FOLDER_SIMULATOR, BASE_FOLDER_RUNNING)

benchmark_list = ['dedup', 'canneal', 'lu', 'fft','radix', 'water', 'vips', 'fluid']
for benchmark in benchmark_list:
	SIM_BASE_FOLDER = "../../data/link_distribution/NoCsim_coreMapped_16thMay2017/"
	SIM_BASE_FOLDER += benchmark + "/"
	# sim = Simulator(SIM_BASE_FOLDER, "canneal")
	sim = lds.Dummy_Simulator(SIM_BASE_FOLDER)
	print(benchmark, -1*sim.run_mesh())

SIM_BASE_FOLDER = "../../data/link_distribution/NoCsim_coreMapped_16thMay2017/"
SIM_BASE_FOLDER += "canneal/"
# sim = Simulator(SIM_BASE_FOLDER, "canneal")
sim = lds.Dummy_Simulator(SIM_BASE_FOLDER)
max_value = 0
max_idx = 0
for i in range(6000, 100000):
	sim_value = sim.run(ld_data[i])
	if(sim_value > max_value):
		max_value = sim_value
		max_idx = i
		print(max_idx, max_value)

INPUT_BENCHMARK = SIM_BASE_FOLDER+"input_benchmark.txt"
# print get_communication_frequency(INPUT_BENCHMARK)

# generate_sw_connections(node_connection_list, SW_CONNECTION_1)
# feature_vector = generate_feature_list(connection_idx_list)
# print(dummy_simulator(INPUT_BENCHMARK, feature_vector))

# SIM_BASE_FOLDER = "../../../../../reference/expensive_experiments/simulator/NoCsim_coreMapped_16thMay2017/canneal/"

# INPUT_BENCHMARK = SIM_BASE_FOLDER+"input_benchmark.txt"

# SW_CONNECTION_1 = SIM_BASE_FOLDER+"sw_connection_1.txt"
# SW_CONNECTION = SIM_BASE_FOLDER+"sw_connection.txt"

# node_connection_list = get_node_connection_list(SW_CONNECTION)

# (100000, 480)
# ('dedup', -3.4695760224618248)
# ('canneal', -7.335418487746181)
# ('lu', -1.7346678104742481)
# ('fft', -5.7308587164255425)
# ('radix', -14.668833496158918)
# ('water', -7.335418487746181)
# ('vips', -1.2781163404717246)
# ('fluid', -1.5873140173649869)
# (6000, 16.581923349457313)
# (6004, 17.412739874810669)
# (6012, 17.600501636774347)
# (6066, 18.071702646747205)
# (33205, 18.157691483110529)
# (58741, 18.171799185373469)
