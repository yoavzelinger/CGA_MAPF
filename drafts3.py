import numpy as np

from functions_plotting import *

# file_dir = 'logs_for_experiments/MAPF_2025-04-09--12-10_ALGS-2_RUNS-30_MAP-random-32-32-20.json'
# file_dir = 'logs_for_experiments/MAPF_2025-04-10--07-59_ALGS-2_RUNS-30_MAP-room-32-32-4.json'
# file_dir = 'logs_for_experiments/MAPF_2025-04-10--08-12_ALGS-2_RUNS-30_MAP-maze-32-32-2.json'
file_dir = 'logs_for_experiments/MAPF_2025-04-10--08-36_ALGS-2_RUNS-30_MAP-maze-32-32-4.json'

with open(f'{file_dir}', 'r') as openfile:
    # Reading from json file
    logs_dict = json.load(openfile)
    expr_type = logs_dict['expr_type']

    # print(logs_dict)

    for alg in ['MACGA+PIBT', 'LaCAM*']:
        # for n in ['200', '300']:
        # for n in ['350', '400']:
        for n in ['250', '300']:
            for d_type in ['makespan', 'time']:
                print(f'{alg} {n} agents {d_type}: {np.average(logs_dict[alg][n][d_type]): .2f} (+- {np.std(logs_dict[alg][n][d_type]): .2f})')
