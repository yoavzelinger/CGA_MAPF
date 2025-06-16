from argparse import ArgumentParser
import csv

from algs.alg_functions_pibt import *
from run_single_MAPF_func import run_mapf_alg


def run_pibt(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict
) -> Tuple[None, Dict] | Tuple[Dict[str, List[Node]], Dict]:

    max_time: int | float = params['max_time']

    start_time = time.time()

    # create agents
    agents, agents_dict = create_agents(start_nodes, goal_nodes)
    n_agents = len(agents_dict)
    agents.sort(key=lambda a: a.priority, reverse=True)

    iteration = 0
    finished = False
    while not finished:

        config_from: Dict[str, Node] = {a.name: a.path[-1] for a in agents}
        occupied_from: Dict[str, AgentAlg] = {a.path[-1].xy_name: a for a in agents}
        config_to: Dict[str, Node] = {}
        occupied_to: Dict[str, AgentAlg] = {}


        # calc the step
        for agent in agents:
            if agent.name not in config_to:
                _ = run_procedure_pibt(
                    agent,
                    config_from, occupied_from,
                    config_to, occupied_to,
                    agents_dict, nodes_dict, h_dict, [])

        # execute the step + check the termination condition
        finished = True
        agents_finished = []
        for agent in agents:
            next_node = config_to[agent.name]
            agent.path.append(next_node)
            agent.prev_node = agent.curr_node
            agent.curr_node = next_node
            if agent.curr_node != agent.get_goal_node():
                finished = False
                agent.priority += 1
            else:
                agent.priority = agent.init_priority
                agents_finished.append(agent)

        # unfinished first
        agents.sort(key=lambda a: a.priority, reverse=True)

        # print + render
        runtime = time.time() - start_time
        print(f'\r{"*" * 10} | [PIBT] {iteration=: <3} | finished: {len(agents_finished)}/{n_agents: <3} | runtime: {runtime: .2f} seconds | {"*" * 10}', end='')
        iteration += 1

        if runtime > max_time:
            return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': iteration, "finished": False}

    # checks
    for i in range(len(agents[0].path)):
        check_vc_ec_neic_iter(agents, i, to_count=False)
    runtime = time.time() - start_time
    if runtime > max_time:
            return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': iteration, "finished": False}
    return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': iteration, "finished": True}

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-e", "--environment", type=str, help="The environment - map name", required=True)
parser.add_argument("-s", "--scenario_index", type=int, help="Scenario index", default=-1)
parser.add_argument("-p", "--plot", type=str, help="Plot the results of <scenario_index active agents inactive_agents>", default="")
args = parser.parse_args()

map_name = args.environment
active_agents_amounts = [25, 50, 75, 100, 125, 150]
inactive_agents_amounts = [0, 25, 50, 75, 100, 125, 150]

if not os.path.exists("Output_files"):
    os.makedirs("Output_files")

with open(f"Output_files/Output_{map_name}_PIBT.csv", mode="w", newline="",
          encoding="utf-8") as file:
    columns = ["map_name", "scenario_index", "total_agents", "active_agents", "inactive_agents", "density", "SOC", "makespan", "runtime"]
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

def addRecordToCsv(current_values):

    record = list(current_values)

    with open(f"Output_files/Output_{map_name}_PIBT.csv", mode="a", newline="", encoding="utf-8") as file:
        writerRecord = csv.writer(file)
        writerRecord.writerow(record)

@use_profiler(save_dir='./stats/alg_pibt.pstat')
def main():

    to_render = False
    # to_render = False

    params = {
        'max_time': 30,
        'alg_name': 'PIBT',
        'alt_goal_flag': 'first',
        # 'alt_goal_flag': 'num', 'alt_goal_num': 3,
        # 'alt_goal_flag': 'all',
        'to_render': to_render,
    }
    if args.plot:
        scenario_index, active_agents, inactive_agents = tuple(map(int, args.plot.split()))
        run_mapf_alg(alg=run_pibt, params=params, final_render=True, map_name=map_name, active_agents=active_agents, inactive_agents=inactive_agents, scenario_index=scenario_index)
        return

    for active_agents in active_agents_amounts:
        for inactive_agents in inactive_agents_amounts:
            print(f'\nRunning with {active_agents=} and {inactive_agents=}')
            for scenario_index in range(1, 26):
                current_values = run_mapf_alg(alg=run_pibt, params=params, final_render=False, map_name=map_name, active_agents=active_agents, inactive_agents=inactive_agents, scenario_index=scenario_index)
                addRecordToCsv(current_values)

if __name__ == '__main__':
    main()

