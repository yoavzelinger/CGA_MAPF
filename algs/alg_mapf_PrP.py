from argparse import ArgumentParser
import csv

from algs.alg_functions_PrP import *
from algs.alg_sipps import run_sipps
from algs.alg_temporal_a_star import run_temporal_a_star
from algs.alg_sipps_functions import init_si_table, update_si_table_soft, update_si_table_hard
from run_single_MAPF_func import run_mapf_alg


def run_prp_sipps(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict,
) -> Tuple[Dict[str, List[Node]] | None, dict]:

    constr_type: str = params['constr_type']
    alg_name: bool = params['alg_name']
    to_render: bool = params['to_render']
    max_time: bool = params['max_time']

    start_time = time.time()

    # create agents
    agents = []
    for num, (s_node, g_node) in enumerate(zip(start_nodes, goal_nodes)):
        new_agent = AgentAlg(num, s_node, g_node)
        agents.append(new_agent)

    longest_len = 1
    # vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, longest_len)
    # vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, longest_len)
    ec_hard_np = init_ec_table(map_dim, longest_len)
    ec_soft_np = init_ec_table(map_dim, longest_len)

    r_iter = 0
    while time.time() - start_time < max_time:
        # preps
        if constr_type == 'hard':
            # vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, longest_len)
            ec_hard_np = init_ec_table(map_dim, longest_len)
        elif constr_type == 'soft':
            # vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, longest_len)
            ec_soft_np = init_ec_table(map_dim, longest_len)
        si_table: Dict[str, List[Tuple[int, int, str]]] = init_si_table(nodes)

        # calc paths
        h_priority_agents: List[AgentAlg] = []
        for agent in agents:
            new_path, alg_info = run_sipps(
                agent.start_node, agent.goal_node, nodes, nodes_dict, h_dict,
                None, ec_hard_np, None, None, ec_soft_np, None,
                agent=agent, si_table=si_table
            )

            if time.time() - start_time > max_time:
                return None, {}

            if new_path is None:
                agent.path = None
                break

            agent.path = new_path[:]
            h_priority_agents.append(agent)
            align_all_paths(h_priority_agents)

            if constr_type == 'hard':
                si_table = update_si_table_hard(new_path, si_table)
            elif constr_type == 'soft':
                si_table = update_si_table_soft(new_path, si_table)

            if longest_len < len(new_path):
                longest_len = len(new_path)
                # vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, longest_len)
                # vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, longest_len)
                ec_hard_np = init_ec_table(map_dim, longest_len)
                ec_soft_np = init_ec_table(map_dim, longest_len)
                if constr_type == 'hard':
                    for h_agent in h_priority_agents:
                        # update_constraints(h_agent.path, vc_hard_np, ec_hard_np, pc_hard_np)
                        update_ec_table(h_agent.path, ec_hard_np)
                elif constr_type == 'soft':
                    for h_agent in h_priority_agents:
                        # update_constraints(h_agent.path, vc_soft_np, ec_soft_np, pc_soft_np)
                        update_ec_table(h_agent.path, ec_soft_np)
            else:
                if constr_type == 'hard':
                    # update_constraints(new_path, vc_hard_np, ec_hard_np, pc_hard_np)
                    update_ec_table(new_path, ec_hard_np)
                elif constr_type == 'soft':
                    # update_constraints(new_path, vc_soft_np, ec_soft_np, pc_soft_np)
                    update_ec_table(new_path, ec_soft_np)


            # checks
            runtime = time.time() - start_time
            print(f'\r[{alg_name}] {r_iter=: <3} | agents: {len(h_priority_agents): <3} / {len(agents)} | {runtime= : .2f} s.', end='')  # , end=''
            # collisions: int = 0
            # for i in range(len(h_priority_agents[0].path)):
            #     check_vc_ec_neic_iter(h_priority_agents, i, False)
            # if collisions > 0:
            #     print(f'{collisions=} | {alg_info['c']=}')

        # return check
        if solution_is_found(agents):
            runtime = time.time() - start_time
            makespan: int = max([len(a.path) for a in agents])
            return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': makespan}

        # reshuffle
        r_iter += 1
        random.shuffle(agents)
        for agent in agents:
            agent.path = []

    return None, {}


def run_prp_a_star(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict,
) -> Tuple[Dict[str, List[Node]] | None, dict]:

    alg_name: bool = params['alg_name']
    to_render: bool = params['to_render']
    max_time: bool = params['max_time']

    start_time = time.time()

    # create agents
    agents = []
    for num, (s_node, g_node) in enumerate(zip(start_nodes, goal_nodes)):
        new_agent = AgentAlg(num, s_node, g_node)
        agents.append(new_agent)

    r_iter = 0
    while time.time() - start_time < max_time:
        # calc paths
        h_priority_agents: List[AgentAlg] = []
        longest_len = 1
        vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, longest_len)
        vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, longest_len)

        for agent in agents:
            new_path, alg_info = run_temporal_a_star(
                agent.start_node, agent.get_goal_node(), nodes, nodes_dict, h_dict,
                vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np,
                agent=agent,
            )

            if time.time() - start_time > max_time:
                return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': r_iter, "finished": False}

            if new_path is None:
                agent.path = None
                break
            agent.path = new_path[:]
            h_priority_agents.append(agent)
            align_all_paths(h_priority_agents)
            if longest_len < len(new_path):
                longest_len = len(new_path)
                vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, longest_len)
                vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, longest_len)
                for h_agent in h_priority_agents:
                    update_constraints(h_agent.path, vc_hard_np, ec_hard_np, pc_hard_np)
            else:
                update_constraints(new_path, vc_hard_np, ec_hard_np, pc_hard_np)

            # checks
            runtime = time.time() - start_time
            # print(f'\r[{alg_name}] {r_iter=: <3} | agents: {len(h_priority_agents): <3} / {len(agents)} | {runtime= : .2f} s.')  # , end=''
            # collisions: int = 0
            # for i in range(len(h_priority_agents[0].path)):
            #     to_count = False if constr_type == 'hard' else True
            #     # collisions += check_vc_ec_neic_iter(h_priority_agents, i, to_count)
            # if collisions > 0:
            #     print(f'{collisions=} | {alg_info['c']=}')

        # return check
        if solution_is_found(agents):
            runtime = time.time() - start_time
            makespan: int = max([len(a.path) for a in agents])
            return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': makespan, "finished": True}

        # reshuffle
        r_iter += 1
        random.shuffle(agents)
        # agents = get_shuffled_agents(agents) # - no meaning
        for agent in agents:
            agent.path = []

    return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': r_iter, "finished": False}


def run_k_prp(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        params: Dict,
) -> Tuple[Dict[str, List[Node]] | None, dict]:
    """
    -> MAPF:
    - stop condition: all agents at their locations or time is up
    - behaviour, when agent is at its goal: the goal remains the same
    - output: success, time, makespan, soc
    LMAPF:
    - stop condition: the end of n iterations where every iteration has a time limit
    - behaviour, when agent is at its goal: agent receives a new goal
    - output: throughput
    """
    # constr_type: str = params['constr_type']
    k_limit: int = params['k_limit']
    alg_name: bool = params['alg_name']
    pf_alg = params['pf_alg']
    pf_alg_name = params['pf_alg_name']
    to_render: bool = params['to_render']
    max_time: bool = params['max_time']
    img_np: np.ndarray = params['img_np']

    if to_render:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    start_time = time.time()

    # create agents
    agents: List[AgentAlg] = []
    for num, (s_node, g_node) in enumerate(zip(start_nodes, goal_nodes)):
        new_agent = AgentAlg(num, s_node, g_node)
        agents.append(new_agent)



    # main loop
    k_iter = 0
    while time.time() - start_time < max_time:

        si_table: Dict[str, List[Tuple[int, int, str]]] = init_si_table(nodes)
        vc_soft_np, ec_soft_np, pc_soft_np = init_constraints(map_dim, k_limit + 1)
        vc_hard_np, ec_hard_np, pc_hard_np = init_constraints(map_dim, k_limit + 1)

        # calc k paths
        all_good: bool = True
        h_priority_agents: List[AgentAlg] = []

        for agent in agents:
            new_path, alg_info = pf_alg(
                agent.curr_node, agent.goal_node, nodes, nodes_dict, h_dict,
                vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np,
                flag_k_limit=True, k_limit=k_limit, agent=agent, si_table=si_table
            )
            if new_path is None:
                all_good = False
                break
            new_path = align_path(new_path, k_limit + 1)
            agent.k_path = new_path[:]
            # checks
            # for i in range(k_limit + 1):
            #     other_paths = {a.name: a.k_path for a in h_priority_agents if a != agent}
            #     check_one_vc_ec_neic_iter(agent.k_path, agent.name, other_paths, i)
            h_priority_agents.append(agent)


            if pf_alg_name == 'sipps':
                update_constraints(new_path, vc_hard_np, ec_hard_np, pc_hard_np)
                si_table = update_si_table_hard(new_path, si_table, consider_pc=False)
            elif pf_alg_name == 'a_star':
                update_constraints(new_path, vc_hard_np, ec_hard_np, pc_hard_np)
            else:
                raise RuntimeError('nono')

        # reset k paths if not good
        if not all_good:
            for agent in agents:
                agent.k_path = []

        # ------------------------------ #
        # ------------------------------ #
        # ------------------------------ #
        if all_good and to_render:
            for i in range(k_limit):
                for a in agents:
                    a.curr_node = a.k_path[i]
                # plot the iteration
                i_agent = agents[0]
                plot_info = {
                    'img_np': img_np,
                    'agents': agents,
                    'i_agent': i_agent,
                }
                plot_step_in_env(ax[0], plot_info)
                plt.pause(0.001)
                # plt.pause(1)

        # append paths
        for agent in agents:
            agent.path.extend(agent.k_path[1:])
            if len(agent.path) > 0:
                agent.curr_node = agent.path[-1]

        # print
        runtime = time.time() - start_time
        finished: List[AgentAlg] = [a for a in agents if len(a.path) > 0 and a.path[-1] == a.get_goal_node()]
        print(f'\r[{alg_name}] {k_iter=: <3} | agents: {len(finished): <3} / {len(agents)} | {runtime=: .2f} s.')  # , end=''

        # return check
        if solution_is_found(agents):
            runtime = time.time() - start_time
            makespan: int = max([len(a.path) for a in agents])
            return {a.name: a.path for a in agents}, {'agents': agents, 'time': runtime, 'makespan': makespan}

        # reshuffle
        k_iter += 1
        # random.shuffle(agents)
        agents = get_shuffled_agents(agents)
        for agent in agents:
            agent.k_path = []

    return None, {}

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

with open(f"Output_files/Output_{map_name}_PRP.csv", mode="w", newline="",
          encoding="utf-8") as file:
    columns = ["map_name", "scenario_index", "total_agents", "active_agents", "inactive_agents", "density", "SOC", "makespan", "runtime"]
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

def addRecordToCsv(current_values):

    record = list(current_values)

    with open(f"Output_files/Output_{map_name}_PRP.csv", mode="a", newline="", encoding="utf-8") as file:
        writerRecord = csv.writer(file)
        writerRecord.writerow(record)

@use_profiler(save_dir='./stats/alg_prp.pstat')
def main():

    to_render = False
    # to_render = False

    params = {
        'max_time': 30,
        'alg_name': 'PRP',
        'alt_goal_flag': 'first',
        # 'alt_goal_flag': 'num', 'alt_goal_num': 3,
        # 'alt_goal_flag': 'all',
        'to_render': to_render,
        'final_render': to_render,
        'k_limit': 5,
        'constr_type': 'hard',
        'pf_alg_name': 'sipps',
        'pf_alg': run_sipps,
    }
    if args.plot:
        scenario_index, active_agents, inactive_agents = tuple(map(int, args.plot.split()))
        run_mapf_alg(alg=run_prp_a_star, params=params, final_render=True, map_name=map_name, active_agents=active_agents, inactive_agents=inactive_agents, scenario_index=scenario_index)
        return

    for active_agents in active_agents_amounts:
        for inactive_agents in inactive_agents_amounts:
            print(f'\nRunning with {active_agents=} and {inactive_agents=}')
            for scenario_index in range(1, 26):
                current_values = run_mapf_alg(alg=run_prp_a_star, params=params, final_render=False, map_name=map_name, active_agents=active_agents, inactive_agents=inactive_agents, scenario_index=scenario_index)
                addRecordToCsv(current_values)


if __name__ == '__main__':
    main()







