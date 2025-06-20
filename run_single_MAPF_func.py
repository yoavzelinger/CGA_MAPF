from globals import *
from functions_general import *
from functions_plotting import *
import pandas as pd

SCENARIOS_FOLDER_PATH = 'scenarios'
ACTIVE_INACTIVE_MARGIN = 150

def get_start_goal_nodes(node_dict: Dict[str, Node], map_name: str, active_agents: int, inactive_agents: int, scenario_index: int) -> Tuple[List[Node], List[Node]]:
    scenario_file = f"{map_name}__scenario_{scenario_index}.csv"
    scenario_path = os.path.join(SCENARIOS_FOLDER_PATH, scenario_file)
    assert os.path.exists(scenario_path), f"Scenario file {scenario_file} does not exist in {SCENARIOS_FOLDER_PATH}."

    active_agents_df = pd.read_csv(scenario_path, nrows=active_agents)
    inactive_agents_df = pd.read_csv(scenario_path, nrows=inactive_agents, skiprows=ACTIVE_INACTIVE_MARGIN)

    # iterate through the rows and create start and goal nodes
    start_nodes = []
    goal_nodes = []
    for agent_index, (start_y, start_x, goal_y, goal_x) in active_agents_df.iterrows():
        start_node = node_dict[f"{start_x}_{start_y}"]
        goal_node = node_dict[f"{goal_x}_{goal_y}"]
        start_nodes.append(start_node)
        goal_nodes.append(goal_node)
    for agent_index, (start_y, start_x, goal_y, goal_x) in inactive_agents_df.iterrows():
        start_node = node_dict[f"{start_x}_{start_y}"]
        start_nodes.append(start_node)
        goal_nodes.append(None)

    return start_nodes, goal_nodes

def run_mapf_alg(alg, params, final_render: bool, map_name: str, active_agents: int, inactive_agents: int, scenario_index: int):
    set_seed(random_seed_bool=False)
    img_dir = f'{map_name}.map'

    path_to_maps: str = './maps'
    path_to_heuristics: str = './logs_for_heuristics'
    path_to_sv_maps: str = './logs_for_freedom_maps'

    img_np, (height, width) = get_np_from_dot_map(img_dir, path_to_maps)
    map_dim = (height, width)
    nodes, nodes_dict = build_graph_from_np(img_np, show_map=False)
    density = round((((active_agents + inactive_agents) + (height * width - len(nodes))) / (height * width)), 3)
    h_dict: Dict[str, np.ndarray] = exctract_h_dict(img_dir, path_to_heuristics)
    blocked_sv_map: np.ndarray = get_blocked_sv_map(img_dir, folder_dir=path_to_sv_maps)
    # sv_map: np.ndarray = get_sv_map(img_dir, folder_dir=path_to_sv_maps)

    start_nodes, goal_nodes = get_start_goal_nodes(nodes_dict, map_name, active_agents, inactive_agents, scenario_index)

    params['img_np'] = img_np
    # params['sv_map'] = sv_map
    params['blocked_sv_map'] = blocked_sv_map
    paths_dict, info = alg(
        start_nodes, goal_nodes, nodes, nodes_dict, h_dict, map_dim, params
    )

    finished = info['finished']
    agents: List = info['agents']
    SOC = sum([len(a.path) for a in agents])
    makespan = info['makespan']
    runtime = info['time']
    
    # plot
    if final_render:
        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_rate = 0.001
        # plot_rate = 0.5
        # plot_rate = 1
        max_path_len = max([len(a.path) for a in agents])

        print(f'\n{max_path_len=}')
        print(f'{SOC=}')
        for i in range(max_path_len):
            # update curr nodes
            for a in agents:
                a.update_curr_node(i)
            # plot the iteration
            i_agent = agents[0]
            plot_info = {
                'img_np': img_np,
                'agents': agents,
                'iteration': i,
            }
            plot_step_in_env(ax[0], plot_info)
            plt.pause(plot_rate)
        plt.show()

    if not finished:
        SOC, makespan, runtime = None, None, None
    print(map_name, scenario_index, (active_agents + inactive_agents), active_agents, inactive_agents, density, SOC, makespan, runtime)
    return map_name, scenario_index, (active_agents + inactive_agents), active_agents, inactive_agents, density, SOC, makespan, runtime