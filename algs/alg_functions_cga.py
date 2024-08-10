import heapq
import random

from globals import *
from functions_general import *
from functions_plotting import *


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# CLASSES
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# FUNCS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def stay(agent: AgentAlg, at_node: Node, config_to: Dict[str, Node], occupied_to: Dict[str, AgentAlg]) -> None:
    config_to[agent.name] = at_node
    occupied_to[at_node.xy_name] = agent


def get_min_h_nei_node(curr_node: Node, goal_node: Node, h_dict: Dict[str, np.ndarray]) -> Node:
    nei_nodes = curr_node.neighbours_nodes[:]
    goal_h_np: np.ndarray = h_dict[goal_node.xy_name]
    min_h_nei_node = min(nei_nodes, key=lambda n: int(goal_h_np[n.x, n.y]))
    return min_h_nei_node


def build_corridor_from_nodes(
        curr_node: Node,
        goal_node: Node,
        h_dict: Dict[str, np.ndarray],
        non_sv_nodes_np: np.ndarray
) -> List[Node]:
    main_next_node = get_min_h_nei_node(curr_node, goal_node, h_dict)
    corridor: List[Node] = [curr_node, main_next_node]
    while non_sv_nodes_np[main_next_node.x, main_next_node.y] == 0 and main_next_node != goal_node:
        main_next_node = get_min_h_nei_node(main_next_node, goal_node, h_dict)
        corridor.append(main_next_node)
    return corridor


def unfold_path(next_node: Node, son_to_father_dict: Dict[str, Node | None]) -> List[Node]:
    path = [next_node]
    father = son_to_father_dict[next_node.xy_name]
    while father is not None:
        path.append(father)
        father = son_to_father_dict[father.xy_name]
    path.reverse()
    return path


def find_ev_path(
        node: Node,
        corridor: List[Node],
        blocked_nodes: List[Node],
        i_goal_node: Node,
        edge_blocked_nodes: List[Node],
        captured_free_nodes: List[Node],
        occupied_from: Dict[str, AgentAlg],
        config_to: Dict[str, Node],
        occupied_to: Dict[str, AgentAlg],
) -> Tuple[List[Node], Node, bool] | Tuple[None, None, bool]:
    open_list: Deque[Node] = deque([node])
    closed_names_list_heap = []

    son_to_father_dict: Dict[str, Node | None] = {node.xy_name: None}
    blocked_is_involved: bool = False
    iteration: int = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.popleft()
        if next_node not in corridor and next_node not in captured_free_nodes and next_node.xy_name not in occupied_from:
            ev_path = unfold_path(next_node, son_to_father_dict)
            return ev_path, next_node, blocked_is_involved

        for nei_node in next_node.neighbours_nodes:
            # self ref
            if nei_node == next_node:
                continue
            if nei_node.xy_name in closed_names_list_heap:
                continue
            if nei_node == i_goal_node and nei_node in edge_blocked_nodes:
                continue
            if nei_node in blocked_nodes:
                blocked_is_involved = True
                continue
            if nei_node.xy_name in occupied_to:
                blocked_is_involved = True
                continue
            open_list.append(nei_node)
            son_to_father_dict[nei_node.xy_name] = next_node
        heapq.heappush(closed_names_list_heap, next_node.xy_name)

    return None, None, blocked_is_involved


def get_alt_goal_node(
        node: Node,
        non_sv_nodes_np: np.ndarray,
        agents: List[AgentAlg],
) -> Node:
    """
    Switch to alt goal (non-SV node that is also not a goal of anyone)
    """
    other_goals: List[Node] = [agent.get_goal_node() for agent in agents]
    open_list: Deque[Node] = deque([node])
    closed_names_list_heap = []
    iteration: int = 0
    while len(open_list) > 0:
        iteration += 1
        next_node = open_list.popleft()
        if non_sv_nodes_np[next_node.x, next_node.y] and next_node not in other_goals and random.random() > 0.5:
            return next_node
        for nei_node in next_node.neighbours_nodes:
            # self ref
            if nei_node == next_node:
                continue
            if nei_node.xy_name in closed_names_list_heap:
                continue
            open_list.append(nei_node)
        heapq.heappush(closed_names_list_heap, next_node.xy_name)

    raise RuntimeError('nope')


def update_last_visit_dict(last_visit_dict: Dict[str, int], given_agents: List[AgentAlg]) -> None:
    for m_agent in given_agents:
        for i_n, n in enumerate(m_agent.path):
            last_visit_dict[n.xy_name] = max(i_n, last_visit_dict[n.xy_name])


def push_ev_agents(
        ev_path: List[Node],
        config_from: Dict[str, Node],
        occupied_from: Dict[str, AgentAlg],
        moved_agents: List[AgentAlg],
        nodes: List[Node],
        main_agent: AgentAlg,
        last_visit_dict,
        iteration: int
) -> Tuple[int, List[AgentAlg]]:

    curr_moved_agents = moved_agents[:]

    ev_chain_dict: Dict[str, Node | None] = {}
    max_i = len(ev_path)
    for i, n in enumerate(ev_path):
        if i + 1 == max_i:
            ev_chain_dict[n.xy_name] = None
            break
        ev_chain_dict[n.xy_name] = ev_path[i + 1]
    assert len(ev_chain_dict) == max_i

    agents_to_assign: List[AgentAlg] = []
    locations_to_assign: List[Node] = []
    for n in ev_path:
        if n.xy_name in occupied_from:
            i_agent: AgentAlg = occupied_from[n.xy_name]
            assert i_agent != main_agent
            agents_to_assign.append(i_agent)
            locations_to_assign.append(n)
    locations_to_assign = locations_to_assign[1:]
    locations_to_assign.append(ev_path[-1])
    agents_to_assign.reverse()
    locations_to_assign.reverse()

    for a, final_n in zip(agents_to_assign, locations_to_assign):
        a_name = a.name
        new_path: List[Node] = []
        curr_node: Node = config_from[a.name]
        while curr_node != final_n:
            next_node: Node = ev_chain_dict[curr_node.xy_name]
            next_n_last_visit = last_visit_dict[next_node.xy_name]
            while len(a.path) + len(new_path) <= next_n_last_visit:
                new_path.append(curr_node)
            assert next_node.xy_name in curr_node.neighbours
            new_path.append(next_node)
            curr_node = next_node
        # assert a.path[-1].xy_name in new_path[0].neighbours
        a.path.extend(new_path)
        curr_moved_agents.append(a)
        update_last_visit_dict(last_visit_dict, [a])
    max_len = max([len(a.path) for a in agents_to_assign])
    return max_len, agents_to_assign


def get_last_visit_dict(given_list: List[Node], given_agents: List[AgentAlg], iteration: int) -> Dict[str, int]:
    last_visit_dict = {n.xy_name: 0 for n in given_list}
    for m_agent in given_agents:
        for i_n, n in enumerate(m_agent.path[iteration:]):
            if n in given_list:
                last_visit_dict[n.xy_name] = max(i_n + iteration, last_visit_dict[n.xy_name])
    return last_visit_dict


def push_main_agent(main_agent: AgentAlg, corridor: List[Node], moved_agents: List[AgentAlg], iteration: int) -> None:
    assert main_agent not in moved_agents
    assert len(main_agent.path) - 1 == iteration
    last_visit_dict = get_last_visit_dict(corridor, moved_agents, iteration)
    assert corridor[0] == main_agent.path[-1]
    prev_n = corridor[0]
    for c_n in corridor[1:]:
        c_n_last_visit = last_visit_dict[c_n.xy_name]
        while len(main_agent.path) <= c_n_last_visit:
            main_agent.path.append(prev_n)
        assert c_n.xy_name in prev_n.neighbours
        main_agent.path.append(c_n)
        prev_n = c_n


def calc_cga_step(
        agent_i: AgentAlg,
        iteration: int,
        config_from: Dict[str, Node],
        occupied_from: Dict[str, AgentAlg],
        config_to: Dict[str, Node],
        occupied_to: Dict[str, AgentAlg],
        agents: List[AgentAlg],
        agents_dict: Dict[str, AgentAlg],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        non_sv_nodes_np: np.ndarray,
        blocked_nodes: List[Node],
) -> None:
    """
    v - Build corridor
    v - Find EV paths (with the goal node blocked around except the corridor edge)
    v - If there are no EV:
    v         - Because it is impossible:
    v             - Switch to alt goal (non-SV node that is also not a goal of anyone)
    v     - Stay
    v     - Continue
    - Evacuate ev-agents
    - Build the steps in the corridor to the main agent
    """
    # Build corridor
    corridor: List[Node] = build_corridor_from_nodes(
        config_from[agent_i.name], agent_i.get_goal_node(), h_dict, non_sv_nodes_np
    )

    # Find ev-agents
    ev_agents: List[AgentAlg] = [occupied_from[node.xy_name] for node in corridor[1:] if node.xy_name in occupied_from]

    # Find EV paths (with the goal node blocked around except the corridor edge)
    ev_paths_list: List[List[Node]] = []
    captured_free_nodes: List[Node] = []
    i_goal_node: Node = agent_i.get_goal_node()
    curr_blocked_nodes = blocked_nodes[:]
    curr_blocked_nodes.append(config_from[agent_i.name])

    # get edge_blocked_nodes
    edge_blocked_nodes = []
    if i_goal_node == corridor[-1]:
        # edge_blocked_nodes.append(i_goal_node)
        for n in i_goal_node.neighbours_nodes:
            if n not in corridor:
                heapq.heappush(edge_blocked_nodes, n)

    for ev_agent in ev_agents:
        ev_path, captured_free_node, blocked_is_involved = find_ev_path(
            config_from[ev_agent.name], corridor,
            curr_blocked_nodes, i_goal_node, edge_blocked_nodes,
            captured_free_nodes,
            occupied_from, config_to, occupied_to
        )
        if ev_path is None:
            if not blocked_is_involved:
                agent_i.alt_goal_node = get_alt_goal_node(
                    config_from[agent_i.name], non_sv_nodes_np, agents,
                )
            stay(agent_i, config_from[agent_i.name], config_to, occupied_to)
            return

        captured_free_nodes.append(captured_free_node)
        ev_paths_list.append(ev_path)

    # Evacuate ev-agents
    moved_agents = []
    last_visit_dict = {n.xy_name: 0 for n in nodes}
    for i_ev_path, ev_path in enumerate(ev_paths_list):
        max_len, assigned_agents = push_ev_agents(ev_path, config_from, occupied_from,
                                                  moved_agents, nodes, agent_i, last_visit_dict,
                                                  iteration)
        assert agent_i not in assigned_agents
        # extend_other_paths(max_len, self.main_agent, self.agents)
        moved_agents.extend(assigned_agents)
        moved_agents = list(set(moved_agents))

    # Build the steps in the corridor to the main agent + extend the path
    push_main_agent(agent_i, corridor, moved_agents, iteration)
    moved_agents.append(agent_i)

    # update config_to, occupied_to, blocked_nodes
    for agent in moved_agents:
        next_node = agent.path[iteration + 1]
        config_to[agent.name] = next_node
        occupied_to[next_node.xy_name] = agent
        for n in agent.path[iteration + 1:]:
            if n not in blocked_nodes:
                heapq.heappush(blocked_nodes, n)
    return








