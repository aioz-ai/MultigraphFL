import json
import os
import argparse
import networkx as nx

from utils.utils import get_connectivity_graph, add_upload_download_delays, get_delta_mbst_overlay,\
    get_star_overlay, get_ring_overlay, get_matcha_cycle_time, generate_multigraph, compute_cycle_time_multigraph

# Model size in bit
MODEL_SIZE_DICT = {"femnist": 4843243,
                   "sent140": 19269416,
                   "inaturalist": 44961717}

# Model computation time in ms
COMPUTATION_TIME_DICT = { "femnist": 4.6,
                         "sent140": 9.8,
                         "inaturalist": 25.4}

ROUNDS_DICT = {"femnist": 6400,
               "sent140": 20000,
               "inaturalist": 1600}

parser = argparse.ArgumentParser()

parser.add_argument('name',
                    help='name of the network to use;')
parser.add_argument("--experiment",
                    type=str,
                    help="name of the experiment that will be run on the network;"
                         "possible are femnist, inaturalist, sent140;"
                         "if not precised --model_size will be used as model size;",
                    default=None)
parser.add_argument('--model_size',
                    type=float,
                    help="size of the model that will be transmitted on the network in bit;"
                         "ignored if --experiment is precised;",
                    default=1e8)
parser.add_argument("--local_steps",
                    type=int,
                    help="number of local steps, used to get computation time",
                    default=1)
parser.add_argument("--upload_capacity",
                    type=float,
                    help="upload capacity at edge in bit/s; default=1e32",
                    default=1e32)
parser.add_argument("--download_capacity",
                    type=float,
                    help="download capacity at edge in bit/s; default=1e32",
                    default=1e32)
parser.add_argument("--communication_budget",
                    type=float,
                    help="communication budget to use with matcha; will be ignored if name is not matcha",
                    default=0.5)
parser.add_argument("--default_capacity",
                    type=float,
                    help="default capacity (in bit/s) to use on links with unknown capacity",
                    default=1e9)
parser.add_argument('--centrality',
                    help="centrality type; default: load;",
                    default="load")
parser.add_argument("--arch",
                    type=str,
                    help="name of the architecture;",
                    default=None)
parser.add_argument("--t_max",
                    type=int,
                    help="number of t_max",
                    default=5)

parser.set_defaults(user=False)

args = parser.parse_args()
args.default_capacity *= 1e-3


if __name__ == "__main__":
    if args.experiment is not None:
        args.model_size = MODEL_SIZE_DICT[args.experiment]
        args.computation_time = args.local_steps * COMPUTATION_TIME_DICT[args.experiment]

    upload_delay = (args.model_size / args.upload_capacity) * 1e3
    download_delay = (args.model_size / args.download_capacity) * 1e3

    result_dir = "./results/{}/{}".format(args.name, args.experiment)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    cycle_time_dict = {}
    cycle_time_json = os.path.join(result_dir, "cycle_time")
    if not os.path.exists(cycle_time_json):
        os.makedirs(cycle_time_json)
    print("*"*50)
    print(">>>>>>>>>>>> Multigraph: {} dataset - {} network".format(args.experiment, args.name))
    print("Upload capacity: ", int(args.upload_capacity))
    cycle_time_json = os.path.join(cycle_time_json, "{}_multigraph_{}.json".format(args.arch, str(int(args.upload_capacity)).zfill(11)))
    # results_txt_path = os.path.join(result_dir, "cycle_time.txt")
    # results_file = open(results_txt_path, "w")

    path_to_graph = "./data/{}.gml".format(args.name)

    underlay = nx.read_gml(path_to_graph)

    print("Number of Workers: {}".format(underlay.number_of_nodes()))
    print("Number of links: {}".format(underlay.number_of_edges()))
    print('Number of rounds: {}'.format(ROUNDS_DICT[args.experiment]))

    nx.set_node_attributes(underlay, upload_delay, 'uploadDelay')
    nx.set_node_attributes(underlay, download_delay, "downloadDelay")

    # nx.write_gml(underlay.copy(), os.path.join(result_dir, "original.gml"))

    connectivity_graph = get_connectivity_graph(underlay, args.default_capacity)

    # Ring
    if args.arch == "ring":
        ring = get_ring_overlay(connectivity_graph.copy(), args.computation_time, args.model_size)

        ring_multigraph = generate_multigraph(ring.copy(), args.computation_time, args.model_size, t_max=args.t_max)
        list_delay_time_multigraph, cycle_time_multigraph = compute_cycle_time_multigraph(ring_multigraph, args.computation_time, args.model_size, round=ROUNDS_DICT[args.experiment])
        print("Cycle time MULTIGRAPH for RING architecture: {0:.1f} ms".format(cycle_time_multigraph))
        result_dir_ring = os.path.join(result_dir, "ring")
        if not os.path.exists(result_dir_ring):
            os.makedirs(result_dir_ring)
        json_path = os.path.join(result_dir_ring, "delay_time_list.json")
        with open(json_path, "w") as f:
            json.dump(list_delay_time_multigraph, f)
        for i in range(len(ring_multigraph)):
            nx.write_gml(ring_multigraph[i], os.path.join(result_dir_ring, "ring_{}.gml".format(i)))

        cycle_time_dict['ring_multigraph'] = [cycle_time_multigraph]
    elif args.arch == "all":
        ring = get_ring_overlay(connectivity_graph.copy(), args.computation_time, args.model_size)

        ring_multigraph = generate_multigraph(ring.copy(), args.computation_time, args.model_size, t_max=args.t_max)
        list_delay_time_multigraph, cycle_time_multigraph = compute_cycle_time_multigraph(ring_multigraph,
                                                                                          args.computation_time,
                                                                                          args.model_size, round=ROUNDS_DICT[args.experiment])
        print("Cycle time MULTIGRAPH for RING architecture: {0:.1f} ms".format(cycle_time_multigraph))
        result_dir_ring = os.path.join(result_dir, "ring")
        if not os.path.exists(result_dir_ring):
            os.makedirs(result_dir_ring)
        json_path = os.path.join(result_dir_ring, "delay_time_list.json")
        with open(json_path, "w") as f:
            json.dump(list_delay_time_multigraph, f)
        for i in range(len(ring_multigraph)):
            nx.write_gml(ring_multigraph[i], os.path.join(result_dir_ring, "ring_{}.gml".format(i)))

        cycle_time_dict['ring_multigraph'] = [cycle_time_multigraph]

        # MST
        for u, v, data in connectivity_graph.edges(data=True):
            weight = args.computation_time + data["latency"] + args.model_size / data["availableBandwidth"]
            connectivity_graph.add_edge(u, v, weight=weight)

        MST = nx.minimum_spanning_tree(connectivity_graph.copy(), weight="weight")

        MST = MST.to_directed()

        mst_multigraph = generate_multigraph(MST.copy(), args.computation_time, args.model_size, t_max=args.t_max)
        list_delay_time_multigraph, cycle_time_multigraph = compute_cycle_time_multigraph(mst_multigraph,
                                                                                          args.computation_time,
                                                                                          args.model_size, round=ROUNDS_DICT[args.experiment])
        print("Cycle time MULTIGRAPH for MST architecture: {0:.1f} ms".format(cycle_time_multigraph))

        result_dir_mst = os.path.join(result_dir, "mst")
        if not os.path.exists(result_dir_mst):
            os.makedirs(result_dir_mst)
        json_path = os.path.join(result_dir_mst, "delay_time_list.json")
        with open(json_path, "w") as f:
            json.dump(list_delay_time_multigraph, f)
        for i in range(len(mst_multigraph)):
            nx.write_gml(mst_multigraph[i], os.path.join(result_dir_mst, "mst_{}.gml".format(i)))

        cycle_time_dict['mst_multigraph'] = [cycle_time_multigraph]


        # delta-MBST
        delta_mbst, best_cycle_time, best_delta = \
            get_delta_mbst_overlay(connectivity_graph.copy(), args.computation_time, args.model_size)

        delta_mbst_multigraph = generate_multigraph(delta_mbst.copy(), args.computation_time, args.model_size, t_max=args.t_max)
        list_delay_time_multigraph, cycle_time_multigraph = compute_cycle_time_multigraph(delta_mbst_multigraph,
                                                                                          args.computation_time,
                                                                                          args.model_size, round=ROUNDS_DICT[args.experiment])
        print("Cycle time MULTIGRAPH for delta-MBST architecture: {0:.1f} ms".format(cycle_time_multigraph))

        result_dir_delta_mst = os.path.join(result_dir, "delta_mst")
        if not os.path.exists(result_dir_delta_mst):
            os.makedirs(result_dir_delta_mst)
        json_path = os.path.join(result_dir_delta_mst, "delay_time_list.json")
        with open(json_path, "w") as f:
            json.dump(list_delay_time_multigraph, f)
        for i in range(len(delta_mbst_multigraph)):
            nx.write_gml(delta_mbst_multigraph[i], os.path.join(result_dir_delta_mst, "delta_mst_{}.gml".format(i)))

        cycle_time_dict['delta_mst_multigraph'] = [cycle_time_multigraph]

        # Star
        star = get_star_overlay(connectivity_graph.copy(), args.centrality)

        star_multigraph = generate_multigraph(star.copy(), args.computation_time, args.model_size, t_max=args.t_max)
        list_delay_time_multigraph, cycle_time_multigraph = compute_cycle_time_multigraph(star_multigraph,
                                                                                          args.computation_time,
                                                                                          args.model_size, round=ROUNDS_DICT[args.experiment])
        cycle_time_multigraph = (cycle_time_multigraph - args.computation_time) * 2 + args.computation_time

        print("Cycle time MULTIGRAPH for STAR architecture: {0:.1f} ms".format(cycle_time_multigraph))

        result_dir_star = os.path.join(result_dir, "star")
        if not os.path.exists(result_dir_star):
            os.makedirs(result_dir_star)
        json_path = os.path.join(result_dir_star, "delay_time_list.json")
        with open(json_path, "w") as f:
            json.dump(list_delay_time_multigraph, f)
        for i in range(len(star_multigraph)):
            nx.write_gml(star_multigraph[i], os.path.join(result_dir_star, "star_{}.gml".format(i)))

        cycle_time_dict['star_multigraph'] = [cycle_time_multigraph]

    json.dump(cycle_time_dict, open(cycle_time_json, "w"))
