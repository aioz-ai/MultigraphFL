import networkx as nx
import os
import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument('name',
                    type=str,
                    help='name of the network to use;',
                    default='gaia')
parser.add_argument("--number_silos",
                    type=int,
                    help="number of silos",
                    default=6)

args = parser.parse_args()

def remove_silos(underlay, number_silos):
    output_underlay = underlay.copy()
    num_nodes = underlay.number_of_nodes()
    idx_silos = random.sample(range(0, num_nodes), number_silos)
    for i in range(num_nodes):
        if i not in idx_silos:
            output_underlay.remove_node(list(underlay.nodes)[i])
    return output_underlay

if __name__ == "__main__":
    path_to_graph = "./data/{}.gml".format(args.name)

    underlay = nx.read_gml(path_to_graph)
    print("Number of Workers: {}".format(underlay.number_of_nodes()))
    print("Number of links: {}".format(underlay.number_of_edges()))
    removed_underlay = remove_silos(underlay.copy(), args.number_silos)
    nx.write_gml(removed_underlay.copy(), os.path.join("data", "{}_{}.gml".format(args.name, args.number_silos)))
