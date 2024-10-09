#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random
from collections import defaultdict

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Gabriel Duval"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Gabriel Duval"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Gabriel Duval"
__email__ = "gabriel.duval@etu.u-paris.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    file = open(fastq_file)
    lines = file.readlines()
    max_length = len(lines) // 4 # Reads are in blocks of 4

    for i in range(max_length):
        yield lines[i*4 + 1].strip()

    
# for sequence in read_fastq(Path("../data/eva71_two_reads.fq")):
#     print(sequence)


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i, pos in enumerate(read[:-kmer_size + 1]):
        yield read[i:i+kmer_size]

# for sequence in read_fastq(Path("../data/eva71_two_reads.fq")):
#     for kmer in cut_kmer(sequence, 4):
#         print(kmer)

def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    # Initialize dictionnary
    kmer_dict = defaultdict(int)

    for sequence in read_fastq(fastq_file):
        for kmer in cut_kmer(sequence, kmer_size):
            kmer_dict[kmer] += 1

    return kmer_dict

# test = build_kmer_dict(Path("data/eva71_two_reads.fq"), 3)
# print(test.keys())

def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()

    for key in kmer_dict:
        graph.add_edge(key[:-1], key[1:], weight = kmer_dict[key])

    return graph



def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if not delete_entry_node:
            path = path[1:]
        
        if not delete_sink_node:
            path = path[:-1]
        
        graph.remove_nodes_from(path)
    
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    # Frequency
    if statistics.stdev(weight_avg_list) > 0:
        best_path = weight_avg_list.index(max(weight_avg_list))

    # Length test
    elif statistics.stdev(path_length) > 0:
        best_path = path_length.index(max(path_length))

    # Draw random number otherwise
    else:
        best_path = random.randint(0, len(path_list))

    # Remove best path from the list
    del path_list[best_path]

    # Remove remaining paths
    graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)
    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    # Calculate paths, weights and lengths
    possible_paths = list(all_simple_paths(graph, ancestor_node, descendant_node))
    path_lengths = [len(path) for path in possible_paths]
    path_weights = [path_average_weight(graph, path) for path in possible_paths]

    # Select the best
    graph = select_best_path(graph, possible_paths, path_lengths, 
                             path_weights)

    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    for n in graph.nodes:
        predecessors = list(graph.predecessors(n))
        if len(predecessors) > 1:
            for i in range(len(predecessors)-1):
                for j in range(i+1, len(predecessors)):
                    ancestor = lowest_common_ancestor(graph, predecessors[i], predecessors[j])
                    if ancestor != None:
                        bubble = True
                        break

        if bubble == True:
            chosen_node = n
            break
    
    if bubble:
        graph = simplify_bubbles(solve_bubble(graph, ancestor, chosen_node))

    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    entry_tip = False
    for n in graph.nodes:
        predecessors = list(graph.predecessors(n))
        
                        entry_tip = True
                        break

        if entry_tip == True:
            chosen_node = n
            break
    
    if entry_tip:
        graph = simplify_bubbles(solve_bubble(graph, ancestor, chosen_node))

    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    pass


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    no_predecessors = [n for n in graph.nodes if len(list(graph.predecessors(n))) == 0]
    return no_predecessors


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    no_successors = [n for n in graph.nodes if len(list(graph.successors(n))) == 0]
    return no_successors
    

def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start in starting_nodes:
        for end in ending_nodes:
            if has_path(graph, start, end):
                for path in all_simple_paths(graph, start, end):
                    contig = path[0]
                    path = path[1:]
                    for kmer in path:
                        contig += kmer[-1]
                    contigs.append((contig, len(contig)))
    
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as filin:
        for i, contigs in enumerate(contigs_list):
            filin.write(f'>contig_{i} len={contigs[1]}\n')
            filin.write(textwrap.fill(contigs[0], width=80))
            filin.write('\n')


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Testing 
    fastq_file = args.fastq_file
    kmer_size = args.kmer_size

    test_dico = build_kmer_dict(fastq_file, kmer_size)

    # Build graphs and save contigs
    test_graph = build_graph(test_dico)
    
    starts = get_starting_nodes(test_graph)

    ends = get_sink_nodes(test_graph)

    res = get_contigs(test_graph, starts, ends)

    save_contigs(res, 'tests/testing_save_contig.txt')
    



    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
