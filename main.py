import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

def loadGraph(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    edges = []
    for i in range(1, len(lines)):
        if lines[-i][0] == 'A':
            break
        edges.append(lines[-i])
    return nx.parse_edgelist(edges, create_using=nx.Graph())


def deleteVertex(graph, vertex):
    graph_copy = graph.copy()
    graph_copy.remove_node(vertex)
    return graph_copy

def deleteVertices(graph, vertices):
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(vertices)
    return graph_copy

def getVerticesDegrees(graph):
    return dict(graph.degree())

def generateRandomGraph(n, p):
    return nx.fast_gnp_random_graph(n, p)

def algo_couplage(graph):
    C = []
    for edge in graph.edges():
        if not any(v in C for v in edge):
            for v in edge:
                C.append(v)
    return C

def algo_glouton(graph):
    C = []
    graph = graph.copy()
    while len(graph.edges()) > 0:
        v = max(graph.degree(), key=lambda x: x[1])[0]
        C.append(v)
        for e in graph.edges():
            if v in e:
                graph.remove_edge(*e)
    return C

def showGraphs(graph, fs=10, ns=200):
    couplage = algo_couplage(graph)
    glouton = algo_glouton(graph)

    layout = nx.spring_layout(graph, seed=1)

    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    color_couplage = ['grey' if v not in couplage else 'red' for v in g]
    axs[0].set_title(f'Couplage {len(couplage)} sommets')
    nx.draw(g, pos=layout, with_labels=True, node_color=color_couplage, font_color='white', font_size=fs, node_size=ns, ax=axs[0])

    color_glouton = ['grey' if v not in glouton else 'blue' for v in g]
    axs[1].set_title(f'Glouton {len(glouton)} sommets')
    nx.draw(g, pos=layout, with_labels=True, node_color=color_glouton, font_color='white', font_size=fs, node_size=ns, ax=axs[1])

    plt.savefig('img/graph.png')
    plt.show()

def tic():
    global startTime
    startTime = time.time()
    return startTime

def tac():
    return time.time() - startTime

def timeAlgo(alg, nMax):
    times = []
    for n in range(nMax//10, nMax, nMax//10):
        g = generateRandomGraph(n, 0.01)
        tic()
        alg(g)
        times.append(tac())
    return times

def showTimes(n):
    times = timeAlgo(algo_couplage, n)
    plt.plot(range(n//10, n, n//10), times, label='Couplage', color='red')
    times = timeAlgo(algo_glouton, n)
    plt.plot(range(n//10, n, n//10), times, label='Glouton', color='blue')
    plt.xlabel('Nombre de sommets')
    plt.ylabel('Temps (s)')
    plt.legend()
    plt.savefig('img/times.png')
    plt.show()


if __name__ == "__main__":

    #g = generateRandomGraph(12,0.3)
    #showGraphs(g)
    #showTimes(2000)