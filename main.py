import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import os

def loadGraph(filename):
    """Charge un graphe à partir d'un fichier texte
    
    Paramètres:
    -
    filename: str
        Le chemin vers le fichier texte contenant le graphe
        Le fichier doit être au format .txt et suivre un format précis:
        - La première ligne doit être "Nombre de sommets:"
        - La deuxième ligne doit être le nombre de sommets
        - La troisième ligne doit être "Sommets:"
        - Les lignes suivantes doivent être les labels des sommets
        - La ligne suivante doit être "Nombre d'arêtes:"
        - La ligne suivante doit être le nombre d'arêtes
        - La ligne suivante doit être "Aretes:"
        - Les lignes suivantes doivent être les arêtes du graphe au format "label1 label2"

    Retour:
    -
    graph: networkx.Graph
        Un graphe NetworkX
    
    """
    if not isinstance(filename, str):
        raise TypeError("filename doit être de type str")
    f = open(filename, 'r') # ouvre le fichier
    lines = f.readlines() # charge les lignes
    f.close() # ferme le fichier
    edges = [] # liste des arêtes
    vertices = set() # liste des sommets (sans doublons)
    for i in range(1, len(lines)): 
        line = lines[-i] # lire les lignes à l'envers
        if line[0] == 'A': # si on atteint la ligne "Aretes" on ajoute tous les sommets qui ne sont connectés à rien
            i += 3 # skip les lignes avant la liste des sommets
            for j in range(i, len(lines)): # on lit les sommets à l'envers
                line = lines[-j] 
                if line[0] == 'S': # si on atteint la ligne "Sommets" on arrête de lire
                    break
                vertices.add(line[:-1]) # ajoute le sommet à la liste sans le \n
            break # on arrête de lire les lignes
        edges.append(line) # ajoute l'arête à la liste
        vertices.update(line.split()) # ajoute les sommets de l'arête à la liste
    graph = nx.parse_edgelist(edges, create_using=nx.Graph(), nodetype=int) # crée le graphe à partir de la liste des arêtes
    graph.add_nodes_from(vertices) # ajoute les sommets manquants au graphe
    return graph


def deleteVertex(graph, vertex):
    """Supprime un sommet du graphe

    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX
    vertex: int
        Le label d'un sommet du graphe
    
    Retour:
    -
    graph_copy: graphe
        Une copy du graphe NetworkX d'origine sans le sommet supprimé
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph doit être un graphe NetworkX")
    graph_copy = graph.copy()
    graph_copy.remove_node(vertex)
    return graph_copy

def deleteVertices(graph, vertices):
    """Supprime un sommet du graphe

    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX
    vertices: list
        Une liste de labels de sommets du graphe

    Retour:
    -
    graph_copy: graphe
        Une copy du graphe NetworkX d'origine sans les sommets supprimés
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph doit être un graphe NetworkX")
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(vertices)
    return graph_copy

def getVerticesDegrees(graph):
    """Renvoie les degrés de tous les sommets du graphe

    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX

    Retour:
    -
    Un dictionnaire avec les labels des sommets en clé et leurs degrés respectifs en valeurs
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph doit être un graphe NetworkX")
    return dict(graph.degree())

def generateRandomGraph(n, p):
    """Génère un graphe aléatoire à partir de la loi de probabilité de Erdős-Rényi

    Paramètres:
    -
    n: int
        Le nombre de sommets 
    p: float
        La probabilité d'avoir une arête entre deux sommets

    Retour:
    -
    Un graphe NetworkX
    """
    if p < 0 or p > 1 or n < 0:
        raise ValueError("n doit être positif et p doit être entre 0 et 1 exclus")
    return nx.fast_gnp_random_graph(n, p)

def algo_couplage(graph):
    """Trouve une couverture du graphe à partir d'un couplage

    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX

    Retour:
    -
    C: list
        Une liste de labels de sommets qui forment une couverture du graphe
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph doit être un graphe NetworkX")
    C = []
    for edge in graph.edges():
        if not any(v in C for v in edge):
            for v in edge:
                C.append(v)
    return C

def algo_glouton(graph):
    """Trouve une couverture du graphe à partir d'un algorithme glouton

    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX

    Retour:
    -
    C: list
        Une liste de labels de sommets qui forment une couverture du graphe
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph doit être un graphe NetworkX")
    C = []
    graph = graph.copy()
    while len(graph.edges()) > 0:
        v = max(graph.degree(), key=lambda x: x[1])[0]
        C.append(v)
        for e in graph.edges():
            if v in e:
                graph.remove_edge(*e)
    return C

def showGraphs(graph, algs, fs=10, ns=200):
    """Affiche le graphe avec les couvertures trouvées par les deux algorithmes de couverture (couplage et glouton)
    
    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX
    algs: list
        Une liste d'algorithmes de couverture à afficher
    fs: int
        La taille de la police dans les labels des sommets
    ns: int
        La taille des sommets
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph doit être un graphe NetworkX")

    layout = nx.spring_layout(graph, seed=1) # pour avoir le même layout sur nos deux graphes

    _, axs = plt.subplots(nrows=1, ncols=len(algs), figsize=(10, 5)) # pour avoir deux graphes côte à côte

    colors = iter(['red', 'blue', 'purple', 'orange', 'green', 'brown', 'pink', 'yellow', 'cyan', 'magenta']) # pour avoir une couleur différente pour chaque couverture

    for i, alg in enumerate(algs):
        if not callable(alg): # vérifie que alg est une fonction
            raise TypeError(f"{alg.__name__} doit être une fonction")
        cover = alg(graph) # trouve une couverture du graphe avec l'algorithme alg
        color = next(colors) # pour avoir une couleur différente pour chaque couverture
        cmap = ['grey' if v not in cover else color for v in g] # colorie les sommets de la couverture
        axs[i].set_title(f'{alg.__name__} {len(cover)} sommets')
        nx.draw(g, pos=layout, with_labels=True, node_color=cmap, font_color='white', font_size=fs, node_size=ns, ax=axs[i])

    i = 0
    while os.path.exists(f'img/graph_{i}.png'): # sauvegarde le graphique dans un fichier png avec un nom unique
        i += 1
    plt.savefig(f'img/graph_{i}.png')
    plt.show()

def tic():
    """Lance un chrono"""
    global startTime
    startTime = time.time()
    return startTime

def tac():
    """Renvoie le temps écoulé depuis l'exécution du dernier tic()"""
    return time.time() - startTime

def timeAlgo(alg, nMax):
    """Calcule le temps d'exécution d'un algorithme pour 10 graphes de tailles croissantes
    
    Les graphes sont générés aléatoirement

    Paramètres:
    -
    alg: fonction
        L'algorithme de couverture à tester
    nMax: int
        La taille du plus grand graphe à tester
    
    Retour:
    -
    times: list
        Une liste des temps d'exécution de l'algorithme pour chaque graphe testé
    """
    times = [] # liste des temps d'exécution
    for n in range(nMax//10, nMax + 1, nMax//10): # pour n allant de nMax/10 à nMax par pas de nMax/10 (ex: 100, 200, 300, ..., 1000)
        g = generateRandomGraph(n, 1/np.sqrt(n)) # génère un graphe aléatoire
        tic() # lance le chrono
        alg(g) # lance l'algorithme
        times.append(tac()) # ajoute le temps d'exécution à la liste
    return times

def showTimes(n, algs):
    """Affiche un graphique montrant les temps d'exécution des algorithmes de couverture pour différentes tailles de graphes

    Paramètres:
    -
    n: int
        La taille du plus grand graphe à tester
    algs: list
        Une liste d'algorithmes de couverture à tester
    """
    if not isinstance(n, int): # vérifie que n est un entier
        raise TypeError("n doit être un entier")
    
    colors = iter(['red', 'blue', 'purple', 'orange', 'green', 'brown', 'pink', 'yellow', 'cyan', 'magenta']) # pour avoir une couleur différente pour chaque courbe

    for alg in algs:
        if not callable(alg): # vérifie que alg est une fonction
            raise TypeError(f"{alg.__name__} doit être une fonction")
        times = timeAlgo(alg, n) # calcule les temps d'exécution pour l'algorithme de couverture
        plt.plot(range(n//10, n + 1, n//10), times, label=alg.__name__, color=next(colors)) # affiche la courbe des temps d'exécution
        print(f"{alg.__name__} terminé en {round(sum(times), 2)} secondes")

    plt.xlabel('Nombre de sommets')
    plt.ylabel('Temps (s)')
    plt.legend()

    i = 0
    while os.path.exists(f'img/times_{i}.png'): # sauvegarde le graphique dans un fichier png avec un nom unique
        i += 1
    plt.savefig(f'img/times_{i}.png')
    plt.show()


def branch_and_bound(graphe):
    stack = [(graphe, set())]
    best_solution = float('inf')
    best_cover = set()

    while stack:
        current_graph, current_cover = stack.pop()

        if len(current_cover) >= best_solution:
            continue

        if not current_graph or not current_graph.edges():
            if len(current_cover) < best_solution:
                best_solution = len(current_cover)
                best_cover = current_cover
            continue

        u, v = list(current_graph.edges())[0]

        G1 = deleteVertex(current_graph.copy(), u)
        stack.append((G1, current_cover | {u}))

        G2 = deleteVertex(current_graph.copy(), v)
        stack.append((G2, current_cover | {v}))

    return best_cover


if __name__ == "__main__":
    g = generateRandomGraph(12,0.2)
    #g = loadGraph('exempleinstance.txt')
    showGraphs(g, [algo_couplage, algo_glouton, branch_and_bound])
    showTimes(3000, [algo_couplage, algo_glouton])