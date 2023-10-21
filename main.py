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
                vertices.update(line[:-1]) # ajoute le sommet à la liste sans le \n
            break # on arrête de lire les lignes
        edges.append(line) # ajoute l'arête à la liste
        vertices.update(line.split()) # ajoute les sommets de l'arête à la liste
    graph = nx.parse_edgelist(edges, create_using=nx.Graph(), nodetype=int) # crée le graphe à partir de la liste des arêtes
    for v in vertices:
        if int(v) not in graph.nodes():
            graph.add_node(v)
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

def getMaxDegreeVertex(graph):
    """Renvoie le sommet de degré maximal du graphe

    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX

    Retour:
    -
    Le label du sommet de degré maximal et son degré
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph doit être un graphe NetworkX")
    if len(graph) == 0:
        return -1, -1
    return max(graph.degree(), key=lambda x: x[1])

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
        v = getMaxDegreeVertex(graph)[0]
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

    colors = iter(['red', 'blue', 'magenta', 'orange', 'green', 'brown', 'pink', 'yellow', 'cyan', 'purple']) # pour avoir une couleur différente pour chaque couverture

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
    #plt.savefig(f'img/graph_{i}.png')
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

def showTimes(n, algs, log=False):
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
    
    colors = iter(['red', 'blue', 'magenta', 'orange', 'green', 'brown', 'pink', 'yellow', 'cyan', 'purple']) # pour avoir une couleur différente pour chaque courbe

    for alg in algs:
        if not callable(alg): # vérifie que alg est une fonction
            raise TypeError(f"{alg.__name__} doit être une fonction")
        if log: times = np.log(timeAlgo(alg, n)) # calcule les temps d'exécution pour l'algorithme de couverture avec une échelle logarithmique
        else: times = timeAlgo(alg, n) # calcule les temps d'exécution pour l'algorithme de couverture
        plt.plot(range(n//10, n + 1, n//10), times, label=alg.__name__, color=next(colors)) # affiche la courbe des temps d'exécution
        if log: print(f"{alg.__name__} éxécuté en {round(sum(np.exp(times)), 2)} secondes")
        else: print(f"{alg.__name__} éxécuté en {round(sum(times), 2)} secondes")

    plt.xlabel('Nombre de sommets')
    if log: plt.ylabel('Temps (log(s))')
    else: plt.ylabel('Temps (s)')
    plt.legend()

    i = 0
    while os.path.exists(f'img/times_{i}.png'): # sauvegarde le graphique dans un fichier png avec un nom unique
        i += 1
    plt.savefig(f'img/times_{i}.png')
    plt.show()


def branch_and_bound_0(graph, C=None):
    if not graph or not graph.edges():
        return C or set()

    C = C or set()

    u, v = list(graph.edges())[0] 

    solution1 = branch_and_bound_0(deleteVertex(graph, u), C | {u})
    solution2 = branch_and_bound_0(deleteVertex(graph, v), C | {v})

    return solution1 if len(solution1) < len(solution2) else solution2


def branch_and_bound_1(graph):
    """Algorithme de branch and bound pour la couverture de graphe optimale
    
    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX
    
    Retour:
    -
    best_cover: set
        Une couverture optimale du graphe"""
    stack = [(graph.copy(), set())]
    best_solution = float('inf')
    best_cover = set()

    while stack:
        current_graph, current_cover = stack.pop()

        # Si la solution courante est plus grande que la meilleure solution, on élague
        if len(current_cover) >= best_solution: 
            continue
        
        # Calcul de la borne inférieure
        m, n = len(current_graph.edges()), len(current_graph.nodes())
        b1 = np.ceil(m / max(1, getMaxDegreeVertex(current_graph)[1])) 
        b2 = len(algo_couplage(current_graph)) / 2
        b3 = (2*n - 1 - np.sqrt(max((2*n - 1)**2 - 8*m, 0))) / 2
        lower_bound = max(b1, b2, b3)

        # Si toutes les arêtes ont été couvertes ou si la solution courante est plus grande que la meilleure solution
        if not current_graph.edges() or len(current_cover) + lower_bound >= best_solution:
            # Si la solution courante est meilleure que la meilleure solution et que toutes les arêtes ont été couvertes
            if not current_graph.edges() and len(current_cover) < best_solution:
                # On met à jour la meilleure solution
                best_solution = len(current_cover)
                best_cover = current_cover
            # On élague (dans le cas où toutes les arêtes n'ont pas été couvertes mais que la solution courante est pire que la meilleure solution)
            continue
        
        # On choisit un sommet de degré non nul
        i = 0
        while current_graph.degree(u:=list(current_graph.nodes())[i]) == 0:
            i += 1
        # On choisit un voisin de u
        v = next(iter(current_graph.neighbors(u)))

        # On branch avec u ajouté à la couverture
        G1 = current_graph.copy()
        G1.remove_node(u)
        stack.append((G1, current_cover | {u}))

        # On branch avec v ajouté à la couverture
        G2 = current_graph.copy()
        G2.remove_node(v)
        stack.append((G2, current_cover | {v}))

    return best_cover

def branch_and_bound_2(graph):
    """Algorithme de branch and bound pour la couverture de graphe optimale
    
    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX
    
    Retour:
    -
    best_cover: set
        Une couverture optimale du graphe"""
    stack = [(graph.copy(), set())]
    best_solution = float('inf')
    best_cover = set()

    while stack:
        current_graph, current_cover = stack.pop()

        # Si la solution courante est plus grande que la meilleure solution, on élague
        if len(current_cover) >= best_solution: 
            continue
        
        # Calcul de la borne inférieure
        m, n = len(current_graph.edges()), len(current_graph.nodes())
        b1 = np.ceil(m / max(1, getMaxDegreeVertex(current_graph)[1])) 
        b2 = len(algo_couplage(current_graph)) / 2
        b3 = (2*n - 1 - np.sqrt(max((2*n - 1)**2 - 8*m, 0))) / 2
        lower_bound = max(b1, b2, b3)

        # Si toutes les arêtes ont été couvertes ou si la solution courante est plus grande que la meilleure solution
        if not current_graph.edges() or len(current_cover) + lower_bound >= best_solution:
            # Si la solution courante est meilleure que la meilleure solution et que toutes les arêtes ont été couvertes
            if not current_graph.edges() and len(current_cover) < best_solution:
                # On met à jour la meilleure solution
                best_solution = len(current_cover)
                best_cover = current_cover
            # On élague (dans le cas où toutes les arêtes n'ont pas été couvertes mais que la solution courante est pire que la meilleure solution)
            continue
        
        # On choisit un sommet de degré non nul
        i = 0
        while current_graph.degree(u:=list(current_graph.nodes())[i]) == 0:
            i += 1
        # On choisit un voisin de u
        v = next(iter(current_graph.neighbors(u)))

        # On branch avec u ajouté à la couverture
        G1 = current_graph.copy()
        G1.remove_node(u)
        stack.append((G1, current_cover | {u}))

        # On branch avec v ajouté à la couverture
        G2 = current_graph.copy()
        G2.remove_node(v)
        for w in list(G2.neighbors(u)):  # On retire tous les voisins de u qu'on ajoute à la couverture
            G2.remove_node(w)
            current_cover.add(w)
        G2.remove_node(u) # On retire u
        stack.append((G2, current_cover | {v}))

    return best_cover

def branch_and_bound_3(graph):
    """Algorithme de branch and bound pour la couverture de graphe optimale
    
    Paramètres:
    -
    graph: networkx.Graph
        Un graphe NetworkX
    
    Retour:
    -
    best_cover: set
        Une couverture optimale du graphe"""
    stack = [(graph.copy(), set())]
    best_solution = float('inf')
    best_cover = set()

    while stack:
        current_graph, current_cover = stack.pop()

        # Si la solution courante est plus grande que la meilleure solution, on élague
        if len(current_cover) >= best_solution: 
            continue
        
        # Calcul de la borne inférieure
        m, n = len(current_graph.edges()), len(current_graph.nodes())
        b1 = np.ceil(m / max(1, getMaxDegreeVertex(current_graph)[1])) 
        b2 = len(algo_glouton(current_graph)) / 2
        b3 = (2*n - 1 - np.sqrt(max((2*n - 1)**2 - 8*m, 0))) / 2
        lower_bound = max(b1, b2, b3)

        # Si toutes les arêtes ont été couvertes ou si la solution courante est plus grande que la meilleure solution
        if not current_graph.edges() or len(current_cover) + lower_bound >= best_solution:
            # Si la solution courante est meilleure que la meilleure solution et que toutes les arêtes ont été couvertes
            if not current_graph.edges() and len(current_cover) < best_solution:
                # On met à jour la meilleure solution
                best_solution = len(current_cover)
                best_cover = current_cover
            # On élague (dans le cas où toutes les arêtes n'ont pas été couvertes mais que la solution courante est pire que la meilleure solution)
            continue
        
        # On choisit un sommet de degré maximal
        u = getMaxDegreeVertex(current_graph)[0]
        # On choisit un voisin de u
        v = next(iter(current_graph.neighbors(u)))

        # On branch avec u ajouté à la couverture
        G1 = current_graph.copy()
        G1.remove_node(u)
        stack.append((G1, current_cover | {u}))

        # On branch avec v ajouté à la couverture
        G2 = current_graph.copy()
        G2.remove_node(v)
        for w in list(G2.neighbors(u)):  # On retire tous les voisins de u qu'on ajoute à la couverture
            G2.remove_node(w)
            current_cover.add(w)
        G2.remove_node(u) # On retire u
        stack.append((G2, current_cover | {v}))

    return best_cover

def rapportApproximation(n, algo, iter=100):
    if not isinstance(n, int) and not isinstance(algo, callable):
        raise TypeError("n doit être un entier et algo doit être une fonction")
    rapports = []
    for _ in range(iter):
        g = generateRandomGraph(n, 1/np.sqrt(n))
        rapports.append(len(algo(g)) / len(branch_and_bound_3(g)))
    return rapports


if __name__ == "__main__":
    g = generateRandomGraph(25,1/5)
    #g = loadGraph('exempleinstance.txt')
    #showGraphs(g, [algo_glouton, branch_and_bound_2])
    #showTimes(100, [algo_couplage, algo_glouton, branch_and_bound_3], log=True)
    #showTimes(25, [branch_and_bound_3, branch_and_bound_2, branch_and_bound_1, branch_and_bound_0], log=True)
    #showTimes(25, [branch_and_bound_1, branch_and_bound_0])
    #showTimes(40, [branch_and_bound_3, branch_and_bound_2, branch_and_bound_1], log=True)
    #showTimes(80, [branch_and_bound_2, branch_and_bound_3])
    for alg in (algo_couplage, algo_glouton):
        print(f"{alg.__name__}:")
        for n in range(10,51,10):
            rapports = rapportApproximation(n, alg)
            print("\tn = {}, moyenne = {}, pire = {}".format(n, round(np.mean(rapports),2), round(np.max(rapports),2)))