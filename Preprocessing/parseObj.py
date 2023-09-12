import numpy as np
import networkx as nx
from scipy.spatial import KDTree

def calcularGrafoYArbol( fileObj, fileRadios ):
    radios = np.load(fileRadios)

    verticesCrudos = []
    vertices = []
    lineas = []

    for row in fileObj:
        if row[0:2] == 'v ':
      
            vertice = np.fromstring( row[2:], dtype=np.float32, sep=' ')
            
            #print("nodo", len(verticesCrudos),{'radio': [vertice[0], vertice[1], vertice[2],radios[len(verticesCrudos)][0], radios[len(verticesCrudos)][1]]})
            #vertices.append( (len(verticesCrudos), {'posicion': Vec3( vertice[0], vertice[1], vertice[2]), 'radio': radios[len(verticesCrudos)]} ))
            vertices.append( (len(verticesCrudos), {'radio': (vertice[0], vertice[1], vertice[2],radios[len(verticesCrudos)])} ))

            verticesCrudos.append(vertice)
        elif row[0:2] == 'l ':
            linea = np.fromstring(row[2:], dtype=np.uint32, sep=' ')
            lineas += [ (linea[i] - 1, linea[i+1] - 1) for i in range( len(linea) - 1)]
        else:
            continue

    G = nx.Graph()
    G.add_nodes_from( vertices )
    G.add_edges_from( lineas )
    return G, KDTree( verticesCrudos )

def combinarNodos( grafo, repetidos ):
    gruposYaProcesados = {}
    for grupo in repetidos:
        if str(grupo) in gruposYaProcesados:
            continue

        gruposYaProcesados[str(grupo)] = True
        nodos = [ grafo.nodes[nodo] for nodo in grupo ]
        aristas = np.unique( np.concatenate([ [arista[1] for arista in grafo.edges(nodo) if arista[1] not in grupo] for nodo in grupo]))


       
        nombreNodo = np.min(grupo)
       
        nuevoVertice = {
            #'posicion': np.sum( [ nodo['posicion'] for nodo in nodos ] ) / len(nodos),
            'radio': grafo.nodes[nombreNodo]['radio'] }
        grafo.remove_nodes_from(grupo)
        grafo.add_nodes_from( [(nombreNodo, nuevoVertice)])
        grafo.add_edges_from( [(nombreNodo, arista) for arista in aristas])

def calcularMatriz( fileObj, fileRadios, r=1e-5 ):
   
    grafo, arbolVertices = calcularGrafoYArbol( fileObj, fileRadios )

    repetidos = [ i for i in arbolVertices.query_ball_tree( arbolVertices, r=r ) if len(i) > 1]
    
    combinarNodos( grafo, repetidos )

    if nx.number_connected_components( grafo ) != 1:
        raise Exception("El grafo no se pudo unificar con r=" + str(r) + ". Se tienen " + str(nx.number_connected_components( grafo )) + " componentes conexas.")
    
    return grafo