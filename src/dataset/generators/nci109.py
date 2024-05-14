from abc import ABCMeta, abstractmethod

from src.core.configurable import Configurable
from src.dataset.dataset_base import Dataset
from src.utils.context import Context

from os import listdir
from os.path import isfile, join

import numpy as np
import networkx as nx

from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator

class NCI109(Generator):

    def init(self):
        base_path = self.local_config['parameters']['data_dir'] #ricorda va nei file config .jsonc sotto la voce do-pairs
        # Path to the files of the "NCI109" dataset
        self.nci109_edge_path = join(base_path, 'NCI109_A.txt')
        self.nci109_graphid_path = join(base_path, 'NCI109_graph_indicator.txt')
        self.nci109_graphlabels_path = join(base_path, 'NCI109_graph_labels.txt')
        self.nci109_nodelabels_path = join(base_path, 'NCI109_node_labels.txt')
        self.generate_dataset()
            
    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.read_adjacency_matrices()

    def read_adjacency_matrices(self):
        #salviamo lista con graph id e ground truth, in posizione i sta la gt del grafo con id i
        f = open(self.nci109_graphlabels_path, 'r')
        lista_idgt = []
        for line in f:
            lista_idgt.append(int(line.strip()))
        f.close()

        #per ogni grafo salviamo un set con tutti i nodi del grafo.
        #tutti i set sono salvati in una lista (set in i(+1)-esima posizione è il set di nodi del grafo i)
        gf = open(self.nci109_graphid_path, 'r')
        gf_lista = gf.readlines()
        gf.close()
        gf_lista = [int(x.strip()) for x in gf_lista]


        #Lista dei nodi e delle corrispondenti labels (nella posizione j c'è l'etichetta del nodo j+1)
        f_labels = open(self.nci109_nodelabels_path, 'r')
        node_labels = f_labels.readlines()
        f_labels.close()
        node_labels = [int(y.strip()) for y in node_labels]

        #print(node_labels)

        listaset = []
        n = 0
        for i in range(len(lista_idgt)):
            nodi = set()
            
            while n < len(gf_lista):
                if (gf_lista[n] == i+1):
                        nodi.add(n + 1)
                        n += 1
                else:  
                    break
            listaset.append(nodi)

        #calcolo le dimensioni di ogni grafo e la dimensione massima consentita
        graph_dims = []

        for graph_set in listaset:
            graph_dims.append(len(graph_set))

        max_dim = np.mean(graph_dims) + np.std(graph_dims)

        #abbiamo tutti i nodi, creaiamo insieme di edges per ogni grafo, partendo da listaset
        #idea: iterare listaset, iterare ogni set e vedere quali nodi stanno in quali edges
        Af = open(self.nci109_edge_path, 'r')
        edges = []
        for line in Af:
            edge = line.split(",")
            edge = [int(x) for x in edge]
            edges.append(tuple(edge))
        Af.close()

        graphs_edges = []

        counter = 0
        counter2 = 0
        for ins in listaset:
            edges_graph = set()
            massimo = max(ins)
            for nodo in ins:
                counter = counter2
                while counter < len(edges):
                    e = edges[counter]
                    uno = e[0]
                    due = e[1]
                    if (nodo == uno):
                        edges_graph.add(e)
            
                    if (uno > massimo or due > massimo):
                        break
                    
                    counter +=1
            graphs_edges.append(edges_graph)
            counter2 = counter

        for i in range(len(lista_idgt)): 
            dim = graph_dims[i]
            if(dim < max_dim):
                #i = id del grafo
                label = lista_idgt[i] #groundtruth
                nodi = listaset[i] #insieme dei nodi di i
                minimo = min(nodi) #primo nodo in ogni grafo
                n_nodi = len(nodi) #numero di nodi di i
                #creiamo la matrice di adiacenza per il grafo i 
                archi = graphs_edges[i]
                matrice_archi = np.zeros((n_nodi, n_nodi)) 

                for arco in archi:
                    n1 = arco[0]
                    n2 = arco[1]
                    matrice_archi[n1 - minimo][n2 - minimo] = 1

            #creiamo matrice node labels
                matrice_labels = np.zeros((n_nodi,1))
                for nodo in nodi:
                    matrice_labels[nodo - minimo][0] = node_labels[nodo - 1]
                
                self.dataset.instances.append(GraphInstance(i, label=label, data=np.array(matrice_archi, dtype=np.int32), node_features=np.array(matrice_labels, dtype=np.float32)))
            


