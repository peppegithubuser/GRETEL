import networkx as nx
import numpy as np

from src.dataset.manipulators.base import BaseManipulator

#manipulator for GCounterGAN explainer
class GraphPaddingGCGAN(BaseManipulator):
    
    
    def node_info(self, instance):
        #max_nodes=max(self.dataset.num_nodes_values)
        n_nodes=instance.data.shape[0]
        mult=4
        #new_dim= mult if max_nodes==0 else max_nodes+((mult-(max_nodes%mult))%mult)
        #num_padding=max(0,new_dim-n_nodes)
        num_padding= mult if n_nodes==0 else (mult-(n_nodes%mult))%mult
        instance.data=np.pad(instance.data,((0,num_padding),(0,num_padding)),'constant',constant_values=0)
        instance.node_features=np.pad(instance.node_features,((0,num_padding),(0,0)),'constant',constant_values=0)
        return {}