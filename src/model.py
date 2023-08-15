import torch
from torch import nn
from dataloader import Dataloader
import math
from utils import *
from graph import Graph

class ConvolutionalLayer(nn.Module):
    def __init__(self, relations_count, in_features_count, out_features_count):
        super().__init__()
        self.relations_count = relations_count
        self.in_features_count = in_features_count
        self.out_features_count = out_features_count

        self.weight = nn.Parameter(torch.rand(self.relations_count, self.out_features_count, self.in_features_count))    # W has size ((d+1), d).

    def forward(self, x: Graph):
        entities_count = x.get_entities_count()
        adjacency_matrices = x.get_adjacency_matrices()
        y = torch.zeros(entities_count, self.out_features_count)
        features = x.get_features()
        
        for i in range(0, self.relations_count):
            temp = torch.matmul(adjacency_matrices[i], features)
            temp = torch.matmul(temp, self.weight[i].t())
            y += temp

        x.set_features(y)
        return x

class DistMult (nn.Module):
    def __init__ (self, features_count, relations_count):
        super().__init__()
        self.features_count = features_count
        self.relations_count = relations_count
        self.relations_matrices = torch.empty(relations_count, features_count, features_count)

        for i in range (0, self.relations_count):
            self.relations_matrices[i] = torch.diag(torch.rand(features_count))

        self.relations_matrices = nn.Parameter(self.relations_matrices)

    def forward(self, x: Graph):
        entity_count = x.get_entities_count()
        features = x.get_features()
        res = torch.empty(self.relations_count, entity_count, entity_count)
        
        for i in range (1, self.relations_count):
            res[i] = torch.matmul(torch.matmul(features, self.relations_matrices[i]), features.t())

        return res
    
class BasicRGCN (nn.Module):
    def __init__(self, in_features, out_features, relations_count, layers_count=2):
        super().__init__()
        self.layers_count = layers_count
        self.model = nn.Sequential()

        for i in range(layers_count):
            self.model.append(ConvolutionalLayer(relations_count, in_features, out_features))
            # self.model.append(nn.ReLU())
            
        self.model.append(DistMult(out_features, relations_count))

        print(self.model)

    def forward(self, graph):
        return self.model(graph)
    
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward (self, predicted_values, graph: Graph):
        # pred = Matrice N * N probabilité de la relation entre les pairs de noeuds
        # y = liste d'exemples positifs et négatifs de lien entre des pairs de noeuds
        adjacency_matrixes = graph.get_adjacency_matrices()
        loss = torch.mean((predicted_values - adjacency_matrixes)**2)
        
        return loss


if __name__ == '__main__':
    # loader = Dataloader ()
    # samples = sorted(loader.load_sample_by_drug_types(drug_types_list, entities_limit))
    # relations = loader.load_relations_triples(relations_list, entities_limit)

    # builder = MatricesBuilder(samples, relations)
    # matrices = builder.construct_matrices()

    # rgcn = BasicRGCN(matrices, 8, 4)
    d = DistMult(4, 5)