import torch
from torch import nn
from dataloader import Dataloader
from config.config import *
from adjacencymatricesbuilder import MatricesBuilder
import math
from utils import *

class ConvolutionalLayer(nn.Module):
    def __init__(self, adjacency_matrices, in_features, out_features):
        super().__init__()
        self.adjacency_matrices = adjacency_matrices
        self.entities_count = len(adjacency_matrices[0])
        self.relations_count = len(adjacency_matrices)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.rand(self.relations_count, self.out_features, self.in_features))    # W has size ((d+1), d).

    def forward(self, x):
        y = torch.zeros(self.entities_count, self.out_features)

        for i in range(0, self.relations_count):
            temp = torch.matmul(torch.matmul(self.adjacency_matrices[i], x), self.weight[i].t())
            y += temp

        return y

class DistMult (nn.Module):
    def __init__ (self, features_count, relations_count):
        super().__init__()
        self.features_count = features_count
        self.relations_count = relations_count
        self.relations_matrices = torch.empty(relations_count, features_count, features_count)

        for i in range (0, relations_count):
            self.relations_matrices[i] = torch.diag(torch.rand(features_count))

        self.relations_matrices = nn.Parameter(self.relations_matrices)

        print(self.relations_matrices)

    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.relations_matrices), x.t())
    
class BasicRGCN (nn.Module):
    def __init__(self, adjacency_matrices, in_features, out_features, matrice_builder, layers_count=2):
        super().__init__()
        self.layers_count = layers_count
        self.model = nn.Sequential()
        self.matrice_builder = matrice_builder

        for i in range(layers_count):
            self.model.append(ConvolutionalLayer(adjacency_matrices, in_features, out_features))
            self.model.append(nn.ReLU())
            
        self.model.append(DistMult(out_features, adjacency_matrices.size()[0]))

        print(self.model)

    def forward(self, x):
        return self.model(x)
    
<<<<<<< HEAD
class Loss(nn.module):
    def __init__(self):
        super().__init__()
        
    def forward (self, predicted_values, training_examples):
        # pred = Matrice N * N probabilité de la relation entre les pairs de noeuds
        # y = liste d'exemples positifs et négatifs de lien entre des pairs de noeuds
        negative_examples = 0        
        loss = 0        
        
        for samples in training_examples:
            entity_1_index = self.matrice_builder.get_index(samples[0])
            entity_2_index = self.matrice_builder.get_index(samples[2])
            
            predicted_value = predicted_values[entity_1_index][entity_2_index]
            label = samples[3]  # 0 or 1
            
            if label == 0:
                negative_examples += 1
            
            sig_value = sigmoid(predicted_value)
            
            loss += label * math.log (sig_value) + (1 - label) * math.log (1 - sig_value)
            
        loss = (-1) / ((1 + negative_examples) * training_examples.size()) * loss
        
        return loss

    
=======
    def loss (self, pred, y):
        # pred = Matrice N * N probabilité de la relation entre les pairs de noeuds
        # y = liste d'exemples positifs et négatifs de lien entre des pairs de noeuds
        pass
>>>>>>> 179af60c9e9b494228702155a3c4bb071a263907


if __name__ == '__main__':
    # loader = Dataloader ()
    # samples = sorted(loader.load_sample_by_drug_types(drug_types_list, entities_limit))
    # relations = loader.load_relations_triples(relations_list, entities_limit)

    # builder = MatricesBuilder(samples, relations)
    # matrices = builder.construct_matrices()

    # rgcn = BasicRGCN(matrices, 8, 4)
    d = DistMult(4, 5)