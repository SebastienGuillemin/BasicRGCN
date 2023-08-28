import torch
from math import log
from torch import nn
from utils import sigmoid
from graph import Graph
from datamanager import DataManager

class BasicRGCN (nn.Module):
    def __init__(self, graph: Graph, in_features, out_features, data_manager: DataManager, layer_count=2):
        super().__init__()
        self.graph = graph
        self.relations_count = self.graph.get_relations_count_without_self_loop()
        self.data_manager = data_manager
        self.model = nn.Sequential()

        for i in range (layer_count):
            self.model.append(ConvolutionalLayer(self.relations_count, in_features, out_features))

        self.model.append(DistMult(out_features, self.relations_count))
        # self.model.append(nn.Tanh())

    def forward(self, x):
        pred = self.model.forward(self.graph)

        batch_size = len(x[0])
        res = torch.empty(batch_size)

        for i in range(batch_size):
            relation_tuple = (x[0][i], x[1][i], x[2][i])
            entity_1_index, relation_index, entity_2_index = self.data_manager.get_relation_entities_index(relation_tuple)           
            res[i] = pred[relation_index][entity_1_index][entity_2_index]
        
        return res

class ConvolutionalLayer(nn.Module):
    def __init__(self, relations_count, in_features_count, out_features_count):
        super().__init__()
        self.relations_count = relations_count
        self.in_features_count = in_features_count
        self.out_features_count = out_features_count

        self.weight = nn.Parameter(torch.randn(self.relations_count, self.out_features_count, self.in_features_count))    # W has size ((d+1), d).

    def forward(self, graph: Graph):
        entities_count = graph.get_entities_count()
        adjacency_matrices = graph.get_adjacency_matrices()

        y = torch.zeros(entities_count, self.out_features_count).to(self.weight.device)
        features = graph.get_features()
        
        for i in range(0, self.relations_count):
            c_i_r = torch.count_nonzero(adjacency_matrices[i], dim=1)   # Use cache for the 3 next lines ?
            c_i_r[c_i_r == 0] = 1   # Used to avoid 0 division.
            c_i_r = (1 / c_i_r).reshape(-1, 1)

            y += torch.mul(c_i_r, torch.matmul(torch.matmul(adjacency_matrices[i], features), self.weight[i].t()))

        y = torch.sigmoid(y)

        return Graph(graph.get_name(), graph.get_adjacency_matrices(), y)

class DistMult (nn.Module):
    def __init__ (self, features_count, relations_count):
        super().__init__()
        self.features_count = features_count
        self.relations_count = relations_count
        self.relations_matrices = torch.empty(relations_count, features_count, features_count)

        for i in range (0, self.relations_count):
            self.relations_matrices[i] = torch.diag(torch.randn(features_count))

        self.relations_matrices = nn.Parameter(self.relations_matrices)

    def forward(self, x: Graph):
        entity_count = x.get_entities_count()
        features = x.get_features()

        res = torch.empty(self.relations_count, entity_count, entity_count)
        for i in range (0, self.relations_count):
            res[i] = torch.matmul(torch.matmul(features, self.relations_matrices[i]), features.t())

        return res

class GraphReLU (nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, graph: Graph):
        features = graph.get_features()
        features = self.relu(features)

        graph.set_features(features)
        return graph

class Dropout (nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout()

    def forward(self, graph: Graph):
        features = graph.get_features()
        features = self.dropout(features)
        graph.set_features(features)

        return graph
    
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward (self, predicted_values, labels):
        device = predicted_values.device
        loss = torch.zeros(1, requires_grad=True).to(device)

        negative_examples, positive_examples = 0, 0

        for i in range(labels.size()[0]):
            label = labels[i]

            if label.item() == 0:
                negative_examples += 1
            else:
                positive_examples += 1

            predicted_value = predicted_values[i]
            
            sigmoid_value = torch.sigmoid(predicted_value).to(device)            
            loss += label * torch.log(sigmoid_value) + (1 - label) * torch.log(1 - sigmoid_value)

        return (-1 / ((1 + negative_examples) * positive_examples)) * loss
