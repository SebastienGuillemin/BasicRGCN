import torch
from torch import nn, max, min
from utils import *
from graph import Graph

class BasicRGCN (nn.Module):
    def __init__(self, in_features, out_features, relations_count, layers_count=2):
        super().__init__()
        self.layers_count = layers_count
        self.model = nn.Sequential()

        for _ in range(layers_count):
            self.model.append(ConvolutionalLayer(relations_count, in_features, out_features))
            # self.model.append(GraphReLU())
            # self.model.append(Dropout())
            
        self.model.append(DistMult(out_features, relations_count))
        # self.model.append(nn.Sigmoid())

    def forward(self, graph: Graph):
        return self.model(graph)

class ConvolutionalLayer(nn.Module):
    def __init__(self, relations_count, in_features_count, out_features_count):
        super().__init__()
        self.relations_count = relations_count
        self.in_features_count = in_features_count
        self.out_features_count = out_features_count

        self.weight = nn.Parameter(torch.randn(self.relations_count, self.out_features_count, self.in_features_count))    # W has size ((d+1), d).

    def forward(self, x: Graph):
        entities_count = x.get_entities_count()
        adjacency_matrices = x.get_adjacency_matrices()

        y = torch.zeros(entities_count, self.out_features_count).to(self.weight.device)
        features = x.get_features()
        
        for i in range(0, self.relations_count):
            c_i_r = torch.count_nonzero(adjacency_matrices[i], dim=1)   # Use cache for the 3 next lines ?
            c_i_r[c_i_r == 0] = 1   # Used to avoid 0 division.
            c_i_r = (1 / c_i_r).reshape(-1, 1)

            y += torch.mul(c_i_r, torch.matmul(torch.matmul(adjacency_matrices[i], features), self.weight[i].t()))

        y = torch.sigmoid(y)

        return Graph(x.get_name(), x.get_adjacency_matrices(), y)

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
        
    def forward (self, predicted_values, graph: Graph):
        entities_count = graph.get_entities_count()
        adjacency_matrixes = graph.get_adjacency_matrices()
        
        device = adjacency_matrixes.device

        loss = torch.empty(graph.get_relations_count()).to(device)
        ones = torch.ones(entities_count).to(device)

        for i in range(0, graph.get_relations_count()):
            a = torch.mul(adjacency_matrixes[i], torch.log(torch.sigmoid(predicted_values[i]).to(device)))
            b = torch.mul(torch.sub(ones, adjacency_matrixes[i]), torch.log(torch.sub(ones, torch.sigmoid(predicted_values[i]).to(device))))

            loss[i] = torch.sum(torch.add(a, b))

        loss = -1 * torch.sum(loss)
        return loss

        # positive = torch.count_nonzero(adjacency_matrixes).item()
        # negative = (entities_count**2) * graph.get_relations_count() - positive        
        # return (-1 / ((1 + negative) * positive)) * loss.sum()