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
        self.relations_count = self.graph.get_relations_count()
        self.data_manager = data_manager

        ## Conv. layers
        self.model = nn.Sequential()

        ## DistMult
        self.dist_mult = DistMult(out_features, self.relations_count, self.data_manager)

        for i in range (layer_count):
            self.model.append(ConvolutionalLayer(self.relations_count, in_features, out_features))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print (name, '\n', param.data)

        for name, param in self.dist_mult.named_parameters():
            if param.requires_grad:
                print (name, '\n', param.data)

    def forward(self, x):
        embedded_graph: Graph = self.model.forward(self.graph)

        batch_size = len(x[0])
        res = torch.empty(batch_size)

        for i in range(batch_size):
            relation_tuple = (x[0][i], x[1][i], x[2][i])
            entity_1_index, relation_index, entity_2_index = self.data_manager.get_relation_entities_index(relation_tuple)
            entity_1_embedding = embedded_graph.get_features_by_index(entity_1_index)
            entity_2_embedding = embedded_graph.get_features_by_index(entity_2_index)

            res[i] = self.dist_mult.forward(entity_1_embedding, relation_index, entity_2_embedding)
        
        return res

class ConvolutionalLayer(nn.Module):
    def __init__(self, relations_count, in_features_count, out_features_count):
        super().__init__()
        self.relations_count = relations_count
        self.in_features_count = in_features_count
        self.out_features_count = out_features_count

        ## Weight parameters
        # torch.manual_seed(3)
        self.weight = nn.Parameter(torch.randn(self.relations_count, self.out_features_count, self.in_features_count))    # W has size ((d+1), d).

    def forward(self, graph: Graph):
        entities_count = graph.get_entities_count()
        adjacency_matrices = graph.get_adjacency_matrices()
        c_matrices = graph.get_c_matrices()

        y = torch.zeros(entities_count, self.out_features_count).to(self.weight.device)
        features = graph.get_features()
        
        for i in range(0, self.relations_count):
            y += torch.mul(c_matrices[i], torch.matmul(torch.matmul(adjacency_matrices[i], features), self.weight[i].t()))

        # y = torch.relu(y)
        return Graph(graph.get_name(), graph.get_adjacency_matrices(), y, graph.get_c_matrices())

class DistMult (nn.Module):
    def __init__ (self, features_count, relations_count, data_manager: DataManager):
        super().__init__()
        self.features_count = features_count
        self.relations_count = relations_count
        self.data_manager = data_manager
        
        ## Relations matrices parameters
        self.relations_matrices = torch.empty(relations_count, features_count, features_count)

        for i in range (0, self.relations_count):
            # torch.manual_seed(3 + i)
            self.relations_matrices[i] = torch.diag(torch.randn(features_count))

        self.relations_matrices = nn.Parameter(self.relations_matrices)

    def forward(self, entity_1_embedding, relation_index, entity_2_embedding):
        return torch.sigmoid(torch.matmul(torch.matmul(entity_1_embedding, self.relations_matrices[relation_index]), entity_2_embedding))

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
        # print(predicted_values)

        device = predicted_values.device
        loss = torch.zeros(1, requires_grad=True).to(device)
        examples_count = labels.size()[0]

        # negative_examples, positive_examples = 0, 0

        for i in range(examples_count):

            label = labels[i]

            if label.item() == 0:
                negative_examples += 1
            else:
                positive_examples += 1

            predicted_value = predicted_values[i]
            
            sigmoid_value = torch.sigmoid(predicted_value).to(device)        
            loss += label * torch.log(sigmoid_value) + (1 - label) * torch.log(1 - sigmoid_value)

            ## Erreur sur Log(1 - Sigmoid) ? (1 - Sigmoid retourne sur [0; 1])
            if loss.isnan():
                print(f'Sigmoid  = {sigmoid_value}')
                print(f'Log(Sigmoid)  = {torch.log(sigmoid_value)}')
                print(f'Log(1 - Sigmoid)  = {torch.log(1 - predicted_value)}')
                exit()

        return (-1 / ((1 + negative_examples) * positive_examples)) * loss
