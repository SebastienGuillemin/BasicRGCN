import torch

class Graph():
    def __init__(self, name, adjacency_matrices, features, c_matrices=None):
        self.name = name
        self.adjacency_matrices = adjacency_matrices
        self.features = features

        if c_matrices != None:
            self.c_matrices = c_matrices
        else:
            self.c_matrices = torch.empty(self.get_relations_count(), self.get_entities_count(), 1)
            self.compute_c_matrices()
        
        if (self.adjacency_matrices.size()[1] != self.adjacency_matrices.size()[2]):
            raise Exception('The adjacency matrix must have as many rows as columns.')
        
        if (self.adjacency_matrices.size()[1] != self.features.size()[0]):
            raise Exception('The feature matrix must have as many rows as the adjacency matrix.')
        
    def get_name(self):
        return self.name
                            
    def get_adjacency_matrices(self):
        return self.adjacency_matrices
    
    def get_features(self):
        return self.features
    
    def get_features_by_index(self, index):
        return self.features[index]

    def get_c_matrices(self):
        return self.c_matrices
    
    def set_features(self, features):
        self.features = features
    
    def get_relations_count(self):
        return self.adjacency_matrices.size()[0]
    
    def get_entities_count(self):
        return self.adjacency_matrices.size()[1]
    
    def compute_c_matrices(self):
        for relation_index in range(self.get_relations_count()):
            neighbors = torch.sum(self.adjacency_matrices[relation_index], dim=0)
            for i in range(self.get_entities_count()):
                if neighbors[i] != 0:
                    self.c_matrices[relation_index][i] = 1 / neighbors[i].item()
                else:
                    self.c_matrices[relation_index][i] = 0

    
    def __str__(self):
        # print(self.adjacency_matrices)
        # print(self.features)

        adjacency_matrices_size = self.adjacency_matrices.size()
        return '%s: \n- %d entities\n- %d relations\n- %d features\n' % (self.name, adjacency_matrices_size[1], adjacency_matrices_size[0], self.features.size()[1])
    
    def to(self, device):
        self.adjacency_matrices = self.adjacency_matrices.to(device)
        self.features = self.features.to(device)
        self.c_matrices = self.c_matrices.to(device)