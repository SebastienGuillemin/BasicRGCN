class Graph():
    def __init__(self, adjacency_matrices, features):
        self.adjacency_matrices = adjacency_matrices
        self.features = features
        
        if (self.adjacency_matrices.size()[1] != self.adjacency_matrices.size()[2]):
            raise Exception('The adjacency matrix must have as many rows as columns.')
        
        if (self.adjacency_matrices.size()[1] != self.features.size()[0]):
            raise Exception('The feature matrix must have as many rows as the adjacency matrix.')
                            
    def get_adjacency_matrices(self):
        return self.adjacency_matrices
    
    def get_features(self):
        return self.features
    
    def set_features(self, features):
        self.features = features
    
    def get_relations_count(self):
        return self.adjacency_matrices.size()[0]
    
    def get_entities_count(self):
        return self.adjacency_matrices.size()[1]