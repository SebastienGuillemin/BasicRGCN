class Graph():
    def __init__(self, name, adjacency_matrices, features):
        self.name = name
        self.adjacency_matrices = adjacency_matrices
        self.features = features
        
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
    
    def set_features(self, features):
        self.features = features
    
    def get_relations_count(self):
        return self.adjacency_matrices.size()[0]
    
    def get_entities_count(self):
        return self.adjacency_matrices.size()[1]
    
    def __str__(self):
        # print(self.adjacency_matrices)
        # print(self.features)

        adjacency_matrices_size = self.adjacency_matrices.size()
        return '%s: \n- %d entities\n- %d relations\n- %d features\n' % (self.name, adjacency_matrices_size[1], adjacency_matrices_size[0], self.features.size()[1])
    
    def to(self, device):
        self.adjacency_matrices = self.adjacency_matrices.to(device)
        self.features = self.features.to(device)