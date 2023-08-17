import torch
from graph import Graph
from math import floor
from config.config import features as features_names

class GraphBuilder():
    def __init__(self, entities, relations) :
        # TODO : à adapter pour ne prendre en compte que les échantillons
        self.entities = entities
        self.relations = relations
        self.entities_count = len(self.entities)
        self.relations_count = len(self.relations)
        self.indexes_cache = {}
        self.relation_name_mapping = {}

        self._compute_index()
        self._compute_relation_name_mapping()

    def construct_graphs(self, split_ratio=0.7):
        split_index = floor(self.entities_count * split_ratio)        
        
        # Constructu adjacency matrices
        adjacency_matrices = torch.zeros(self.relations_count + 1, self.entities_count, self.entities_count)

        adjacency_matrices[0] = torch.eye(self.entities_count)  # Self loop matrix.

        for relation_name, triples in self.relations.items() :
            i = self.get_relation_name_mapping(relation_name)
            for triple in triples:
                indexe_1 = self.get_index(triple[0])
                indexe2 = self.get_index(triple[2])
                
                if (indexe_1 != None and indexe2 != None):
                    adjacency_matrices[i][indexe_1][indexe2] = 1
        
        adjacency_matrices_training, adjacency_matrices_testing = self._split_adjacency_matrices(adjacency_matrices, split_index)            
        
        # Construct features matrix
        features_matrice = torch.empty(self.entities_count, len(self.entities[next(iter(self.entities))]))  # Size of the feature list of the first element in the entities dictionnary.

        for entity in self.entities:
            index = self.get_index(entity)
            i = 0
            for name in features_names['Echantillon']:
                entity_features = self.entities[entity]
                features_matrice[index][i] = entity_features[name]
                i += 1

        features_matrice = features_matrice / features_matrice.max(0, keepdim=True)[0] # Normalize features matrix
        features_matrice_training, features_matrice_testing = self._split_features_matrice(features_matrice, split_index)        

        return Graph('Training graph', adjacency_matrices_training, features_matrice_training), Graph('Testing graph', adjacency_matrices_testing, features_matrice_testing)
    
    def _split_adjacency_matrices(self, adjacency_matrices, split_index):
        size = adjacency_matrices.size()
        adjacency_matrices_training = torch.empty(size[0], split_index, split_index)
        adjacency_matrices_testing = torch.empty(size[0], size[1] - split_index, size[1] - split_index)
        
        for i in range(0, size[0]):
            adjacency_matrices_training[i] = adjacency_matrices[i, :split_index, :split_index]
            adjacency_matrices_testing[i] = adjacency_matrices[i, split_index:, split_index:]
            
        return adjacency_matrices_training, adjacency_matrices_testing
    
    def _split_features_matrice(self, features_matrice, split_index):
        size = features_matrice.size()
        features_matrice_training = torch.empty(split_index, split_index)
        features_matrice_testing = torch.empty(size[0] - split_index, size[0] - split_index)

        features_matrice_training= features_matrice[:split_index, :]
        features_matrice_testing = features_matrice[split_index:, :]
        
        return features_matrice_training, features_matrice_testing
    
    def _compute_index(self):
        for index, entity in enumerate(self.entities):
            self.indexes_cache[entity] = index

    def get_index (self, entity):
        if (entity in self.indexes_cache):   
            index = self.indexes_cache.get(entity)
            return index

        return None
    
    def _compute_relation_name_mapping(self):
        for index, relation in enumerate(self.relations):
            self.relation_name_mapping[relation] = index

    def get_relation_name_mapping (self, relation):
        if (relation in self.relation_name_mapping):   
            relation_name_mapping = self.relation_name_mapping.get(relation)
            return relation_name_mapping

        return None