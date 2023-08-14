import torch

class MatricesBuilder():
    def __init__(self, entities, relations) :
        self.entities = entities
        self.relations = relations
        self.size_entities = len(self.entities)
        self.size_relations = len(self.relations)
        self.indexes_cache = {}
        self.relation_name_mapping = {}

        self._compute_index()
        self._compute_relation_name_mapping()

    def construct_matrices(self):
        adjacency_matrices = torch.zeros(self.size_relations + 1, self.size_entities, self.size_entities)

        adjacency_matrices[0] = torch.eye(self.size_entities)  # Self loop matrix.

        for relation_name, triples in self.relations.items() :
            i = self.get_relation_name_mapping(relation_name)
            for triple in triples:
                indexe_1 = self.get_index(triple[0])
                indexe2 = self.get_index(triple[2])
                
                if (indexe_1 != None and indexe2 != None):
                    adjacency_matrices[i][indexe_1][indexe2] = 1   
        
        features_matrice = torch.randn(self.size_entities, 2)

        return adjacency_matrices, features_matrice
    
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