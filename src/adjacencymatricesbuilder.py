import torch

class MatricesBuilder():
    def __init__(self, entities, relations) :
        self.entities = entities
        self.relations = relations
        self.size_entities = len(self.entities)
        self.size_relations = len(self.relations)
        self.indexes_cache = {}

        self._compute_index()

    def construct_matrices(self):
        matrices = torch.zeros(self.size_relations + 1, self.size_entities, self.size_entities)

        matrices[0] = torch.eye(self.size_entities)  # Self loop matrix.

        i = 0
        for _, triples in self.relations.items() :
            for triple in triples:
                indexe_1 = self.get_index(triple[0])
                indexe2 = self.get_index(triple[2])
                
                if (indexe_1 != None and indexe2 != None):
                    matrices[i][indexe_1][indexe2] = 1   
            

        return matrices
    
    def _compute_index(self):
        for index, entity in enumerate(self.entities):
            self.indexes_cache[entity] = index

    def get_index (self, entity):
        if (entity in self.indexes_cache):   
            index = self.indexes_cache.get(entity)
            return index

        return None