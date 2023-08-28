import torch
from graph import Graph
from math import floor
from config.config import features as features_names

class DataManager():
    def __init__(self, entities_features, relations) :
        # TODO : à adapter pour ne prendre en compte que les échantillons
        self.relations = relations
        self.entities = {}
        self.indexes_cache = {}
        self.relation_name_mapping = {}
        
        self._compute_indexes(entities_features)

        self.entities_count = len(self.indexes_cache)
        self.relations_count = len(self.relations)

    def construct_graph(self):        
        # Construct adjacency matrices
        adjacency_matrices = torch.zeros(self.relations_count + 1, self.entities_count, self.entities_count)

        adjacency_matrices[0] = torch.eye(self.entities_count)  # Self loop matrix.

        for relation_name, relation_data in self.relations.items() :
            i = self.get_relation_index(relation_name)
            for triple, label in relation_data:
                indexe_1 = self.get_index(triple[0])
                indexe_2 = self.get_index(triple[2])
                
                if (indexe_1 != None and indexe_2 != None):
                    adjacency_matrices[i][indexe_1][indexe_2] = 1   
        
        # Construct features matrix
        features_matrice = torch.empty(self.entities_count, len(self.entities[next(iter(self.entities))]))  # Size of the feature list of the first element in the entities dictionnary.

        for index, entity in enumerate(self.indexes_cache):
            i = 0
            for name in features_names['Echantillon']:
                if entity not in self.entities:
                    break
                entity_features = self.entities[entity]
                features_matrice[index][i] = entity_features[name]
                i += 1

        features_matrice = features_matrice / features_matrice.max(0, keepdim=True)[0] # Normalize features matrix     

        return Graph('Graph', adjacency_matrices, features_matrice)
    
    def _compute_indexes(self, entities_features):
        cpt_entity = 0
        cpt_relation = 0
        for name in self.relations:
            for (relation_triple, label) in self.relations[name]:
                entity_1 = relation_triple[0]
                relation = relation_triple[1]
                entity_2 = relation_triple[2]

                if entity_1 not in self.entities:
                    self.entities[entity_1] = entities_features.get(entity_1)
                
                if entity_2 not in self.entities:
                    self.entities[entity_2] = entities_features.get(entity_2)

                if entity_1 not in self.indexes_cache and entity_1 in self.entities:
                    self.indexes_cache[entity_1] = cpt_entity
                    cpt_entity += 1
                
                if entity_2 not in self.indexes_cache and entity_2 in self.entities:
                    self.indexes_cache[entity_2] = cpt_entity
                    cpt_entity += 1

                if relation not in self.relation_name_mapping:
                    self.relation_name_mapping[relation] = cpt_relation
                    cpt_relation += 1

    def get_entity_count(self):
        return len(self.entities)
        
    def get_index(self, entity):
        if (entity in self.indexes_cache):   
            index = self.indexes_cache.get(entity)
            return index

        return None
    
    def get_entity(self, index):
        for entity, entity_index in self.indexes_cache.items():
            if entity_index == index:
                return entity
            
        return None

    def get_relation_index(self, relation):
        if (relation in self.relation_name_mapping):   
            relation_name_mapping = self.relation_name_mapping.get(relation)
            return relation_name_mapping

        return None
    
    def get_relations(self, ):
        return self.relations
    
    def get_relation_name (self, index):
        for relation_name, relation_index in self.relation_name_mapping.items():
            if relation_index == index:
                return relation_name
            
        return None
    
    def get_relation_name_count(self):
        return len(self.relation_name_mapping)
    
    def get_relation_entities_index(self, relation_triples):
        index_1 = self.get_index(relation_triples[0])
        relation_index = self.get_relation_index(relation_triples[1])
        index_2 = self.get_index(relation_triples[2])        

        return index_1, relation_index, index_2