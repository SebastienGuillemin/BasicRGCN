import torch
from torch.utils.data import Dataset
from random import randrange
from datamanager import DataManager

class RelationDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.Y = []
        self.negative_examples = 0

        for x, y in data:
            self.X.append(x)
            self.Y.append(torch.tensor(float(y)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def add(self, data):
        for x, y in data:
            self.X.append(x)
            self.Y.append(torch.tensor(float(y)))

    def negative_sampling(self, negative_examples_count, data_manager: DataManager, relation=None):
        random_upper_bound_entity = data_manager.get_entity_count()
        random_upper_bound_relation = data_manager.get_relation_name_count()
        relation_name = relation

        for i in range (negative_examples_count):
            if relation == None:
                index_relation = randrange(random_upper_bound_relation)
                relation_name = data_manager.get_relation_name(index_relation)

            index_1 = randrange(random_upper_bound_entity)
            entity_1 = data_manager.get_entity(index_1)
            if (entity_1 == None):
                print("None : ", index_1)
  
            new_tuple = None

            while True:
                index_2 = randrange(random_upper_bound_entity)
                entity_2 = data_manager.get_entity(index_2)
                new_tuple = (entity_1, relation_name, entity_2)
                if entity_1 != entity_2 and new_tuple not in data_manager.get_relations():
                    break
            self.X.append(new_tuple)
            self.Y.append(torch.tensor(float(0.0)))
        
        print(negative_examples_count, ' negative examples added.')
