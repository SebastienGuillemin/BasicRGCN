from model import BasicRGCN
import torch
from dataloader import Dataloader
from config.config import features
from graphbuilder import GraphBuilder

def merge_entities(entities_1, entities_2):
    for instance in entities_2:
        if instance not in entities_1:
            entities_1[instance] = entities_2[instance]

def merge_relations(relations_1, relations_2):
    for triplet in relations_2:
        if triplet not in relations_1:
            relations_1.append(triplet)

def delete_relations(sample_1, sample_2, relations):
    relation_1 = (sample_1, 'estProcheDe', sample_2, 1)
    relation_2 = (sample_2, 'estProcheDe', sample_1, 1)
    
    if (relation_1 in relations):
        relations.remove(relation_1)

    if (relation_2 in relations):
        relations.remove(relation_2)


if __name__ == '__main__':
    device = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )
    print(f'Using {device} device')

    model = BasicRGCN(2, 2, 2).to(device)
    model.load_state_dict(torch.load('model.pth'))

    ## Retrieve sampels neighborhood.
    loader = Dataloader()
    relations_1, entities_1 = loader.load_sample_neighborhood('echantillon_9477', features['Echantillon'])
    relations_2, entities_2 = loader.load_sample_neighborhood('echantillon_10423', features['Echantillon'])

    ## Merge entities and relations construct graph
    merge_entities(entities_1, entities_2)
    merge_relations(relations_1, relations_2)
    print(relations_1)

    ## Delete relation between the two samples if exists
    delete_relations(loader.prefix + 'echantillon_9477', loader.prefix + 'echantillon_10423', relations_1)

    ## Create graph
    builder = GraphBuilder(entities_1, relations_1)
    graph = builder.construct_graph()
    graph.to(device)
    print(graph)

    model.eval()
    with torch.no_grad():
        pred = model(graph)
        print(pred)