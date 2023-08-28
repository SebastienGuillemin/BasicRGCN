from model import BasicRGCN
import torch
from repository import Repository, prefix
from datamanager import DataManager
from config.config import relations_list, drug_type

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
    
    if (relation_1 in relations['estProcheDe']):
        print('remove')
        relations['estProcheDe'].remove(relation_1)

    if (relation_2 in relations['estProcheDe']):
        relations['estProcheDe'].remove(relation_2)
        print('remove')

def retrieve_data():
    data_loader = Repository()
    raw_data_relations = data_loader.load_relations_triples(relations_list)
    raw_data_entities = data_loader.load_sample_by_drug_type(drug_type)
    
    return raw_data_relations, raw_data_entities

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

    ## Retrieve data
    raw_data_relations, raw_data_entities = retrieve_data()

    if (raw_data_relations == None):
        raise Exception('Impossible to retrieve data for relations.')
    else:
        print('%d relation types retrieved :' % (len(raw_data_relations)))
        for key, value in raw_data_relations.items():
            print('     %s : %s triples retrieved.' % (key, str(len(value))))

    if (raw_data_entities == None):
        raise Exception('Impossible to retrieve data for entities.')
    else:
        print('%d entities retrieved.\n' % (len(raw_data_entities)))

    ## Delete relation between the two samples if exists
    delete_relations(prefix + 'echantillon_9477', prefix + 'echantillon_10423', raw_data_relations)

    ## Create graph
    data_manager = DataManager(raw_data_entities, raw_data_relations)
    graph = data_manager.construct_graph()
    graph.to(device)
    print(graph)

    rgcn = BasicRGCN(in_features=2, out_features=2, relations_count=2).to(device)
    print(rgcn)

    model.eval()
    with torch.no_grad():
        pred = model(graph)
        # print(pred)
        relation_index, index_1, index_2 = data_manager.get_relation_entities_index('estProcheDe', prefix + 'echantillon_9477', prefix + 'echantillon_10423')

        print('Link prediction : ', pred[relation_index][index_1][index_2].item())