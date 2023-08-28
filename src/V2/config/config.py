import yaml
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path + "/config.yml", "r") as stream:
    try:
        config = yaml.safe_load(stream)

        entities = config['entities']
        drug_type = config['drug_type']
        relations_list = config['relations']
        target_relations_list = config['target_relations'][0]

        network = {}
        network['graph_db_host'] = config['network']['graph_db_host']
        network['graph_db_port'] = config['network']['graph_db_port']

        features = {}

        for entity in entities:
            features[entity] = config['features'][entity]


        if 'entities_limit' not in config:
            entities_limit = None
        else:
            entities_limit = config['entities_limit']
        
        print('################## Config ##################') 
        print('Entitie(s) name(s) and feature(s) : ')
        
        for entity, features_list in features.items():
            print('    - %s : %s' % (entity, features_list))

        print('\nDrug type : ' + drug_type)
        
        print('\nTraget relation(s) : ' + target_relations_list)
        
        print('\nNetwork configuration : ')
        for parameter, value in network.items():
            print('    -%s : %s' % (parameter, value))
        
        print('############################################\n') 
    except yaml.YAMLError as exc:
        print(exc)