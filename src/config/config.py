import yaml
from pprint import pprint

with open("config/config.yml", "r") as stream:
    try:
        config = yaml.safe_load(stream)

        entities_list = config['entities'][0]
        drug_types_list = config['drug_type'][0]
        relations_list = config['relations']
        target_relations_list = config['target_relations'][0]
        network = config['network'][0]
        
        if 'entities_limit' not in config:
            entities_limit = None
        else:
            entities_limit = config['entities_limit']
        
        print('###### Config ######') 
        print('    Entities list : ' + entities_list)
        print('    Drug types : ' + drug_types_list)
        print('    Traget realtions : ' + target_relations_list)
        print('    Network configuration : ')
        pprint(network)
        print('####################\r\n') 
    except yaml.YAMLError as exc:
        print(exc)