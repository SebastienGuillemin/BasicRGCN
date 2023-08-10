import yaml
import sys

with open("config/config.yml", "r") as stream:
    try:
        config = yaml.safe_load(stream)

        entities_list = config['entities']
        drug_types_list = config['drug_type']
        relations_list = config['relations']
        target_relations_list = config['target_relations']
        
        if 'entities_limit' not in config:
            entities_limit = None
        else:
            entities_limit = config['entities_limit']
    except yaml.YAMLError as exc:
        print(exc)