from SPARQLWrapper import SPARQLWrapper, JSON
from config.config import *

class Dataloader () :
    def __init__(self) :
        self.sparql = SPARQLWrapper('http://%s:7200/repositories/STUPS' % (network['graph_db_host']))
        self.sparql.setReturnFormat(JSON)
        
        self.prefix = 'http://www.stups.fr/ontologies/2023/stups/'

    def load_relations_triplets(self, relations_names, limit=None) :
        triplets = {}

        for relation_name in relations_names:
            realtion_triplets = self._load_relation_triplets(relation_name, limit)

            if (realtion_triplets != None):
                triplets[relation_name] = realtion_triplets

        return triplets

    def _load_relation_triplets(self, relation_name, limit=None) :
        query = '''
                PREFIX : <%s>
                SELECT * WHERE {
                    ?s :%s ?o .
                    FILTER(?s != ?o)
                }
                ''' % (self.prefix, relation_name)
        
        if limit != None:
            query += 'LIMIT %s' % (limit)
        
        self.sparql.setQuery(query)
        
        # print(query)

        try:
            ret = self.sparql.queryAndConvert()
            triplets = []

            for r in ret['results']['bindings']:
                triplets.append((r['s']['value'], relation_name, r['o']['value'], 1))

            return triplets
        
        except Exception as e:
                print(e)

    def count_instances (self, class_name):
        query = '''
                PREFIX : <%s>

                SELECT (COUNT(?e) as ?cpt)
                WHERE { 
                    ?e a :%s
                }
                ''' % (self.prefix, class_name)
    
        self.sparql.setQuery(query)

        try:
            ret = self.sparql.queryAndConvert()
            return ret["results"]["bindings"][0]["cpt"]["value"]

        except Exception as e:
            print(e)

    def load_instances (self, class_name, limit=None):
        query = '''
                PREFIX : <%s>

                SELECT ?i
                WHERE { 
                    ?i a :%s
                }
                ''' % (self.prefix, class_name)
        
        if limit != None:
            query += 'LIMIT %s' % (limit)
    
        self.sparql.setQuery(query)

        try:
            ret = self.sparql.queryAndConvert()
            instances = []

            for r in ret['results']['bindings']:
                instances.append(r['i']['value'])

            return instances

        except Exception as e:
            print(e)
    
    def load_sample_by_drug_types(self, drug_types, limit=None):
        instances = []

        for drug_type in drug_types:
            instances += self._load_sample_by_drug_type(drug_type, limit)
        
        return instances


    def _load_sample_by_drug_type (self, drug_type, limit=None):
        query = '''
                PREFIX : <%s>

                SELECT ?e
                WHERE { 
                    ?e a :Echantillon .
                    ?e :typeDrogue '%s'
                }
                ''' % (self.prefix, drug_type)
        
        if limit != None:
            query += 'LIMIT %s' % (limit)

        self.sparql.setQuery(query)

        try:
            ret = self.sparql.queryAndConvert()
            instances = []

            for r in ret['results']['bindings']:
                instances.append(r['e']['value'],)

            return instances

        except Exception as e:
            print(e)