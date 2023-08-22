from SPARQLWrapper import SPARQLWrapper, JSON
from config.config import features, network
import inspect

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

        try:
            ret = self.sparql.queryAndConvert()
            triplets = []

            for r in ret['results']['bindings']:
                triplets.append((r['s']['value'], relation_name, r['o']['value'], 1))

            return triplets
        
        except Exception as e:
                print(e)

    def count_entities (self, class_name):
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

    def load_entities (self, class_name, limit=None):
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
            entities = []

            for r in ret['results']['bindings']:
                entities.append(r['i']['value'])

            return entities

        except Exception as e:
            print(e)
    
    def load_sample_by_drug_type(self, drug_type, limit=None):
        query = '''
                    PREFIX : <%s>
                    SELECT ?e ''' %(self.prefix)

        for feature in features['Echantillon']:
            query += '?%s ' % (feature)

        query +='''
                    WHERE { 
                        ?e a :Echantillon    .
                        ?e :typeDrogue '%s'    .''' % (drug_type)
        
        for feature in features['Echantillon']:
            query += '''
                        ?e :%s ?%s    .''' % (feature, feature)

        query += '''
                    }'''
        
        if limit != None:
            query += 'LIMIT %s' % (limit)

        self.sparql.setQuery(query)

        try:
            ret = self.sparql.queryAndConvert()
            entities = {}

            for r in ret['results']['bindings']:
                entity_name = r['e']['value']
                entities[entity_name] = {}
                for feature in features['Echantillon']:
                    entities[entity_name][feature] = float(r[feature]['value'].replace(',', '.'))

            return entities
        except Exception as e:
            print(e)

    def load_sample_neighborhood (self, entity_name, features):
        query_relations = '''
                PREFIX : <%s>

                SELECT *
                WHERE { 
                    ?s :estProcheDe ?o
                    VALUES ?s {:%s}	.
                    FILTER(?s != ?o)
                }
                ''' % (self.prefix, entity_name)
        
        self.sparql.setQuery(query_relations)
        
        triplets = {}
        triplets['estProcheDe'] = []
        entities = {}
        entities[self.prefix + entity_name] = {}
        try:
            ret = self.sparql.queryAndConvert()

            for r in ret['results']['bindings']:
                triplets['estProcheDe'].append((r['s']['value'], 'estProcheDe', r['o']['value'], 1))
                entities[r['o']['value']] = {}

        except Exception as e:
            print(e)

        query_entities = '''
                    PREFIX : <%s>
                    SELECT ?e ''' %(self.prefix)

        for feature in features:
            query_entities += '?%s ' % (feature)

        query_entities +='''
                    WHERE { 
                        ?e a :Echantillon    .'''
        
        for feature in features:
            query_entities += '''
                        ?e :%s ?%s    .''' % (feature, feature)

        query_entities += '''
                    VALUES(?e){'''
        
        for entities_name, _ in entities.items():
            query_entities += '(<%s>)' % (entities_name)
        
        query_entities += '}}'

        self.sparql.setQuery(query_entities)
        try:
            ret = self.sparql.queryAndConvert()

            for r in ret['results']['bindings']:
                entity_name = r['e']['value']
                for feature in features:
                        entities[entity_name][feature]= float(r[feature]['value'].replace(',', '.'))

        except Exception as e:
            print('Error : %s' % (e))

        return triplets, entities