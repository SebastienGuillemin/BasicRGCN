from SPARQLWrapper import SPARQLWrapper, JSON, RDFXML
from config.config import features, network
from rdflib import Graph

prefix = 'http://www.stups.fr/ontologies/2023/stups/'

class Repository () :
    def __init__(self) :
        self.sparql = SPARQLWrapper('http://%s:%s/repositories/STUPS' % (network['graph_db_host'], network['graph_db_port']))
        self.sparql.setReturnFormat(JSON)
        

    def query_relations_triples(self, relations_names, entities, limit=None) :
        triples = {}

        for relation_name in relations_names:
            relations_triples = self._query_relation_triples(relation_name, entities, limit)

            if (relations_triples != None):
                triples[relation_name] = relations_triples

        return triples

    def _query_relation_triples(self, relation_name, entities, limit=None) :
        query = '''
                PREFIX : <%s>
                SELECT * WHERE {
                    ?s :%s ?o .
                    FILTER(?s != ?o)
                }
                ''' % (prefix, relation_name)
        
        if limit != None:
            query += 'LIMIT %s' % (limit)
        
        self.sparql.setQuery(query)

        try:
            ret = self.sparql.queryAndConvert()
            triples = []

            for r in ret['results']['bindings']:
                entity_1 = r['s']['value']
                entity_2 = r['o']['value']

                if entity_1 in entities and entity_2 in entities:
                    triples.append(((r['s']['value'], relation_name, r['o']['value']), 1))

            return triples
        
        except Exception as e:
                print(e)

    def count_entities (self, class_name):
        query = '''
                PREFIX : <%s>

                SELECT (COUNT(?e) as ?cpt)
                WHERE { 
                    ?e a :%s
                }
                ''' % (prefix, class_name)
    
        self.sparql.setQuery(query)

        try:
            ret = self.sparql.queryAndConvert()
            return ret["results"]["bindings"][0]["cpt"]["value"]

        except Exception as e:
            print(e)

    def query_entities (self, class_name, limit=None):
        query = '''
                PREFIX : <%s>

                SELECT ?i
                WHERE { 
                    ?i a :%s
                }
                ''' % (prefix, class_name)
        
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
    
    def query_sample_by_drug_type(self, drug_type, limit=None):
        query = '''
                    PREFIX : <%s>
                    SELECT ?e ''' %(prefix)

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

    def query_sample_neighborhood (self, entity_name, features):
        self.sparql.setReturnFormat(RDFXML)
        query_relations = '''
                PREFIX : <%s>

                CONSTRUCT {
                    ?s1 ?p1 ?o1 .
                    ?s2 ?p1 ?o2 .
                }
                WHERE {
                    ?s1 ?p1 ?o1	.
                    FILTER (?s1 = :%s && ?p1 = :estProcheDe && ?s1 != ?o1)	.
                    BIND (?o1 as ?s2)	.
                    ?s2 ?p1 ?o2	.
                    FILTER (?o2 != ?s2)	.
                }
                ''' % (prefix, entity_name)
        
        self.sparql.setQuery(query_relations)
        
        triples = {}
        triples['estProcheDe'] = []
        entities = {}
        ret = None

        try:
            ret = self.sparql.queryAndConvert()
        except Exception as e:
            print('Error : %s' % (e))
            print(query_relations)
            exit()

        for t in ret.triples((None,None,None)):
            entity_1 = t[0].n3()
            entity_2 = t[2].n3()

            ## Removing '<' and '>' characters.
            entity_1 = entity_1[1:len(entity_1) - 1]
            entity_2 = entity_2[1:len(entity_2) - 1]

            triples['estProcheDe'].append((entity_1, 'estProcheDe', entity_2, 1))
            if entity_1 not in entities:
                entities[entity_1] = {}

        self.sparql.setReturnFormat(JSON)
        query_entities = '''
                    PREFIX : <%s>
                    SELECT ?e ''' %(prefix)

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

        ret = None
        try:
            ret = self.sparql.queryAndConvert()
        except Exception as e:
            print('Error : %s' % (e))
            print(query_entities)
            exit()

        for r in ret['results']['bindings']:
            entity_name = r['e']['value']
            for feature in features:
                    entities[entity_name][feature]= float(r[feature]['value'].replace(',', '.'))

        return triples, entities