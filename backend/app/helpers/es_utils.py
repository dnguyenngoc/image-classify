from elasticsearch import Elasticsearch


def create_elasticsearch_index(es: Elasticsearch, index_name: str, len_dim: int):
    try:
        body = {
            'mappings': {
                'properties': {
                    'id': {'type': 'integer'},
                    'name': {'type': 'text'},
                    'image': {'type': 'text'},
                    'vector': {
                        'type': 'dense_vector',
                        'dims': len_dim
                    }
                }
            }
        }
        es.indices.create(index = index_name, body = body)
    except Exception as e:
        print('[warning] index is exists! ', e)



def create_elasticsearch_datasets(es: Elasticsearch, index_name: str, id: int, name: str, image: str, feature_maps_list: list, index: int):
    try:
        doc = {
            'id': id, 
            'name': name, 
            'image': image,
            'vector': feature_maps_list,
        }
        es.create(index_name, body = doc, id = index)

    except Exception as e:
        print('[error] can not create es data! ', e)




def delete_elasticsearch_index(es: Elasticsearch, index_name, ):
    es.indices.delete(index=index_name, ignore=[400, 404])


def delete_elasticsearch_doc_by_class_id(es: Elasticsearch, index_name, class_id):
    query = {
        'query': {
            "match": {
                "id": class_id
            },
        }
    }
    res = es.search(index=index_name, body=query)
    end = []
    for item in res['hits']['hits']:
        _id = item['_id']
        es.delete(index=index_name,id=_id)
        end.append(_id)
    return end
    

def get_elasticsearch_doc_by_class_id(es: Elasticsearch, index_name, class_id):
    query = {
        'query': {
            "match": {
                "id": class_id
            },
        }
    }
    res = es.search(index=index_name, body=query)
    return res



def matching_elasticsearch_index(es: Elasticsearch, index_name: str, vector: list):
    query = {
        'size': 3,
        'query': {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'vector') + 1",
                    "params":{
                        "queryVector": vector
                    }
                }
            }
        }
    }
    res = es.search(index= index_name, body = query)
    return res