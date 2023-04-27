from random import random

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    db,
)
from pymilvus.orm import utility

_HOST = '127.0.0.1'
_PORT = '19530'
_ROOT = "root"
_ROOT_PASSWORD = "Milvus"
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3

# Vector parameters
_DIM = 128
_INDEX_FILE_SIZE = 32  # max file size of stored index


def connect_to_milvus(db_name="default"):
    print(f"connect to milvus\n")
    connections.connect(host=_HOST,
                        port=_PORT,
                        user=_ROOT,
                        password=_ROOT_PASSWORD,
                        db_name=db_name,
                        )


def create_collection(collection_name):
    default_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="double", dtype=DataType.DOUBLE),
        FieldSchema(name="fv", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    default_schema = CollectionSchema(fields=default_fields)
    print(f"Create collection:{collection_name}")
    return Collection(name=collection_name, schema=default_schema)


def insert(collection, num, dim):
    data = [
        [i for i in range(num)],
        [[random.random() for _ in range(dim)] for _ in range(num)],
    ]
    collection.insert(data)
    return data[1]


def drop_index(collection):
    collection.drop_index()
    print("\nDrop index sucessfully")

def search(collection, vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "expr": "id_field >= 0"}
    results = collection.search(**search_param)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))


def collection_read_write(collection):
    # insert 10000 vectors with 128 dimension
    vectors = insert(collection, 10000, _DIM)
    collection.flush()
    print("\nThe number of entity:", collection.num_entities)

    # create index
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index("fv", index_param)
    print("\nCreated index:\n{}".format(collection.index().params))

    # load data to memory
    collection.load()
    # search
    search(collection, "fv", "id", vectors[:3])
    # release memory
    collection.release()
    # drop collection index
    collection.drop_index()


if __name__ == '__main__':
    # connect to milvus and using database db1
    # there will not check db1 already exists during connect
    connect_to_milvus(db_name="default")

    # create collection within default
    col1_db1 = create_collection("col1_db1")

    # create db1
    db.create_database(db_name="db1")

    # use database db1
    db.using_database(db_name="db1")
    # create collection within default
    col2_db1 = create_collection("col1_db1")
    col2_db2 = create_collection("col1_db2")

    # verify read and write
    collection_read_write(col2_db2)

    # list collections within db1
    print("\nlist collections of database db1:")
    print(utility.list_collections())

    print("\ndrop collection: col1_db2")
    col2_db1.drop()
    col2_db2.drop()

    print("\ndrop database db1:")
    db.drop_database(db_name="db4", using="con2")

    # list database
    print("\nlist databases:")
    print(db.list_database(using="con2"))