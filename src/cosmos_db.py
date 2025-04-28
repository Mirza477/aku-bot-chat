# src/cosmos_db.py
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from config.settings import COSMOS_URI, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER

client = CosmosClient(url=COSMOS_URI, credential=COSMOS_KEY)
database = client.get_database_client(COSMOS_DATABASE)
container = database.get_container_client(COSMOS_CONTAINER)

def upsert_policy_section(item):
    """
    Upserts a policy section or FAQ item into the Cosmos DB container.
    """
    container.upsert_item(item)


def create_cosmos_container():
    """
    Creates the Cosmos DB container with vector indexing if it doesn't exist.
    Note: Adjust the 'dimensions' field if you use a different embedding model.
    """
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/vector",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 1536  # For text-embedding-ada-002
            }
        ]
    }
    indexing_policy = {
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": "/\"_etag\"/?"}],
        "vectorIndexes": [{"path": "/vector", "type": "flat"}]
    }
    try:
        container_resp = database.create_container_if_not_exists(
            id=COSMOS_CONTAINER,
            partition_key=PartitionKey(path="/id"),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
            offer_throughput=400
        )
        print(f"Container {COSMOS_CONTAINER} created or already exists.", flush=True)
    except exceptions.CosmosHttpResponseError as e:
        print("Error creating container:", e, flush=True)

def query_vector_search(query_vector, top_k=3):
    """
    Executes a vector search against Cosmos DB using the provided query vector.
    """
    query = """
    SELECT TOP @top_k c.id, c.document_name, c.section, c.content,
           VectorDistance(c.vector, @embedding) as score
    FROM c
    ORDER BY VectorDistance(c.vector, @embedding)
    """
    parameters = [
        {"name": "@top_k", "value": top_k},
        {"name": "@embedding", "value": query_vector}
    ]
    results = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))
    return results
