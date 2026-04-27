import json
import logging
import chromadb
import os
from tqdm import tqdm
from omegaconf import DictConfig
from hydra import main as hydra_main


def ensure_embeddings_exist(embedding_metadata_file_path, embedding_detail_dir_path):
    """Check if all embedding files exist based on metadata"""
    with open(embedding_metadata_file_path, "r") as file:
        metadata_list = json.load(file)
    for metadata in metadata_list:
        embedding_file_path = os.path.join(
            embedding_detail_dir_path, metadata["id"] + ".json"
        )
        if not os.path.exists(embedding_file_path):
            print(f"Embedding file does not exist: {embedding_file_path}")
            return False
    return True


def list_collections(client):
    """List all collections in the database with their metadata and count"""
    collections = client.list_collections()
    print("All collections in the database:")
    for collection in collections:
        print(f"- Name: {collection.name}")
        print(f"  Metadata: {getattr(collection, 'metadata', None)}")
        print(
            f"  Embedding function: {getattr(collection, 'embedding_function', None)}"
        )
        try:
            col_obj = client.get_collection(collection.name)
            count = col_obj.count()
            print(f"  Item count: {count}")
        except Exception as e:
            print(f"  Failed to get item count: {e}")
        print("-----------------------------")

    return collections


def create_collection(
    client, collection_name, embedding_metadata_file_path, embedding_detail_dir_path
):
    """Create a new collection and add embeddings from files"""
    ensure_outcome = ensure_embeddings_exist(
        embedding_metadata_file_path, embedding_detail_dir_path
    )
    if not ensure_outcome:
        print(f"Embedding files do not exist: {embedding_metadata_file_path}")
        return

    collections = list_collections(client)

    if collection_name in [c.name for c in collections]:
        client.delete_collection(name=collection_name)
        print(f"Deleted collection: {collection_name}")

    collection = client.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")

    id_list = []
    embedding_list = []
    metadata_list = []
    with open(embedding_metadata_file_path, "r") as file:
        metadata_list = json.load(file)

    for metadata in tqdm(metadata_list, desc="Reading embedding files"):
        embedding_file_path = os.path.join(
            embedding_detail_dir_path, metadata["id"] + ".json"
        )

        id_list.append(metadata["id"])
        embedding_list.append(json.load(open(embedding_file_path, "r")))

    batch_size = 1000
    total = len(id_list)
    if id_list:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_id_list = id_list[start:end]
            batch_embedding_list = embedding_list[start:end]
            batch_metadata_list = metadata_list[start:end]
            collection.add(
                ids=batch_id_list,
                embeddings=batch_embedding_list,
                metadatas=batch_metadata_list,
            )
            print(f"Added {end} / {total} vectors to collection")
        print(f"Successfully added {total} vectors to collection")
    else:
        print("No valid vectors to add")


def query_test_en(
    client,
    collection_name,
    embedding_detail_dir_path,
):
    """Test query functionality with sample embeddings"""
    query_id_list = [
        "en-skill-point-0001",
        "en-skill-point-0002",
        "en-skill-point-0003",
        "en-skill-point-0004",
        "en-skill-point-0005"
    ]
    collection = client.get_collection(name=collection_name)

    for query_id in query_id_list:
        query_embedding = json.load(
            open(os.path.join(embedding_detail_dir_path, query_id + ".json"), "r")
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["distances"],
        )

        filtered_results = [
            result for result in results["distances"][0] if result <= 0.7
        ]

        result_ids = results["ids"][0]
        print(f"Query ID: {query_id}")
        print(f"Matched ID list: {result_ids}")
        print(f"Match distances: {results['distances'][0]}")
        print(f"Results with distance <= 0.7: {filtered_results}")
        print("--------------------------------")


def query_test_cn(
    client,
    collection_name,
    embedding_detail_dir_path,
):
    """Test query functionality with Chinese sample embeddings"""
    query_id_list = [
        "cn-skill-point-0001",
        "cn-skill-point-0002",
        "cn-skill-point-0003",
        "cn-skill-point-0004",
        "cn-skill-point-0005"
    ]
    collection = client.get_collection(name=collection_name)

    for query_id in query_id_list:
        query_embedding = json.load(
            open(os.path.join(embedding_detail_dir_path, query_id + ".json"), "r")
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["distances"],
        )

        filtered_results = [
            result for result in results["distances"][0] if result <= 0.7
        ]

        result_ids = results["ids"][0]
        print(f"Query ID: {query_id}")
        print(f"Matched ID list: {result_ids}")
        print(f"Match distances: {results['distances'][0]}")
        print(f"Results with distance <= 0.7: {filtered_results}")
        print("--------------------------------")


def delete_all_collections(client):
    """Delete all collections in the database"""
    collections = client.list_collections()
    if not collections:
        print("No collections found to delete.")
        return
    
    print(f"Found {len(collections)} collections to delete:")
    for collection in collections:
        print(f"- {collection.name}")
    
    # Delete each collection
    for collection in collections:
        try:
            client.delete_collection(name=collection.name)
            print(f"Successfully deleted collection: {collection.name}")
        except Exception as e:
            print(f"Failed to delete collection {collection.name}: {e}")
    
    print("All collections have been deleted.")


@hydra_main(config_path="config", config_name="chroma", version_base=None)
def main(config: DictConfig):
    """Main entry function that receives the Hydra-parsed configuration object"""
    # Set logging level for httpx and chromadb to WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    # Get action parameter passed through Hydra override
    action = getattr(config, 'action', None)
    
    if action is None:
        print("Usage: python chroma.py action=<action>")
        print("Actions: list, create-en, create-cn, test-en, test-cn, delete-all")
        print("Example: python chroma.py action=list")
        return
    
    # Initialize ChromaDB client
    client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
    
    if action == 'list':
        print("Listing all collections...")
        list_collections(client)
        
    elif action == 'create-en':
        print("Creating English collection...")
        create_collection(
            client, 
            config.chroma_collection_name_en, 
            config.chroma_collection_meta_data_path_en, 
            config.chroma_collection_embedding_dir
        )
        
    elif action == 'create-cn':
        print("Creating Chinese collection...")
        create_collection(
            client, 
            config.chroma_collection_name_cn, 
            config.chroma_collection_meta_data_path_cn, 
            config.chroma_collection_embedding_dir
        )
        
    elif action == 'test-en':
        print("Testing English collection queries...")
        query_test_en(
            client,
            config.chroma_collection_name_en,
            config.chroma_collection_embedding_dir,
        )
        
    elif action == 'test-cn':
        print("Testing Chinese collection queries...")
        query_test_cn(
            client,
            config.chroma_collection_name_cn,
            config.chroma_collection_embedding_dir,
        )
        
    elif action == 'delete-all':
        print("Deleting all collections...")
        delete_all_collections(client)
    else:
        print(f"Unknown action: {action}")
        print("Available actions: list, create-en, create-cn, test-en, test-cn, delete-all")


if __name__ == "__main__":
    main()
