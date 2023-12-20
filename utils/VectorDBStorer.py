# Import necessary modules
from utils.DatasetLoader import DatasetLoader
from utils.DataPreprocessor import DataPreprocessor

class VectorDBStorer:
    def __init__(self, client, collection_name, text_splitter, dataset_name):
        self.client = client
        self.collection_name = collection_name
        self.text_splitter = text_splitter
        self.dataset_name = dataset_name

    def store_Vectordb(self):
        # Try to store Vectordb
        try:
            collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' exists!")

            if len(collection.get()['documents']) == 0:
                # Load articles from dataset
                articles = DatasetLoader(dataset_name=self.dataset_name).load_scientific_dataset()
                # Preprocess articles
                docs = DataPreprocessor(self.text_splitter).preprocessing_dataset(articles)
                # Add documents to collection
                collection.add(
                    ids=[str(i) for i in range(0, 100)],
                    documents=[str(i.page_content) for i in docs[:100]],
                    metadatas=[{"type": "support"} for _ in range(0, 100)],
                )
        except ValueError:
            print(f"Collection '{self.collection_name}' does not exist.")

        return collection

    def get_collection(self):
        return self.store_Vectordb()
