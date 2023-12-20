# Define DataPreprocessor class to preprocess the dataset
class DataPreprocessor:

    def __init__(self, text_splitter):
        self.text_splitter = text_splitter

    def preprocessing_dataset(self, articles):
        # Split articles into documents
        docs = self.text_splitter.create_documents(articles[:100])
        return docs
