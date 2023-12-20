# Import necessary modules
from datasets import load_dataset

# Define DatasetLoader class to load the scientific dataset
class DatasetLoader:
    def __init__(self, dataset_name, dataset_type = 'pubmed') -> None:
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        pass

    def load_scientific_dataset(self):
        # loading pubmed scientific dataset
        dataset = load_dataset(self.dataset_name, self.dataset_type)
        articles = dataset['train']['article']
        return articles
