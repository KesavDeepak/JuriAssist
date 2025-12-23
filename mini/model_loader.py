import pickle
import torch
import pandas as pd

class JudgementModel:
    def __init__(self, model_path='trained_model.pkl', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.kmeans = None
        self.cluster_labels = None
        self.data = None
        self.embeddings = None
        self.load_model()

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.model = saved_data['model'].to(self.device)
        self.kmeans = saved_data['kmeans']
        self.cluster_labels = saved_data['cluster_labels']
        self.data = saved_data['data']
        self.embeddings = saved_data['embeddings']
        print("✅ Model and data loaded successfully!")

    def predict_cluster(self, text):
        embedding = self.model.encode([text], convert_to_tensor=True, device=self.device)
        embedding = embedding.cpu().numpy()  # KMeans expects numpy
        cluster_id = self.kmeans.predict(embedding)[0]
        label = self.cluster_labels.get(cluster_id, "Unknown")
        return cluster_id, label

    def get_cluster_examples(self, cluster_id, n=5):
        cluster_data = self.data[self.data['cluster'] == cluster_id]
        return cluster_data.head(n).to_dict(orient='records')
