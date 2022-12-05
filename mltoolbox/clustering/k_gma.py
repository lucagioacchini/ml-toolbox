# type: ignore

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import community as community_louvain
import networkx as nx
import pandas as pd
import numpy as np
import joblib

class kGMA():
    """_summary_

        Parameters
        ----------
        model_path : _type_, optional
            _description_, by default None
        n_neighbors : int, optional
            _description_, by default 3
        metric : str, optional
            _description_, by default 'cosine'
        _load_model : bool, optional
            _description_, by default False
    """
    def __init__(self, model_path=None, n_neighbors=3, metric='cosine', 
                 _load_model=False):
        self.model_path=model_path
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.scaler = StandardScaler()
        self.graph = None
                    
        if _load_model:
            self.graph = nx.read_gexf(f'{self.model_path}_graph.gexf')
            self.scaler = joblib.load(f'{self.model_path}_kgma_scaler.save')
    
    def fit(self, X, scale_data=True, node_names=None):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_
        scale_data : bool, optional
            _description_, by default True
        node_names : _type_, optional
            _description_, by default None
        """
        node_names = X.index
        X = X.to_numpy()
        if scale_data:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        adj_maxtrix = kneighbors_graph(X, n_neighbors=self.n_neighbors, 
                            mode='distance', metric=self.metric, n_jobs=-1)\
                            .toarray()
        adj_maxtrix = pd.DataFrame(adj_maxtrix, index=node_names, 
                                   columns=node_names)
        self.graph = nx.from_pandas_adjacency(adj_maxtrix)
        
    def predict(self, X, cluster_name='cluster', save=False):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_
        cluster_name : str, optional
            _description_, by default 'cluster'
        save : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        node_names = X.index
        clusters = community_louvain.best_partition(self.graph, 
                                                    random_state=15)
        clusters = {k:v for k,v in clusters.items() if k in node_names}
        nx.set_node_attributes(self.graph, clusters, name=cluster_name)
        clusters = pd.DataFrame(clusters.items(), 
                                index=clusters.keys(), 
                                columns=['node', cluster_name])\
                     .set_index('node').reindex(X.index).to_numpy()
        clusters = np.ravel(clusters)

        if save:
            nx.write_gexf(self.graph, f'{self.model_path}_graph.gexf')
            joblib.dump(self.scaler, f'{self.model_path}_kgma_scaler.save')
            
        return clusters