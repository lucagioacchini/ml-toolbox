# type: ignore

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import community as community_louvain
import networkx as nx
import pandas as pd
import numpy as np
import numpy
import joblib


class kGMA():
    """A k-nearest neighbors graph-based Louvain algorithm.

    This class implements a clustering approach that combines k-nearest neighbors graph
    construction with the Louvain community detection algorithm to identify clusters in data.

    Parameters:
        model_path (str, optional): Path to save/load model files. Defaults to None.
        n_neighbors (int, optional): Number of nearest neighbors for graph construction. Defaults to 3.
        metric (str, optional): Distance metric used for neighbor calculation. Defaults to 'cosine'.
        _load_model (bool, optional): Whether to load an existing model from model_path. Defaults to False.

    Attributes:
        model_path (str): Path for model saving/loading
        n_neighbors (int): Number of neighbors for graph construction
        metric (str): Distance metric for neighbor calculation
        scaler (StandardScaler): Scales input features
        graph (networkx.Graph): The constructed k-nearest neighbors graph

    Methods:
        fit(X, scale_data, node_names): Constructs the k-nearest neighbors graph from input data
        predict(X, cluster_name, save): Performs community detection on the graph

    Example:
        >>> # Generate 20 samples with 4 features 
        >>> X = pd.DataFrame(np.random.random((20, 4)))
        >>> from mltoolbox.clustering import kGMA
        >>> kgma = kGMA(n_neighbors=3, metric='cosine')
        >>> # Build the k-NN-graph and fit the algorithm
        >>> kgma.fit(X, scale_data=True)
        >>> # Get the clusters
        >>> kgma.predict(X)

        >>> array([1, 1, 2, 2, 3, 0, 3, ...])
    """

    def __init__(self, model_path: str = None, n_neighbors: int = 3, metric: str = 'cosine',
                 _load_model: bool = False):
        self.model_path = model_path
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.scaler = StandardScaler()
        self.graph = None

        if _load_model:
            self.graph = nx.read_gexf(f'{self.model_path}_graph.gexf')
            self.scaler = joblib.load(f'{self.model_path}_kgma_scaler.save')

    def fit(self, X: numpy.array, scale_data: bool = True, node_names: list = None):
        """Construct k-nearest neighbors graph from input data.

        Parameters:
            X (numpy.array): Input feature DataFrame with index as node names
            scale_data (bool, optional): Whether to scale input features. Defaults to True.
            node_names (list, optional): Custom names for nodes. If None, uses X.index. Defaults to None.

        Notes:
            - Constructs a graph where nodes are samples and edges connect k-nearest neighbors
            - Edge weights are based on the distance metric specified in initialization
            - The graph is stored in self.graph as a networkx Graph object
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

    def predict(self, X: numpy.array, cluster_name: str = 'cluster', save: bool = False):
        """Perform community detection on the constructed graph using the Louvain algorithm.

        Parameters:
            X (numpy.array): Input DataFrame whose index defines the subset of nodes to cluster
            cluster_name (str, optional): Name of the cluster attribute in the graph. Defaults to 'cluster'.
            save (bool, optional): Whether to save the graph and scaler to disk. Defaults to False.

        Returns:
            numpy.array: Cluster assignments for each node in X.index

        Notes:
            - Uses the Louvain community detection algorithm with a fixed random state
            - Cluster assignments are stored as node attributes in the graph
            - When save=True, saves both the graph in GEXF format and the scaler
        """
        node_names = X.index
        clusters = community_louvain.best_partition(self.graph,
                                                    random_state=15)
        clusters = {k: v for k, v in clusters.items() if k in node_names}
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
