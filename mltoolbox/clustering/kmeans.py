# type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import numpy


class kMeans():
    """A k-Means clustering implementation with data preprocessing and model persistence.

    Parameters:
        n_clusters (int, optional): Number of clusters to form. Defaults to 8.
        model_path (str, optional): Path to save/load model files. Defaults to None.
        _load_model (bool, optional): Whether to load an existing model from model_path. Defaults to False.
        init (str, optional): Method for initialization of centroids. Defaults to 'k-means++'.
        n_init (int, optional): Number of times to run k-means with different centroid seeds. Defaults to 10.
        max_iter (int, optional): Maximum number of iterations for a single run. Defaults to 300.
        tol (float, optional): Relative tolerance for convergence. Defaults to 0.0001.
        random_state (int, optional): Seed for random number generation. Defaults to None.
        algorithm (str, optional): K-means algorithm to use. Defaults to 'lloyd'.

    Attributes:
        model_path (str): Path for model saving/loading
        scaler (StandardScaler): Scales input features
        model (KMeans): The underlying scikit-learn KMeans clusterer
        X (array): Stored training features

    Methods:
        _scale_data(X_train, X_val): Scales training and validation features
        fit(X, scale_data, save): Trains the clustering model on provided data
        predict(X, scale_data): Predicts cluster labels for input features

    Example:
        >>> # Generate 20 (resp. 5) training (resp. validation) samples with 4 features 
        >>> X = np.random.random((20, 4))
        >>> from mltoolbox.clustering import kMeans
        >>> # Initialize the clustering algorithm
        >>> kmeans = kMeans(n_clusters=3)
        >>> # Fit the algorithm
        >>> kmeans.fit(X, scale_data=True)
        >>> # Get the clusters
        >>> kmeans.predict(X)

        >>> array([0, 1, 2, 2, 2, 1, ...])
    """

    def __init__(self, n_clusters: int = 8, model_path: str = None, _load_model: bool = False,
                 init: str = 'k-means++', n_init: int = 10, max_iter: int = 300, tol: float = 0.0001,
                 random_state: int = None, algorithm: str = 'lloyd'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters, init=init, n_init=n_init,
                            max_iter=max_iter, tol=tol, algorithm=algorithm,
                            random_state=random_state)

        if _load_model:
            self.scaler = joblib.load(f'{self.model_path}_kmeans_scaler.save')
            self.X = joblib.load(f'{self.model_path}_kmeans.save')

    def _scale_data(self, X_train: numpy.array, X_val: numpy.array = None):
        """Scale training and validation data using StandardScaler.

        Parameters:
            X_train (numpy.array): Training feature data
            X_val (numpy.array, optional): Validation feature data. Defaults to None.

        Returns:
            tuple: (scaled_X_train, scaled_X_val) - Scaled feature arrays
        """
        # Fit the scaler on training data
        self.scaler.fit(X_train)
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val) != type(None):
            X_val = self.scaler.transform(X_val)

        return X_train, X_val

    def fit(self, X: numpy.array, scale_data: bool = True, save: bool = False):
        """Train the k-Means clustering model on provided data.

        Parameters:
            X (numpy.array): Training feature data
            scale_data (bool, optional): Whether to scale input features. Defaults to True.
            save (bool, optional): Whether to save model and scaler to disk. Defaults to False.
        """
        self.X = X
        # Data standardization
        if scale_data:
            X, _ = self._scale_data(X)

        # Train the classifier
        self.model.fit(X)
        if save:
            joblib.dump(X, f'{self.model_path}_kmeans.save')
            joblib.dump(self.scaler, f'{self.model_path}_kmeans_scaler.save')

    def predict(self, X: numpy.array, scale_data: bool = True):
        """Predict cluster labels for input features.

        Parameters:
            X (numpy.array): Input feature array
            scale_data (bool, optional): Whether to scale input features. Defaults to True.

        Returns:
            numpy.array: Predicted cluster labels for input samples
        """
        if scale_data:
            X = self.scaler.transform(X)
        y_pred = self.model.predict(X)

        return y_pred
