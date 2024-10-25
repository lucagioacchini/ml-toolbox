from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import numpy
import joblib


class KnnClassifier():
    """A k-Nearest Neighbors classifier with data preprocessing and voting capabilities.

    Parameters:
        n_neighbors (int, optional): Number of neighbors to use for prediction. Defaults to 7.
        model_path (str, optional): Path to save/load model files. Defaults to None.
        metric (str, optional): Distance metric used for neighbor calculation. Defaults to 'cosine'.
        _load_model (bool, optional): Whether to load an existing model from model_path. Defaults to False.

    Attributes:
        model_path (str): Path for model saving/loading
        scaler (StandardScaler): Scales input features
        neighbors (array): Indices of nearest neighbors for each sample
        y_train (array): Training labels
        model (KNeighborsClassifier): The underlying scikit-learn KNN classifier
        X (array): Stored training features
        y (array): Stored training labels

    Methods:
        _scale_data(X_train, X_val): Scales training and validation features
        fit(X, y, scale_data, save): Trains the classifier on provided data
        _majority_voting(neigh_idxs): Determines class by majority vote of neighbors
        predict(X, scale_data, loo): Makes class predictions for input features
        predict_proba(X): Calculates probability estimates based on neighbor votes

    Example:
        >>> # Generate 20 (resp. 5) training (resp. validation) samples with 4 features 
        >>> X_train, X_val = np.random.random((20, 4)), np.random.random((5, 4))
        >>> # Generate binary classes
        >>> y_train, y_val = np.random.randint(0,2, (20)), np.random.randint(0,2, (5))

        >>> (array([[0.61798233, 0.65360835, 0.1029108 , 0.54929112],
        ...         [0.89883498, 0.53387149, 0.30059125, 0.26111361],
        ...         [0.23928837, 0.20361755, 0.35225478, 0.76946751], 
        ...         [     ...                               ...     ]),
        >>> array([0, 0, 0, ...]))

        >>> from mltoolbox.classification import KnnClassifier
        >>> knn = KnnClassifier(n_neighbors=5, metric='cosine')
        >>> knn.fit(X_train, y_train, scale_data=True)
        >>> # Leave-One-Out validation: Predict the labels only for 1-labelled samples
        >>> to_keep = np.where(y_train==1)[0].reshape(-1, 1) # Get the indices
        >>> knn.predict(to_keep, scale_data=True, loo=True) # Pass the indices

        >>> array([0 1 0 0 1 1 1 1])

        >>> # Standard validation
        >>> y_pred = knn.predict(X_val, scale_data=True, loo=False)

        >>> array([0, 0, 1, 0, 0])

        >>> # Get k-nearest-neighbors class probability
        >>> knn.predict_proba(to_keep)

        >>> array([0.4, 0.4, 0.8, 0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.8, 0.8])
    """

    def __init__(self, n_neighbors: int = 7, model_path: str = None, metric: str = 'cosine',
                 _load_model: bool = False):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.neighbors, self.y_train = None, None
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          metric=metric, n_jobs=-1)

        if _load_model:
            self.scaler = joblib.load(f'{self.model_path}_knn_scaler.save')
            self.X, self.y = joblib.load(f'{self.model_path}_knn.save')

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

    def fit(self, X: numpy.array, y: numpy.array, scale_data: bool = True, save: bool = False):
        """Train the KNN classifier on provided data.

        Parameters:
            X (numpy.array): Training feature data
            y (numpy.array): Training labels
            scale_data (bool, optional): Whether to scale input features. Defaults to True.
            save (bool, optional): Whether to save model and scaler to disk. Defaults to False.
        """
        self.X, self.y = X, y
        # Data standardization
        if scale_data:
            X, _ = self._scale_data(X)
        # Save the best model according to the max val_accuracy
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.model.fit(X, y)
        # Train the classifier
        if save:
            joblib.dump([X, y], f'{self.model_path}_knn.save')
            joblib.dump(self.scaler, f'{self.model_path}_knn_scaler.save')

    def _majority_voting(self, neigh_idxs: list):
        """Determine class labels through majority voting among neighbors.

        Parameters:
            neigh_idxs (list): Indices of nearest neighbors for each sample

        Returns:
            numpy.array: Predicted class labels based on majority vote
                        Returns 'n.a.' for string labels or -1 for numeric labels
                        when no majority exists
        """
        neigh_labels = self.y[neigh_idxs]
        predictions = []
        for sample in neigh_labels:
            labs, freqs = np.unique(sample, return_counts=True)
            idx = np.argmax(freqs)
            try:
                idx.shape[0]
                if type(labs) == str:
                    predictions.append('n.a.')
                else:
                    predictions.append(-1)
            except IndexError as e:
                predictions.append(labs[idx])

        return np.asarray(predictions)

    def predict(self, X: numpy.array, scale_data: bool = True, loo: bool = False):
        """Generate class predictions for input features.

        Parameters:
            X (numpy.array): Input feature array or indices array if loo=True
            scale_data (bool, optional): Whether to scale input features. Defaults to True.
            loo (bool, optional): Whether to use Leave-One-Out validation mode. Defaults to False.

        Returns:
            numpy.array: Predicted class labels
        """
        if loo:  # Leave-One-Out validation - X is a numpy array with indices
            neighbors = self.model.kneighbors()[1][X]
            y_pred = self._majority_voting(neighbors)
        else:  # Classic fit-predict
            if scale_data:
                X = self.scaler.transform(X)
            y_pred = self.model.predict(X)

        return y_pred

    def predict_proba(self, X: numpy.array):
        """Calculate probability estimates based on neighbor votes.

        Parameters:
            X (numpy.array): Array of indices for samples to predict

        Returns:
            numpy.array: Array of probability estimates for each sample,
                        calculated as the fraction of neighbors matching the true label
        """
        y_neigh = self.y[self.model.kneighbors()[1][X]]
        y_true = self.y[X]
        N = self.model.n_neighbors
        pairs = zip(y_true, y_neigh)
        probas = [np.where(b == a)[0].shape[0]/N for a, b in pairs]
        probas = np.asarray(probas)

        return probas
