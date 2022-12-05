# type: ignore

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

class kMeans():
    def __init__(self, n_clusters=8, model_path=None, _load_model=False, 
                 init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                 random_state=None, algorithm='auto'):
        self.model_path=model_path
        self.scaler = StandardScaler()
        self.neighbors, self.y_train = None, None
        self.model = KMeans(n_clusters, init=init, n_init=n_init, 
                            max_iter=max_iter, tol=tol, algorithm=algorithm,
                            random_state=random_state)
                    
        if _load_model:
            self.scaler = joblib.load(f'{self.model_path}_kmeans_scaler.save')
            self.X = joblib.load(f'{self.model_path}_kmeans.save')
    
    def _scale_data(self, X_train, X_val=None):
        # Fit the scaler on training data
        self.scaler.fit(X_train)
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val)!=type(None): 
            X_val = self.scaler.transform(X_val) 
            
        return X_train, X_val
    
    def fit(self, X, scale_data=True, save=False):
        self.X = X 
        # Data standardization
        if scale_data:
            X, _ = self._scale_data(X)
    
        # Train the classifier
        self.model.fit(X)   
        if save:
            joblib.dump(X, f'{self.model_path}_kmeans.save')
            joblib.dump(self.scaler, f'{self.model_path}_kmeans_scaler.save')
        
    def predict(self, X, scale_data=True):
        if scale_data:
            X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
    
        return y_pred
