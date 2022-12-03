from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
import numpy as np
import joblib


class OneHotLabelEncoder():
    def __init__(self):
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder()
        self.weights = None

    def fit(self, y_train):
        self.le.fit(y_train) # Fit the label encoder
        label_encoded = self.le.transform(y_train).reshape(-1, 1)
        self.ohe.fit(label_encoded) # Fit the one hot encoder
        # Compute sample weights for the training
        self.weights = compute_sample_weight(class_weight='balanced', 
                                             y=self.transform(y_train))
    
    def transform(self, target):
        # Get the int-encoded labels
        le_transformed = self.le.transform(target).reshape(-1, 1)
        # Get the One-Hot-Encoded labels
        ohe_transformed = self.ohe.transform(le_transformed).toarray()

        return ohe_transformed

    def inverse_transform(self, target):
        # Retrieve the original labels
        original_label = self.le.inverse_transform(target)
        
        return original_label


