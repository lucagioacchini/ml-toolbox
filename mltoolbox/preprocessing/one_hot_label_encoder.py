# type: ignore

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

class OneHotLabelEncoder():
    """_summary_
    """
    def __init__(self):
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder()
        self.weights = None

    def fit(self, y_train):
        """_summary_

        Parameters
        ----------
        y_train : _type_
            _description_
        """
        self.le.fit(y_train) # Fit the label encoder
        label_encoded = self.le.transform(y_train).reshape(-1, 1)
        self.ohe.fit(label_encoded) # Fit the one hot encoder
        # Compute sample weights for the training
        self.weights = compute_sample_weight(class_weight='balanced', 
                                             y=self.transform(y_train))
    
    def transform(self, target):
        """_summary_

        Parameters
        ----------
        target : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Get the int-encoded labels
        le_transformed = self.le.transform(target).reshape(-1, 1)
        # Get the One-Hot-Encoded labels
        ohe_transformed = self.ohe.transform(le_transformed).toarray()

        return ohe_transformed

    def inverse_transform(self, target):
        """_summary_

        Parameters
        ----------
        target : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Retrieve the original labels
        original_label = self.le.inverse_transform(target)
        
        return original_label

