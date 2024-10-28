# type: ignore

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight


class OneHotLabelEncoder():
    """A transformer that combines LabelEncoder and OneHotEncoder for categorical variables.

    This class provides functionality to encode categorical labels into one-hot encoded 
    format while maintaining the ability to inverse transform back to original labels.
    It also computes balanced sample weights for handling imbalanced datasets.

    Attributes:
        le (LabelEncoder): Instance of sklearn's LabelEncoder for integer encoding
        ohe (OneHotEncoder): Instance of sklearn's OneHotEncoder for one-hot encoding
        weights (numpy.array): Computed balanced sample weights for the encoded classes

    Methods:
        fit(y_train): Fits the encoder to the training data
        transform(target): Transforms categorical labels to one-hot encoded format
        inverse_transform(target): Converts encoded labels back to original format

    Example:
        >>> # Define a generic label array
        >>> y = np.asarray(['label1', 'label2', 'label1', 'label1', 'label2'])
        >>> from mltoolbox.preprocessing import OneHotLabelEncoder
        >>> # Fit the One-Hot label encoder
        >>> ohle = OneHotLabelEncoder()
        >>> ohle.fit(y)
        >>> # Get One-Hot encoding
        >>> y_one_hot = ohle.transform(y)
        >>> y_one_hot

        >>> [[1. 0.]
        ... [0. 1.]
        ... [1. 0.]
        ... [1. 0.]
        ... [0. 1.]]

        >>> # Get the most probable label
        >>> most_probable = np.argmax(y_one_hot, axis=1)
        >>> # Recover the original labels
        >>> ohle.inverse_transform(most_probable)

        >>> array(['label1', 'label2', 'label1', 'label1', 'label2'])
    """

    def __init__(self):
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder()
        self.weights = None

    def fit(self, y_train):
        """Fits the encoder using the training data.

        Parameters:
            y_train (numpy.array): Array of categorical labels to be encoded.
                                 Shape should be (n_samples,)
        """
        self.le.fit(y_train)  # Fit the label encoder
        label_encoded = self.le.transform(y_train).reshape(-1, 1)
        self.ohe.fit(label_encoded)  # Fit the one hot encoder
        # Compute sample weights for the training
        self.weights = compute_sample_weight(class_weight='balanced',
                                             y=self.transform(y_train))

    def transform(self, target):
        """Transforms categorical labels into one-hot encoded format.

        Parameters:
            target (numpy.array): Array of categorical labels to be transformed.
                                Shape should be (n_samples,)

        Returns:
            numpy.array: One-hot encoded representation of the input labels.
                        Shape will be (n_samples, n_classes)
        """
        # Get the int-encoded labels
        le_transformed = self.le.transform(target).reshape(-1, 1)
        # Get the One-Hot-Encoded labels
        ohe_transformed = self.ohe.transform(le_transformed).toarray()

        return ohe_transformed

    def inverse_transform(self, target):
        """Converts encoded labels back to their original categorical format.

        Parameters:
            target (numpy.array): Array of encoded labels to be converted back.
                                Shape should be (n_samples,) or (n_samples, 1)

        Returns:
            numpy.array: Original categorical labels.
                        Shape will be (n_samples,)
        """
        # Retrieve the original labels
        original_label = self.le.inverse_transform(target)

        return original_label
