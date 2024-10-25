# type: ignore

from sklearn.preprocessing import StandardScaler
from ..preprocessing import OneHotLabelEncoder
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import numpy
import joblib


class DeepClassifier():
    """A deep learning classifier that handles data preprocessing, model 
    training, and predictions.

    Parameters:
        model_path (str, optional): Path to save/load model files. Defaults to None.
        io (tuple, optional): Tuple of (input_layer, output_layer) to initialize model. Defaults to None.
        _load_model (bool, optional): Whether to load an existing model from model_path. Defaults to False.

    Attributes:
        model_path (str): Path for model saving/loading
        model (keras.Model): The neural network classifier model
        scaler (StandardScaler): Scales input features
        label_encoder (OneHotLabelEncoder): Encodes categorical labels

    Methods:
        _init_model(io): Initializes neural network model from input/output layers
        _scale_data(X_train, X_val): Scales training and validation features
        _encode_labels(y_train, y_val): Encodes training and validation labels
        fit(training_data, validation_data, scale_data, epochs, batch_size, verbose, save): Trains the classifier
        predict(X, scale_data): Makes class predictions for input features
        predict_proba(X, scale_data): Generates class probabilities for input features

    Example:
        >>> # Generate 20 (resp. 5) training (resp. validation) samples with 4 features 
        >>> X_train, X_val = np.random.random((20, 4)), np.random.random((5, 4))
        >>> # Generate binary classes
        >>> y_train, y_val = np.random.randint(0,2, (20)), np.random.randint(0,2, (5))
        >>> X_train, y_train

        >>> (array([[0.61798233, 0.65360835, 0.1029108 , 0.54929112],
        >>>         [0.89883498, 0.53387149, 0.30059125, 0.26111361],
        >>>         [0.23928837, 0.20361755, 0.35225478, 0.76946751], 
        >>>         [     ...                               ...     ]),
        >>> array([0, 0, 0, ...]))

        >>> from mltoolbox.classification import DeepClassifier
        >>> from keras import layers 
        >>> # Define architecture
        >>> inputs = layers.Input((4,))
        >>> hidden = layers.Dense(64, activation='relu')(inputs)
        >>> outputs = layers.Dense(2, activation='softmax')(hidden)
        >>> # Initialize the classifier
        >>> classifier = DeepClassifier(io=(inputs, outputs))
        >>> # Train the classifier for 3 epochs. Standardize data before training
        >>> classifier.fit(training_data=(X_train, y_train), scale_data=True, epochs=3)

        >>> Epoch 1/3
        >>> 1/1 [==============================] - 0s 482ms/step - loss: 0.6901 - accuracy: 0.7000
        >>> Epoch 2/3
        >>> 1/1 [==============================] - 0s 4ms/step - loss: 0.6846 - accuracy: 0.7000
        >>> Epoch 3/3
        >>> 1/1 [==============================] - 0s 4ms/step - loss: 0.6792 - accuracy: 0.7500

        >>> # Predict the validation labels
        >>> classifier.predict(X_val)

        array([0, 0, 0, 0, 1])
    """

    def __init__(self, model_path: str = None, io: tuple = None, _load_model: bool = False):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = OneHotLabelEncoder()

        # If input and output layers are provided, init the classifier
        if type(io) == tuple:
            self.model = self._init_model(io)

        if _load_model:
            self.model = load_model(f'{self.model_path}_classifier')
            self.scaler = joblib.load(f'{self.model_path}_scaler.save')
            self.label_encoder = joblib.load(f'{self.model_path}_ohle.save')

    def _init_model(self, io: tuple):
        """Initialize the neural network classifier model.

        Parameters:
            io (tuple): Tuple containing (input_layer, output_layer) for model architecture

        Returns:
            keras.Model: Compiled neural network model with categorical crossentropy loss
        """
        classifier = Model(io[0], io[1])
        classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])

        return classifier

    def _scale_data(self, X_train: numpy.array, X_val: numpy.array):
        """Scale training and validation data using StandardScaler.

        Parameters:
            X_train (numpy.array): Training feature data
            X_val (numpy.array): Validation feature data, can be None

        Returns:
            tuple: (scaled_X_train, scaled_X_val) - Scaled feature arrays
        """
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val) != type(None):
            X_val = self.scaler.transform(X_val)

        return X_train, X_val

    def _encode_labels(self, y_train: numpy.array, y_val: numpy.array):
        """Encode categorical labels using OneHotLabelEncoder.

        Parameters:
            y_train (numpy.array): Training labels
            y_val (numpy.array): Validation labels, can be None

        Returns:
            tuple: (encoded_y_train, encoded_y_val) - One-hot encoded label arrays
        """
        # Fit the encoder on training labels
        self.label_encoder.fit(y_train)
        # Encode training labels
        y_train = self.label_encoder.transform(y_train)
        # Encode validation labels
        if type(y_val) != type(None):
            y_val = self.label_encoder.transform(y_val)

        return y_train, y_val

    def fit(self, training_data: tuple, validation_data: tuple = None, scale_data: bool = True,
            epochs: int = 10, batch_size: int = 256, verbose: bool = True, save: bool = False):
        """Train the neural network classifier on provided data.

        Parameters:
            training_data (tuple): Tuple of (X_train, y_train) - training features and labels
            validation_data (tuple, optional): Tuple of (X_val, y_val) - validation features and labels. Defaults to None.
            scale_data (bool, optional): Whether to scale input features. Defaults to True.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Training batch size. Defaults to 256.
            verbose (bool, optional): Whether to print training progress. Defaults to True.
            save (bool, optional): Whether to save best model based on validation accuracy. Defaults to False.

        Returns:
            keras.callbacks.History: Training history containing loss and metrics
        """
        # Get training datasets
        X_train, y_train = training_data
        # If provided, get validation datasets
        if type(validation_data) == tuple:
            X_val, y_val = validation_data
        else:
            X_val, y_val = None, None

        # Data standardization
        if scale_data:
            # Fit the scaler on training data
            self.scaler.fit(X_train)
            X_train, X_val = self._scale_data(X_train, X_val)

        # Encode labels through One-Hot-Encoding
        y_train, y_val = self._encode_labels(y_train, y_val)

        if type(X_val) != type(None) and type(y_val) != type(None):
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        # Train the classifier
        if save:
            # Save the best model according to the max val_accuracy
            saver = ModelCheckpoint(filepath=f'{self.model_path}_classifier',
                                    monitor='val_accuracy', mode='max',
                                    save_best_only=True, verbose=False)

            history = self.model.fit(X_train, y_train, epochs=epochs,
                                     validation_data=validation_data,
                                     batch_size=batch_size, shuffle=True,
                                     sample_weight=self.label_encoder.weights,
                                     callbacks=[saver], verbose=verbose)
            joblib.dump(self.scaler, f'{self.model_path}_scaler.save')
            joblib.dump(self.label_encoder, f'{self.model_path}_ohle.save')
        else:
            history = self.model.fit(X_train, y_train, epochs=epochs,
                                     validation_data=validation_data,
                                     batch_size=batch_size, shuffle=True,
                                     sample_weight=self.label_encoder.weights,
                                     verbose=verbose)

        return history

    def predict(self, X: numpy.array, scale_data: bool = True):
        """Generate class predictions for input features.

        Parameters:
            X (numpy.array): Input feature array
            scale_data (bool, optional): Whether to scale input features. Defaults to True.

        Returns:
            numpy.array: Predicted class labels
        """
        y_pred_proba = self.predict_proba(X, scale_data)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred)

        return y_pred

    def predict_proba(self, X: numpy.array, scale_data: bool = True):
        """Generate class probability predictions for input features.

        Parameters:
            X (numpy.array): Input feature array
            scale_data (bool, optional): Whether to scale input features. Defaults to True.

        Returns:
            numpy.array: Array of class probabilities for each sample
        """
        if scale_data:
            X = self.scaler.transform(X)
        return self.model(X).numpy()
