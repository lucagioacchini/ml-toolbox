# type: ignore

from sklearn.preprocessing import StandardScaler
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import numpy
import joblib


class MultimodalAE():
    """A Multimodal Autoencoder (AE) for feature extraction and transformation, supporting
    model updates, scaling, and model persistence.

    This class implements a multimodal autoencoder with functionality for training, feature
    extraction, data scaling, and saving/loading of the model and scaler. The model is designed
    for custom input-output pairs, supporting diverse feature dimensions.

    Parameters:
        model_path (str, optional): Path to save or load model and scaler files. Defaults to None.
        io (tuple, optional): Input and output layers for model initialization. Defaults to None.
        _load_model (bool, optional): Flag to load an existing model from `model_path`. Defaults to False.
        losses (dict, optional): Dictionary specifying loss functions for different outputs. Defaults to None.
        weights (dict, optional): Dictionary specifying weights for different loss functions. Defaults to None.

    Attributes:
        model_path (str): Path to save or load model files.
        model (keras.models.Model): Main autoencoder model.
        scaler (StandardScaler): Scaler for data normalization.
        encoder (keras.models.Model): Encoder model for feature extraction.

    Methods:
        _load_model(): Loads a pre-trained model and scaler from `model_path`.
        _scale_data(X_train, X_val): Scales training and validation data.
        _extract_y(y, y_size): Extracts and splits target data based on specified sizes.
        fit(training_data, validation_data=None, y_sizes=[], verbose=True, scale_data=True, epochs=10, batch_size=256, save=False): 
            Trains the autoencoder model on the training dataset with optional validation.
        extract_encoder(trainable=False): Extracts the encoder part of the model.
        transform(X, scale_data=True): Transforms new data using the encoder.

    Example:
        >>> # Define a two modalities dataset. The first modality with 15 features and
        >>> # The second with 20 ones
        >>> X1_train, X2_train = np.random.random((20, 15)), np.random.random((20, 20))
        >>> X_train = np.hstack([X1_train, X2_train]) # Stack the modalities
        >>> X_train.shape

        >>> (20, 35)

        >>> # Define the architecture
        >>> inputs = layers.Input((35,))
        >>> # Encoder branch of modality 1
        >>> hidden1 = layers.Lambda(lambda x: x[:, :15])(inputs)
        >>> hidden1 = layers.Dense(32, activation='relu')(hidden1)
        >>> # Encoder branch of modality 2
        >>> hidden2 = layers.Lambda(lambda x: x[:, 15:20])(inputs)
        >>> hidden2 = layers.Dense(32, activation='relu')(hidden2)
        >>> # Concatenate
        >>> hidden = layers.Concatenate()([hidden1, hidden2])
        >>> # Common architecture
        >>> hidden = layers.Dense(32, activation='relu')(hidden)
        >>> # Bottleneck
        >>> hidden = layers.Dense(4, activation='relu', name='Coded')(hidden)
        >>> hidden = layers.Dense(32, activation='relu')(hidden)
        >>> hidden = layers.Dense(32*2, activation='relu')(hidden)
        >>> # Decoder branch of modality 1
        >>> hidden1 = layers.Dense(32, activation='relu')(hidden)
        >>> output1 = layers.Dense(15, activation='linear', name='out1')(hidden1)
        >>> # Decoder branch of modality 2
        >>> hidden2 = layers.Dense(32, activation='relu')(hidden)
        >>> output2 = layers.Dense(20, activation='linear', name='out2')(hidden2)
        >>> outputs = [output1, output2]

        >>> loss = {'out1':'mse', 'out2':'mse'} # Mean Squared Errors
        >>> weights = {'out1':15/35, 'out2':20/35} # Balance losses

        >>> from mltoolbox.representation import MultimodalAE
        >>> mae = MultimodalAE(io=(inputs, outputs), losses=loss, weights=weights)
        >>> # Fit the multi-modal autoencoder
        >>> mae.fit(training_data=(X_train, X_train), y_sizes=[15, 20], epochs=3)

        >>> Epoch 1/3
        ... 1/1 [==============================] - 1s 703ms/step - loss: 0.3286 - out1_loss: 0.3294 - out2_loss: 0.3279
        ... Epoch 2/3
        ... 1/1 [==============================] - 0s 4ms/step - loss: 0.3236 - out1_loss: 0.3252 - out2_loss: 0.3225
        ... Epoch 3/3
        ... 1/1 [==============================] - 0s 4ms/step - loss: 0.3192 - out1_loss: 0.3213 - out2_loss: 0.3176

        >>> # Encoder the new features
        >>> embeddings = mae.transform(X_train)
        >>> X_train.shape, embeddings.shape

        >>> ((20, 35), (20, 4))
    """

    def __init__(self, model_path: str = None, io: tuple = None,
                 _load_model: bool = False, losses: dict = None, weights: dict = None):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()

        # If input and output layers are provided, init the classifier
        if type(io) == tuple:
            self.model = Model(io[0], io[1])
            self.model.compile(optimizer='adam', loss=losses,
                               loss_weights=weights)

        if _load_model:
            self._load_model()

    def _load_model(self):
        """Loads a pre-trained model and scaler.

        This method loads an existing autoencoder model and scaler from the specified
        `model_path` location. It also sets up the encoder model for feature extraction.
        """
        self.model = load_model(f'{self.model_path}_mae')
        i, o = self.extract_encoder()
        self.encoder = Model(i, o)
        self.scaler = joblib.load(f'{self.model_path}_scaler.save')

    def _scale_data(self, X_train: numpy.array, X_val: numpy.array):
        """Scales training and validation data using the scaler.

        Parameters:
            X_train (numpy.array): Training data to be scaled.
            X_val (numpy.array): Validation data to be scaled, if available.

        Returns:
            tuple: Scaled training and validation datasets.
        """
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val) != type(None):
            X_val = self.scaler.transform(X_val)

        return X_train, X_val

    def _extract_y(self, y: numpy.array, y_size: int):
        """Splits the target data into multiple outputs based on specified sizes.

        Parameters:
            y (numpy.array): Target data array.
            y_size (list): List of sizes for each output split.

        Returns:
            list: List of arrays for each extracted output.
        """
        _y = []
        cnt = 0
        for i in range(len(y_size)):
            if i == 0:
                _y.append(y[:, :y_size[i]])
            elif i == len(y_size)-1:
                _y.append(y[:, cnt:])
            else:
                _y.append(y[:, cnt:cnt+y_size[i]])
            cnt += y_size[i]
        return _y

    def fit(self, training_data: tuple, validation_data: tuple = None, y_sizes: list = [], verbose: bool = True,
            scale_data: bool = True, epochs: int = 10, batch_size: int = 256, save: bool = False):
        """Trains the autoencoder model on the given dataset, with optional validation.

        Parameters:
            training_data (tuple): Tuple of (X_train, y_train) for training.
            validation_data (tuple, optional): Tuple of (X_val, y_val) for validation. Defaults to None.
            y_sizes (list): List of output sizes for each output.
            verbose (bool, optional): Whether to display training progress. Defaults to True.
            scale_data (bool, optional): Whether to scale data before training. Defaults to True.
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            batch_size (int, optional): Size of each training batch. Defaults to 256.
            save (bool, optional): Whether to save the model at the end of training. Defaults to False.

        Returns:
            History: Training history object containing details of training process.
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

        y_train = self._extract_y(X_train, y_sizes)

        if type(X_val) != type(None) and type(y_val) != type(None):
            y_val = self._extract_y(X_val, y_sizes)
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        # Train the classifier
        if save:
            # Save the best model according to the max val_accuracy
            saver = ModelCheckpoint(filepath=f'{self.model_path}_mae',
                                    monitor='val_loss', mode='min',
                                    save_best_only=True, verbose=False)

            history = self.model.fit(X_train, y_train, epochs=epochs,
                                     validation_data=validation_data,
                                     batch_size=batch_size, shuffle=True,
                                     callbacks=[saver], verbose=verbose)
            joblib.dump(self.scaler, f'{self.model_path}_scaler.save')

            self._load_model()
        else:
            history = self.model.fit(X_train, y_train, epochs=epochs,
                                     validation_data=validation_data,
                                     batch_size=batch_size, shuffle=True,
                                     verbose=verbose)
        i, o = self.extract_encoder()
        self.encoder = Model(i, o)

        return history

    def extract_encoder(self, trainable: bool = False):
        """Extracts the encoder part of the autoencoder for feature extraction.

        Parameters:
            trainable (bool, optional): If True, sets encoder layers to be trainable. Defaults to False.

        Returns:
            tuple: Input and output layers for the encoder model.
        """
        for n in range(len(self.model.layers)):
            layer = self.model.layers[n]

            if trainable:
                layer.trainable = False
            else:
                layer.trainable = True

            if n == 0:
                inputs = layer.output
            elif layer.name == 'Coded':
                outputs = layer.output

        return inputs, outputs

    def transform(self, X: numpy.array, scale_data: bool = True):
        """Transforms new data using the trained encoder model.

        Parameters:
            X (numpy.array): Input data to be transformed.
            scale_data (bool, optional): If True, scales the data before transformation. Defaults to True.

        Returns:
            numpy.array: Encoded feature representation of the input data.
        """
        if scale_data:
            X = self.scaler.transform(X)
        return self.encoder(X).numpy()
