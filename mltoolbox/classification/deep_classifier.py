# type: ignore

from sklearn.preprocessing import StandardScaler
from ..preprocessing import OneHotLabelEncoder
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import joblib

class DeepClassifier():
    """_summary_

    Parameters
    ----------
    model_path : _type_, optional
        _description_, by default None
    io : _type_, optional
        _description_, by default None
    _load_model : bool, optional
        _description_, by default False
    """
    def __init__(self, model_path=None, io=None, _load_model=False):
        self.model_path=model_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = OneHotLabelEncoder()
        
        # If input and output layers are provided, init the classifier
        if type(io)==tuple:
            self.model=self._init_model(io)
            
        if _load_model:
            self.model = load_model(f'{self.model_path}_classifier')
            self.scaler = joblib.load(f'{self.model_path}_scaler.save')
            self.label_encoder = joblib.load(f'{self.model_path}_ohle.save')

            
    def _init_model(self, io):
        """_summary_

        Parameters
        ----------
        io : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        classifier = Model(io[0], io[1])
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', 
                           metrics=['accuracy'])
        
        return classifier
    
    def _scale_data(self, X_train, X_val):
        """_summary_

        Parameters
        ----------
        X_train : _type_
            _description_
        X_val : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val)!=type(None): 
            X_val = self.scaler.transform(X_val) 
            
        return X_train, X_val
    
    
    def _encode_labels(self, y_train, y_val):
        """_summary_

        Parameters
        ----------
        y_train : _type_
            _description_
        y_val : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Fit the encoder on training labels
        self.label_encoder.fit(y_train) 
        # Encode training labels
        y_train = self.label_encoder.transform(y_train)
        # Encode validation labels
        if type(y_val)!=type(None): 
            y_val = self.label_encoder.transform(y_val)
            
        return y_train, y_val

    
    def fit(self, training_data, validation_data=None, scale_data=True, 
            epochs=10, batch_size=256, verbose=1, save=False):
        """_summary_

        Parameters
        ----------
        training_data : _type_
            _description_
        validation_data : _type_, optional
            _description_, by default None
        scale_data : bool, optional
            _description_, by default True
        epochs : int, optional
            _description_, by default 10
        batch_size : int, optional
            _description_, by default 256
        verbose : int, optional
            _description_, by default 1
        save : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        # Get training datasets
        X_train, y_train = training_data
        # If provided, get validation datasets
        if type(validation_data)==tuple:
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

        if type(X_val)!=type(None) and type(y_val)!=type(None):
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

    
    def predict(self, X, scale_data=True):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_
        scale_data : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        y_pred_proba = self.predict_proba(X, scale_data)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred

    
    def predict_proba(self, X, scale_data=True):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_
        scale_data : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        if scale_data:
            X = self.scaler.transform(X)
        return self.model(X).numpy()
