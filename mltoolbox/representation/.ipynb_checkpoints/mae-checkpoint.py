# type: ignore

from sklearn.preprocessing import StandardScaler
from ..preprocessing import OneHotLabelEncoder
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import joblib

class MultimodalAE():
    def __init__(self, model_path=None, io=None, _load_model=False):
        self.model_path=model_path
        self.model = None
        self.scaler = StandardScaler()
        
        # If input and output layers are provided, init the classifier
        if type(io)==tuple:
            self.model=self._init_model(io)
            
        if _load_model:
            self.model = load_model(f'{self.model_path}_mae.h5')
            self.scaler = joblib.load(f'{self.model_path}_scaler.save')

            
    def _init_model(self, io):
        mae = Model(io[0], io[1])
        mae.compile(optimizer='adam', loss='mse')
        
        return mae
    
    def _scale_data(self, X_train, X_val):
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val)!=type(None): 
            X_val = self.scaler.transform(X_val) 
            
        return X_train, X_val

    
    def fit(self, training_data, validation_data=None, scale_data=True, 
            epochs=10, batch_size=256, verbose=1, save=False):
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

        if type(X_val)!=type(None) and type(y_val)!=type(None):
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train the classifier
        if save:
            # Save the best model according to the max val_accuracy
            saver = ModelCheckpoint(filepath=f'{self.model_path}_mae.h5',
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

    
    def transform(self, X, scale_data=True):
        if scale_data:
            X = self.scaler.transform(X)
        return self.model(X).numpy()
