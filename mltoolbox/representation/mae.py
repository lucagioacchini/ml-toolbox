# type: ignore

from sklearn.preprocessing import StandardScaler
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import joblib

class MultimodalAE():
    def __init__(self, model_path=None, io=None, _load_model=False, losses=None, 
                 weights=None):
        self.model_path=model_path
        self.model = None
        self.scaler = StandardScaler()
        
        # If input and output layers are provided, init the classifier
        if type(io)==tuple:
            self.model = Model(io[0], io[1])
            self.model.compile(optimizer='adam', loss=losses, 
                               loss_weights=weights)
            
        if _load_model:
            self._load_model()

    def _load_model(self):
        self.model = load_model(f'{self.model_path}_mae.h5')
        i,o = self.extract_encoder()
        self.encoder = Model(i,o)
        self.scaler = joblib.load(f'{self.model_path}_scaler.save')

    
    def _scale_data(self, X_train, X_val):
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val)!=type(None): 
            X_val = self.scaler.transform(X_val) 
            
        return X_train, X_val

    def _extract_y(self, y, y_size):
        _y = []
        cnt = 0
        for i in range(len(y_size)):
            if i == 0:
                _y.append(y[:, :y_size[i]])
            elif i == len(y_size)-1:
                _y.append(y[:, cnt:])
            else:
                _y.append(y[:, cnt:cnt+y_size[i]])
            cnt+=y_size[i]
        return _y
    
    def fit(self, training_data, validation_data=None, y_sizes=[], verbose=1,
            scale_data=True, epochs=10, batch_size=256, save=False):
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

        y_train = self._extract_y(X_train, y_sizes)

        if type(X_val)!=type(None) and type(y_val)!=type(None):
            y_val = self._extract_y(X_val, y_sizes)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train the classifier
        if save:
            # Save the best model according to the max val_accuracy
            saver = ModelCheckpoint(filepath=f'{self.model_path}_mae.h5',
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
        i,o = self.extract_encoder()
        self.encoder = Model(i,o)
        
        return history


    def extract_encoder(self, trainable=False):
        for n in range(len(self.model.layers)):
            layer = self.model.layers[n]

            if trainable: layer.trainable = False
            else: layer.trainable = True

            if n == 0: inputs = layer.input
            elif layer.name == 'Coded': outputs = layer.output
        
        return inputs, outputs

    
    def transform(self, X, scale_data=True):
        if scale_data:
            X = self.scaler.transform(X)
        return self.encoder(X).numpy()