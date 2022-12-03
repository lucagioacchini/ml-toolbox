from sklearn.preprocessing import StandardScaler
from ..preprocessing import OneHotLabelEncoder
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
import numpy as np

class DeepClassifier():
    def __init__(self, model_path=None, io=None, _load_model=False):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = OneHotLabelEncoder()
        
        # If input and output layers are provided, init the classifier
        if type(io)==tuple:
            self.model=self._init_model(io)
            # Save the best model according to the max val_accuracy
            self.saver = ModelCheckpoint(filepath=model_path,
                                    monitor='val_accuracy', mode='max', 
                                    save_best_only=True, verbose=False)
            
        if _load_model:
            self.model = load_model(model_path)

            
    def _init_model(self, io):
        classifier = Model(io[0], io[1])
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', 
                           metrics=['accuracy'])
        
        return classifier
    
    def _scale_data(self, X_train, X_val):
        # Fit the scaler on training data
        self.scaler.fit(X_train)
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val)!=type(None): 
            X_val = self.scaler.transform(X_val) 
            
        return X_train, X_val
    
    
    def _encode_labels(self, y_train, y_val):
        # Fit the encoder on training labels
        self.label_encoder.fit(y_train) 
        # Encode training labels
        y_train = self.label_encoder.transform(y_train)
        # Encode validation labels
        if type(y_val)!=type(None): 
            y_val = self.label_encoder.transform(y_val)
            
        return y_train, y_val

    
    def fit(self, training_data, validation_data = None, scale_data=True, 
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
            X_train, X_val = self._scale_data(X_train, X_val)
        
        # Encode labels through One-Hot-Encoding
        y_train, y_val = self._encode_labels(y_train, y_val)

        if type(X_val)!=type(None) and type(y_val)!=type(None):
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train the classifier
        if save:
            history = self.modelfit(X_train, y_train, epochs=epochs, 
                                validation_data=validation_data,
                                batch_size=batch_size, shuffle=True, 
                                sample_weight=self.label_encoder.weights, 
                                callbacks=[self.saver], verbose=verbose) 
        else:
            history = self.model.fit(X_train, y_train, epochs=epochs, 
                                validation_data=validation_data,
                                batch_size=batch_size, shuffle=True, 
                                sample_weight=self.label_encoder.weights, 
                                verbose=verbose) 
        
        return history

    
    def predict(self, X, scale_data=True):
        y_pred_proba = self.predict_proba(X, scale_data)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred

    
    def predict_proba(self, X, scale_data=True):
        if scale_data:
            X = self.scaler.transform(X)
        return self.model(X).numpy()