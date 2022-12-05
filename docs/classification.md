# Classification
___

- [Deep Classifier](#deep-classifier)
- [k-Nearest-Neighbors Classifier](#k-nearest-neighbors-classifier)

## Deep Classifier

```python
class mltoolbox.classification.DeepClassifier(model_path=None, io=None, _load_model=False)

```
[source code](./../mltoolbox/classification/deep_classifier.py)

Description.

 ### **Parameters** 
   - **model_path**(_str_) - Description
   - **io**(_tuple_) - Description
   - **_load_model**(_bool_) - Description


 ###  **Attributes** 
   - **model_path**(_str_) - Description
   - **io**(_tuple_) - Description
   - **_load_model**(_bool_) - Description

 ### **Methods** 
  
  
 ### **Examples** 

```python
>>> # Generate 20 (resp. 5) training (resp. validation) samples with 4 features 
>>> X_train, X_val = np.random.random((20, 4)), np.random.random((5, 4))
>>> # Generate binary classes
>>> y_train, y_val = np.random.randint(0,2, (20)), np.random.randint(0,2, (5))
>>> X_train, y_train

(array([[0.61798233, 0.65360835, 0.1029108 , 0.54929112],
        [0.89883498, 0.53387149, 0.30059125, 0.26111361],
        [0.23928837, 0.20361755, 0.35225478, 0.76946751], 
        [     ...                               ...     ]),
 array([0, 0, 0, ...]))

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

Epoch 1/3
1/1 [==============================] - 0s 482ms/step - loss: 0.6901 - accuracy: 0.7000
Epoch 2/3
1/1 [==============================] - 0s 4ms/step - loss: 0.6846 - accuracy: 0.7000
Epoch 3/3
1/1 [==============================] - 0s 4ms/step - loss: 0.6792 - accuracy: 0.7500

# Predict the validation labels
>>> classifier.predict(X_val)

array([0, 0, 0, 0, 1])
```


## K-Nearest-Neighbors-Classifier

```python
class mltoolbox.classification.DeepClassifier(model_path=None, io=None, _load_model=False)

```
[source code](./../mltoolbox/classification/knn_classifier.py)

Description.

 ### **Parameters** 
   - **model_path**(_str_) - Description
   - **io**(_tuple_) - Description
   - **_load_model**(_bool_) - Description


 ###  **Attributes** 
   - **model_path**(_str_) - Description
   - **io**(_tuple_) - Description
   - **_load_model**(_bool_) - Description

 ### **Methods** 
  
  
 ### **Examples** 

```python
>>> # Generate 20 (resp. 5) training (resp. validation) samples with 4 features 
>>> X_train, X_val = np.random.random((20, 4)), np.random.random((5, 4))
>>> # Generate binary classes
>>> y_train, y_val = np.random.randint(0,2, (20)), np.random.randint(0,2, (5))
>>> X_train, y_train

(array([[0.61798233, 0.65360835, 0.1029108 , 0.54929112],
        [0.89883498, 0.53387149, 0.30059125, 0.26111361],
        [0.23928837, 0.20361755, 0.35225478, 0.76946751], 
        [     ...                               ...     ]),
 array([0, 0, 0, ...]))

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

Epoch 1/3
1/1 [==============================] - 0s 482ms/step - loss: 0.6901 - accuracy: 0.7000
Epoch 2/3
1/1 [==============================] - 0s 4ms/step - loss: 0.6846 - accuracy: 0.7000
Epoch 3/3
1/1 [==============================] - 0s 4ms/step - loss: 0.6792 - accuracy: 0.7500

# Predict the validation labels
>>> classifier.predict(X_val)

array([0, 0, 0, 0, 1])
```