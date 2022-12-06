# Representation
___

- [iWord2Vec](#iword2vec)
- [Multi-modal Autoencoder (MAE)](#multi-modal-autoencoder)


## iWord2Vec
___
```python
class mltoolbox.representation.iWord2Vec(c=5, e=64, epochs=1, source=None, 
                                         destination=None, seed=15)

```
[`source code`](./../mltoolbox/representation/iword2vec.py)

Description.

 ###  **Attributes** 
   - **model_path**(_str_) - Description
   - **io**(_tuple_) - Description
   - **_load_model**(_bool_) - Description

 ### **Methods** 
  
  
 ### **Examples** 

```python
>>> # Define the corpus as a list of list of strings
>>> corpus = [['Lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur'], 
...           ['adipiscing', 'elit', 'sed', 'do'], 
...           ['eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore']] 
>>> from mltoolbox.representation import iWord2Vec
>>> # Initialize the model
>>> word2vec = iWord2Vec(c=2, e=10, epochs=1, seed=15)
>>> # Train the initialized model
>>> word2vec.train(corpus)
>>> # Retrieve the embeddings after the first training
>>> embeddings = word2vec.get_embeddings()
>>> print(embeddings.shape) # Get the vocabulary size and the embeddings size
>>> embeddings.head(3)

(17, 10)
               0         1         2         3         4         5         6  \
dolore  0.086249  0.038482  0.041049  0.063226 -0.051581 -0.031196 -0.059515   
elit    0.093712 -0.070643  0.096178  0.043789 -0.006850 -0.030944  0.039167   
ipsum  -0.056756  0.056412 -0.080288  0.068822 -0.071940  0.010958  0.004222   

               7         8         9  
dolore -0.091163 -0.011349  0.014431  
elit   -0.008497 -0.046373  0.095279  
ipsum   0.088425  0.077777 -0.096294  

>>> # Get a new corpus with new words
>>> corpus = [['magna', 'aliqua', 'Ut', 'enim', 'ad', 'minim', 'veniam', 'quis'],
...           ['nostrud', 'exercitation', 'ullamco', 'laboris', 'nisi', 'ut']]
>>> # Update the existing model with the new corpus
>>> word2vec.update(corpus)
>>> # Retrieve the updated embeddings
>>> new_embeddings = word2vec.get_embeddings()
>>> print(new_embeddings.shape) # Get the vocabulary and the embeddings size

(30, 10)

>>> # Remove the embeddings for a word
>>> del_embeddings(word2vec, ['dolore'])
>>> # Check the new vocabulary size
>>> final_embeddings = word2vec.get_embeddings()
>>> print(final_embeddings.shape)

(29, 10)
```


## Multi-modal Autoencoder
___
```python
class mltoolbox.representation.MultimodalAE(model_path=None, io=None, _load_model=False, losses=None, weights=None)

```
[`source code`](./../mltoolbox/representation/mae.py)

Description.

 ###  **Attributes** 
   - **model_path**(_str_) - Description
   - **io**(_tuple_) - Description
   - **_load_model**(_bool_) - Description

 ### **Methods** 
  
  
 ### **Examples** 

```python
>>> # Define a two modalities dataset. The first modality with 15 features and
>>> # The second with 20 ones
>>> X1_train, X2_train = np.random.random((20, 15)), np.random.random((20, 20))
>>> X_train = np.hstack([X1_train, X2_train]) # Stack the modalities
>>> X_train.shape

(20, 35)

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
>>> 
>>> loss = {'out1':'mse', 'out2':'mse'} # Mean Squared Errors
>>> weights = {'out1':15/35, 'out2':20/35} # Balance losses
>>> 
>>> from mltoolbox.representation import MultimodalAE
>>> mae = MultimodalAE(io=(inputs, outputs), losses=loss, weights=weights)
>>> # Fit the multi-modal autoencoder
>>> mae.fit(training_data=(X_train, X_train), y_sizes=[15, 20], epochs=3)

Epoch 1/3
1/1 [==============================] - 1s 703ms/step - loss: 0.3286 - out1_loss: 0.3294 - out2_loss: 0.3279
Epoch 2/3
1/1 [==============================] - 0s 4ms/step - loss: 0.3236 - out1_loss: 0.3252 - out2_loss: 0.3225
Epoch 3/3
1/1 [==============================] - 0s 4ms/step - loss: 0.3192 - out1_loss: 0.3213 - out2_loss: 0.3176

>>> # Encoder the new features
>>> embeddings = mae.transform(X_train)
>>> X_train.shape, embeddings.shape

((20, 35), (20, 4))
```