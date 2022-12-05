# Representation
___

- [iWord2Vec](#iword2vec)


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