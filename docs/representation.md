# Representation
___

- [iWord2Vec](#iword2vec)


## iWord2Vec
___
```python
class mltoolbox.representation.iWord2Vec(c=5, e=64, epochs=1, source=None, 
                                         destination=None, seed=15)

```
[`source code`](./../mltoolbox/preprocessing/iword2vec.py)

Description.

 ###  **Attributes** 
   - **model_path**(_str_) - Description
   - **io**(_tuple_) - Description
   - **_load_model**(_bool_) - Description

 ### **Methods** 
  
  
 ### **Examples** 

```python
>>> # Define a generic label array
>>> corpus = [['Lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur'], 
...           ['adipiscing', 'elit', 'sed', 'do'], 
...           ['eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore']] 
>>> word2vec = iWord2Vec(c=2, e=10, epochs=1, seed=15)
>>> word2vec.train(corpus)
>>> embeddings = word2vec.get_embeddings()
>>> print(embeddings.shape)
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

>>> corpus = [['magna', 'aliqua', 'Ut', 'enim', 'ad', 'minim', 'veniam', 'quis'],
...           ['nostrud', 'exercitation', 'ullamco', 'laboris', 'nisi', 'ut']]
>>> word2vec.update(corpus)
>>> new_embeddings = word2vec.get_embeddings()
>>> print(new_embeddings.shape)

(30, 10)
```