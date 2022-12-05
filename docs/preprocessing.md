# Preprocessing
___

- [One-Hot Label Encoder](#one-hot-label-encoder)


## One-Hot Label Encoder
___
```python
class mltoolbox.preprocessing.OneHotLabelEncoder()

```
[`source code`](./../mltoolbox/preprocessing/one_hot_label_encoder.py)

Description.

 ###  **Attributes** 
   - **model_path**(_str_) - Description
   - **io**(_tuple_) - Description
   - **_load_model**(_bool_) - Description

 ### **Methods** 
  
  
 ### **Examples** 

```python
>>> # Define a generic label array
>>> y = np.asarray(['label1', 'label2', 'label1', 'label1', 'label2'])
>>> # Fit the One-Hot label encoder
>>> ohle = OneHotLabelEncoder()
>>> ohle.fit(y)
>>> # Get One-Hot encoding
>>> y_one_hot = ohle.transform(y)
>>> y_one_hot

[[1. 0.]
 [0. 1.]
 [1. 0.]
 [1. 0.]
 [0. 1.]]

>>> # Get the most probable label
>>> most_probable = np.argmax(y_one_hot, axis=1)
>>> # Recover the original labels
>>> ohle.inverse_transform(most_probable)

array(['label1', 'label2', 'label1', 'label1', 'label2'], dtype='<U6')
```