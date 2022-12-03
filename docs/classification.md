# Classification
___

- [Deep Classifier](#deep-classifier)
- [Random Forest](#random-forest)
- [k-Nearest-Neighbors Classifier](#k-nearest-neighbors-classifier)

## Deep Classifier

```python
class mltoolbox.classification.DeepClassifier(model_path=None, io=None, _load_model=False)

```
[source code](./mloolbox/classification/deep_classifier.py)

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
  
  

  ```python
  _init_model(io)

  ```
Description.

  **Parameters** 
   - **io**(_tuple_) - Description


  **Returns** 


<br><br>

  ```python
_scale_data(self, X_train, X_val)

```
Description.

  **Parameters** 
   - **io**(_tuple_) - Description


  **Returns** 


<br><br>

```python
_encode_labels(self, y_train, y_val)

```
Description.

  **Parameters** 
   - **io**(_tuple_) - Description


  **Returns** 
  

  <br><br>
