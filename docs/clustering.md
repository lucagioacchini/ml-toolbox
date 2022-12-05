# Preprocessing
___

- [K-Means](#k-means)


## K-Means
___
```python
class mltoolbox.clustering.kMeans(n_clusters=8, model_path=None, 
              _load_model=False, init='k-means++', n_init=10, max_iter=300, 
              tol=0.0001, random_state=None, algorithm='auto')

```
[`source code`](./../mltoolbox/clustering/kmeans.py)

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
>>> X_train = np.random.random((20, 4))
>>> from MLToolbox.mltoolbox.clustering import kMeans
>>> # Initialize the clustering algorithm
>>> kmeans = kMeans(n_clusters=3)
>>> type(kmeans.model), kmeans.model

(sklearn.cluster._kmeans.KMeans, KMeans(n_clusters=3))

>>> # Fit the algorithm
>>> kmeans.fit(X_train, scale_data=True)
>>> # Get the clusters
>>> kmeans.predict(X_train)

array([0, 1, 2, 2, 2, 1, ...],
      dtype=int32)
```

___
## K-Nearest-Neighbors-Classifier
___
```python
class mltoolbox.classification.KnnClassifier(n_neighbors=7, model_path=None, metric='cosine', _load_model=False)

```
[`source code`](./../mltoolbox/classification/knn_classifier.py)

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

(array([[0.61798233, 0.65360835, 0.1029108 , 0.54929112],
        [0.89883498, 0.53387149, 0.30059125, 0.26111361],
        [0.23928837, 0.20361755, 0.35225478, 0.76946751], 
        [     ...                               ...     ]),
 array([0, 0, 0, ...]))

>>> from mltoolbox.classification import KnnClassifier
>>> knn = KnnClassifier(n_neighbors=5, metric='cosine')
>>> knn.fit(X_train, y_train, scale_data=True)
>>> # Leave-One-Out validation: Predict the labels only for 1-labelled samples
>>> to_keep = np.where(y_train==1)[0].reshape(-1, 1) # Get the indices
>>> knn.predict(to_keep, scale_data=True, loo=True) # Pass the indices
>>> classifier.predict(X_val)

array([0 1 0 0 1 1 1 1])

>>> # Standard validation
>>> y_pred = knn.predict(X_val, scale_data=True, loo=False)

array([0, 0, 1, 0, 0])
```






