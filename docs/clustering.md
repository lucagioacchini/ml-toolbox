# Preprocessing
___

- [K-Means](#k-means)
- [k-GMA (k-Louvain)](#k-gma)


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
>>> X = np.random.random((20, 4))
>>> from MLToolbox.mltoolbox.clustering import kMeans
>>> # Initialize the clustering algorithm
>>> kmeans = kMeans(n_clusters=3)
>>> type(kmeans.model), kmeans.model

(sklearn.cluster._kmeans.KMeans, KMeans(n_clusters=3))

>>> # Fit the algorithm
>>> kmeans.fit(X, scale_data=True)
>>> # Get the clusters
>>> kmeans.predict(X)

array([0, 1, 2, 2, 2, 1, ...],
      dtype=int32)
```

___
## k-GMA
___
```python
class mltoolbox.clustering.kGMA(n_neighbors=7, model_path=None, metric='cosine', _load_model=False)

```
[`source code`](./../mltoolbox/clustering/k_gma.py)

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
>>> # Generate 20 samples with 4 features 
>>> X = pd.DataFrame(np.random.random((20, 4)))
>>> from MLToolbox.mltoolbox.clustering import kGMA
>>> kgma = kGMA(n_neighbors=3, metric='cosine')
>>> # Build the k-NN-graph and fit the algorithm
>>> kgma.fit(X, scale_data=True)
>>> # Get the clusters
>>> kgma.predict(X)

array([1, 1, 2, 2, 3, 0, 3, ...])
```






