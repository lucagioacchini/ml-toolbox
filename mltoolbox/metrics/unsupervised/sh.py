from sklearn.metrics import silhouette_samples
import pandas as pd


def silhouette_report(X, y, output_dict=False):
    """Generates a detailed silhouette score report for clustering evaluation.

    Calculates silhouette scores for each sample and aggregates them by cluster,
    providing both per-cluster and overall statistics in a format similar to
    sklearn's classification_report.

    Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features)
        y (array-like): Cluster labels of shape (n_samples,)
        output_dict (bool, optional): If True, returns the report as a dictionary.
                                    If False, returns a formatted string.
                                    Defaults to False.

    Returns:
        Union[str, dict]: Silhouette analysis report containing:
            - Per-cluster average silhouette scores
            - Number of samples per cluster (support)
            - Macro average (unweighted mean across clusters)
            - Weighted average (weighted by cluster sizes)

    Notes:
        - Uses cosine distance metric for silhouette calculation
        - Clusters are sorted by support (size) in descending order
        - Silhouette scores are rounded to 2 decimal places
        - Cluster labels are formatted as 'cluster_X' in the output
    """
    sh = silhouette_samples(X, y, metric='cosine')
    df = pd.DataFrame(zip(y, sh), columns=['cluster', 'sh'])

    _df = df.reset_index().groupby('cluster')\
            .agg({'sh': 'mean', 'index': 'count'})\
            .rename(columns={'index': 'support'})\
            .sort_values('support', ascending=False)
    _df.index = [f'cluster_{i}' for i in _df.index]

    _df.loc['macro avg'] = [_df[['sh']].mean()['sh'],
                            _df[['support']].sum()['support']]
    _df.loc['weighted avg'] = [df[['sh']].mean()['sh'],
                               df[['sh']].count().rename({'sh': 'support'})['support']]

    df = _df.copy()
    df[['support']] = df[['support']].astype(int)
    df[['sh']] = df[['sh']].round(2)

    if output_dict:
        df = df.T.to_dict()
    else:
        df = df.to_string(index_names=False)

    return df
