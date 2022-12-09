from sklearn.metrics import silhouette_samples
import pandas as pd

def silhouette_report(X, y, output_dict=False):
    sh = silhouette_samples(X, y, metric='cosine')
    df = pd.DataFrame(zip(y, sh), columns=['cluster', 'sh'])

    _df = df.reset_index().groupby('cluster')\
            .agg({'sh':'mean', 'index':'count'})\
            .rename(columns={'index':'support'})\
            .sort_values('support', ascending=False)
    _df.index = [f'cluster_{i}' for i in _df.index]

    _df.loc['macro avg'] = [_df[['sh']].mean()['sh'],  \
                            _df[['support']].sum()['support']]
    _df.loc['weighted avg'] = [df[['sh']].mean()['sh'], \
                        df[['sh']].count().rename({'sh':'support'})['support']]
    
    df = _df.copy()
    df[['support']] = df[['support']].astype(int)
    df[['sh']] = df[['sh']].round(2)

    if output_dict: df = df.T.to_dict()
    else: df=df.to_string(index_names=False)

    return df