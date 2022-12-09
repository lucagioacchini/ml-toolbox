# type: ignore
import pandas as pd

def k_class_proba_report(labels, proba, output_dict=False):
    """_summary_

    Parameters
    ----------
    labels : _type_
        _description_
    proba : _type_
        _description_
    output_dict : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    _df = pd.DataFrame([x for x in zip(labels, proba)], 
                    columns=['label', 'pc'])
    df = _df.groupby('label').agg({'pc':['mean', 'count']})
    df.columns=['kpc', 'support']
    df = df.sort_values('support', ascending=False)

    df.loc['macro avg', :] = [df['kpc'].mean(), 
                              df['support'].sum()]
    df.loc['weighted avg', :] = [_df['pc'].mean(), 
                                 df.loc['macro avg', 'support']]
    
    df[['support']] = df[['support']].astype(int)
    df[['kpc']] = df[['kpc']].round(2)

    if output_dict: df = df.T.to_dict()
    else: df=df.to_string(index_names=False)
    
    return df