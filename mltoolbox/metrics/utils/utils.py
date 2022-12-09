# type: ignore

import pandas as pd

def average_reports(reports, output_dict=False):
    """_summary_

    Parameters
    ----------
    reports : _type_
        _description_
    output_dict : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    df = pd.concat(reports, axis=1)
    df['support'] = df['support'].sum(1)
    df['precision'] = df['precision'].mean(1)
    df['recall'] = df['recall'].mean(1)
    df['f1-score'] = df['f1-score'].mean(1)

    df = pd.DataFrame(df.values[:, :4], index=df.index, columns=df.columns[:4])
    if 'micro avg' in df.index: df=df.drop(index=['micro avg'])
    if 'accuracy' in df.index: df=df.drop(index=['accuracy'])
    report_final = df.drop(index=['macro avg', 'weighted avg'])\
                    .sort_values('support', ascending=False)
    report_final = pd.concat([report_final, 
                            df.loc[['macro avg', 'weighted avg']]])
    report_final[report_final.columns[:3]] = report_final[report_final.columns[:3]]\
                                                                        .round(2)
    report_final[['support']] = report_final[['support']].astype(int)

    if output_dict:
        report_final = report_final.T.to_dict()
    else:
        report_final = report_final.to_string()

    return report_final