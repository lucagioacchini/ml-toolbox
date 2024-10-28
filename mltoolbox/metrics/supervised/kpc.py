# type: ignore
import pandas as pd


def k_class_proba_report(labels, proba, output_dict=False):
    """Generates a probability score report for k-Nearest Neighbors classifier predictions.

    Calculates the mean probability scores assigned by the kNN classifier to each class,
    providing per-class statistics and overall averages. The probability in kNN is 
    determined by the proportion of neighbor votes for each class.

    Parameters:
        labels (array-like): True class labels
        proba (array-like): Probability scores from kNN classifier for the correct classes
                           (not the full probability distribution)
        output_dict (bool, optional): If True, returns the report as a dictionary.
                                    If False, returns a formatted string.
                                    Defaults to False.

    Returns:
        Union[str, dict]: Report containing:
            - Per-class mean probability scores (kpc: k-neighbor probability class)
            - Number of samples per class (support)
            - Macro average (unweighted mean across classes)
            - Weighted average (weighted by class sizes)

    Notes:
        - Classes are sorted by support (frequency) in descending order
        - Probability scores from kNN are rounded to 2 decimal places
        - 'kpc' represents the probability score based on k-nearest neighbors voting
    """
    _df = pd.DataFrame([x for x in zip(labels, proba)],
                       columns=['label', 'pc'])
    df = _df.groupby('label').agg({'pc': ['mean', 'count']})
    df.columns = ['kpc', 'support']
    df = df.sort_values('support', ascending=False)

    df.loc['macro avg', :] = [df['kpc'].mean(),
                              df['support'].sum()]
    df.loc['weighted avg', :] = [_df['pc'].mean(),
                                 df.loc['macro avg', 'support']]

    df[['support']] = df[['support']].astype(int)
    df[['kpc']] = df[['kpc']].round(2)

    if output_dict:
        df = df.T.to_dict()
    else:
        df = df.to_string(index_names=False)

    return df
