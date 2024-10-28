import pandas as pd
import numpy as np
import joblib


def flow_stratified_kfolds(task):
    """Creates stratified k-folds cross-validation splits for network flow data.

    This function performs the following operations:
    1. Loads flow data from a CSV file
    2. Groups flows by session and application
    3. Creates 5 stratified folds while preserving session integrity
    4. Alternates direction of fold assignment to ensure balanced distribution
    5. Saves the resulting folds to disk

    Parameters:
        task (str): Task identifier (e.g., 'task02') that determines the data path
                   and specific processing rules. For 'task02', only flows where
                   app matches label are included.

    Returns:
        None: The function saves the folds to '../data/{task}/skfolds/folds.save'
              Each fold contains (X_train, X_test, y_train, y_test) arrays
              where samples from the same session are kept together in either
              train or test set.

    Note:
        The resulting folds maintain session-level splitting, meaning all flows
        from the same session will appear in either training or test set, never
        split between them. This prevents data leakage between train and test sets.
    """
    df = pd.read_csv(f'../data/{task}/features/flows.csv', index_col=[0])
    df['app'] = [x.split('_')[0] for x in df.index]
    df['session'] = ['_'.join(x.split('_')[:-1]) for x in df.index]
    if task == 'task02':
        df = df[df.app == df.label]
    uniques = df.value_counts(['session', 'label']).reset_index()

    skf_x = {0: [], 1: [], 2: [], 3: [], 4: []}
    skf_y = {0: [], 1: [], 2: [], 3: [], 4: []}

    for c in uniques.label.unique():
        tmp = uniques[uniques.label == c]

        direction = 0
        for i in range(tmp.shape[0]):
            if i > 0 and i % 5 == 0:
                if direction == 0:
                    direction = 1
                else:
                    direction = 0
            if direction == 0:
                skf_x[i % 5].append(
                    df[df.session == tmp.iloc[i].session].index)
                skf_y[i % 5].append(
                    df[df.session == tmp.iloc[i].session].label)
            else:
                skf_x[4-i %
                      5].append(df[df.session == tmp.iloc[i].session].index)
                skf_y[4-i %
                      5].append(df[df.session == tmp.iloc[i].session].label)

    folds = []
    for i in range(5):
        X_train, y_train = [], []
        for j in range(5):
            if i != j:
                X_train += skf_x[j]
                y_train += skf_y[j]
        X_test = np.hstack(skf_x[i])
        y_test = np.hstack(skf_y[i])
        X_train = np.hstack(X_train)
        y_train = np.hstack(y_train)
        folds.append((X_train, X_test, y_train, y_test))

    joblib.dump(folds, f'../data/{task}/skfolds/folds.save')
