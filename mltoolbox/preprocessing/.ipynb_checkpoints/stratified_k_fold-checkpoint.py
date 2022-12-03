import pandas as pd


def flow_stratified_kfolds(task):
    df = pd.read_csv(f'../data/{task}/features/flows.csv', index_col=[0])
    df['app'] = [x.split('_')[0] for x in df.index]
    df['session'] = ['_'.join(x.split('_')[:-1]) for x in df.index]
    if task == 'task02':
        df = df[df.app==df.label]
    uniques = df.value_counts(['session', 'label']).reset_index()

    skf_x = {0:[], 1:[], 2:[], 3:[], 4:[]}
    skf_y = {0:[], 1:[], 2:[], 3:[], 4:[]}

    for c in uniques.label.unique():
        tmp = uniques[uniques.label==c]

        direction = 0
        for i in range(tmp.shape[0]):
            if i>0 and i%5 == 0:
                if direction == 0: direction = 1
                else: direction = 0
            if direction == 0:
                skf_x[i%5].append(df[df.session==tmp.iloc[i].session].index)
                skf_y[i%5].append(df[df.session==tmp.iloc[i].session].label)
            else:
                skf_x[4-i%5].append(df[df.session==tmp.iloc[i].session].index)
                skf_y[4-i%5].append(df[df.session==tmp.iloc[i].session].label)

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