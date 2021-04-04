def sumProbSubSeq(X, y_col_name, Xtst, ex_num, features, nf, ind=-1, feat_c=[], prob=0.0):
    """
        Probability accounting.
        X - train data array - DataFrame
        ex_num - number of Xtst exemple (row in Xtst)
        y_col_name - target col name in X
        features = X.columns without y_col_name.
        nf = len(features)
    """
    if feat_c:
        assert (len(X.columns) - nf) == 1 , '(len(X.columns) - nf) != 1'
        x = X[feat_c + [y_col_name]]
        xtst = Xtst[feat_c]
        for col in feat_c:
            #print(x[col])
            #print(xtst.loc[ex_num, col])
            x = x[x[col] == xtst.loc[ex_num, col]]
        if not len(x[y_col_name]):
            prob += x[y_col_name].sum()/len(x[y_col_name])
        else:
            prob += X[y_col_name].sum()/len(X[y_col_name])
        #print("feat_c0=", feat_c,"p=",prob)
    for i in range(ind + 1,nf):
        feat_c.append(features[i])
        prob = sumProbSubSeq(X, y_col_name, Xtst, ex_num, features, nf, i, feat_c[:], prob)
        feat_c = feat_c[0:len(feat_c) - 1]
    return prob

slope = X_train['target'].sum()/len(X_train)
y_col_name = 'target'
features = X_train.columns[:-1]
nf = len(features)
y_prob = []
for ex_num in X_test.index:
    prob = sumProbSubSeq(X_train, y_col_name, X_test, ex_num, features, nf)
    prob = prob/(2 ** len(features) - 1)
    if prob > slope:
        y_prob.append(1)
    else:
        y_prob.append(0)
        
err = (y_test - y_prob).sum()
print('err=', err)
