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


class NumToCat:
    """    
    преобразование числовых данных в категориальные
    min_obs - минимальное число наблюдений для включения в группу
    max_groups - максимальное число групп
    """    
    def __init__(self,min_obs=0,max_groups=10):
        self.min_obs = min_obs
        self.max_groups = max_groups

    def fit(self, df):
        """
        Convert DataFrame nunerical type data to categorical type data in columns
        """
        self.splits_ = {}
        for col in df.columns:
            if df[col].dtype!='object':
                #xm = (df[col].max() + df[col].min())/2.
                xd = (df[col].max() - df[col].min())/self.max_groups
                ss = []
                for i in range(self.max_groups + 1):
                    ss.append(xd * (i - 0.5) + df[col].min())
                self.splits_[col] = ss
        return self.splits_
    
    def transform(self, df):
        for col in self.splits_.keys():
            n_col = []
            for x in df[col]:
                ex = col+ "_" + str(len(self.splits_[col]))
                for i in range(len(self.splits_[col])): # список
                    if x < self.splits_[col][i]:
                        ex = col+ "_" + str(i)
                        break
                n_col.append(ex)
            df[col] = n_col
        return df
class DataPreparing:
    """ Преобразование массива pandas.DataFrame для задач регрессии
        машинного обучения
        Требует import pandas as pd
        Категорийные данные преобразуются в целые числа, располагающиеся
        по возрастанию среднего значения чисел из столбца результата y = X[col_y],
        относящихся к данной категории.
            Исходные данные:
        X - исходный массив типа pandas.DataFrame
        col_y - имя столбца результата регрессии
        null_pct - default: 10 - процентное отношение количества отсутствующих 
                                    данных в столбце к длине столбца
        nuniqfit_pct - default: 10 - процентное отношение количества уникальных 
                                    данных в столбце к длине столбца

            Возвращает:
        X - преобразованный массив типа pandas.DataFrame
        y - столбец результата регрессии
    """
        
    def __init__(self, null_pct=10, nuniqfit_pct=10):
        self.null_pct = null_pct
        self.nuniqfit_pct = nuniqfit_pct
    def fit(self, X, col_y):
        """ Преобразование X """
        self.y = X[col_y]
        X_drop = [col_y]
        #print(type(X_drop))
        for colmn in X.columns:
            #print("--------------------------------------")
            #print(" Feature:", colmn, X[colmn].dtype)
            if len(X[X[colmn].isna()]) < self.null_pct*len(X)/100:
                if X[colmn].dtype == 'object':
                    X[colmn] = X[colmn].fillna('nofeature')
                    X = self.fit_col(X, colmn, col_y)
                elif X[colmn].dtype == 'int':
                    X[colmn] = X[colmn].fillna(0)
                    #print(" Percent of uniquire data:", 100*X[colmn].nunique()/len(X))
                    if X[colmn].nunique() < self.nuniqfit_pct*len(X)/100:
                        X = self.fit_col(X, colmn, col_y)
            else:
                X_drop.append(colmn)
                #print("   Drop column:", colmn, " Percent of NA data:", 100*len(X[X[colmn].isna()])/len(X))
        X = X.drop(columns=X_drop)
        return X, self.y
        
    def fit_col(self, X, col, col_y):
        """ Преобразование одного столбца X[col] 
            col - имя преобразуемого столбца (признака)
        """
        #print(" Percent of NA data:", 100*len(X[X[col].isna()])/len(X))
        #print("Initial Category data in", col, "feature:", X[col].unique())
        b = X.groupby([col], as_index=False)[col_y].mean().sort_values([col_y])
        b.index = list(range(len(b)))
        b[col+'_r'] = b.index
        #print(b)
        X[col] = X[col].replace(list(b[col]), list(b[col+'_r']))
        #print("Correlation matrix", X[col].corr(self.y))
        #print("Result Category data in", col, "feature:",X[col].unique())
        return X
    
    
class DataPreparingTst:
    """ Преобразование массива pandas.DataFrame для задач регрессии
        машинного обучения
        Требует import pandas as pd
        Категорийные данные преобразуются в целые числа, располагающиеся
        по возрастанию среднего значения чисел из столбца результата y = X[col_y],
        относящихся к данной категории.
            Исходные данные:
        X - исходный массив типа pandas.DataFrame
        Xtst - исходный массив тестовых данных типа pandas.DataFrame
        col_y - имя столбца результата регрессии - string
        null_pct - default: 20 - процентное отношение количества отсутствующих 
                                    данных в столбце к длине столбца - numeric
        nuniqfit_pct - default: 10 - процентное отношение количества уникальных 
                                    данных в столMIN  alpha_min= 0.45бце к длине столбца - numeric

            Возвращает:
        X - преобразованный массив данных типа pandas.DataFrame
        Xtst - преобразованный массив тестовых данных типа pandas.DataFrame
        y - столбец результата регрессии из массива X.
    """
        
    def __init__(self, null_pct=20, nuniqfit_pct=10):
        self.null_pct = null_pct
        self.nuniqfit_pct = nuniqfit_pct
        
        
    def fit(self, X, Xtst, col_y):
        self.y = X[col_y]
        """ Преобразование X """
        self.len_X = len(X)
        X=X.append(Xtst)
        self.X_drop = [col_y]
        
        for colmn in X.columns:
            #print("--------------------------------------")
            #print(" Feature:", colmn, X[colmn].dtype)
            if (len(X[X[colmn].isna()]))< self.null_pct*(len(X))/100:
                if X[colmn].dtype == 'object':
                    X[colmn] = X[colmn].fillna(-1)
                    X = self.fit_col(X, colmn, col_y)
                else: # X[colmn].dtype == 'int' or X[colmn].dtype == 'float':
                    X[colmn] = X[colmn].fillna(0)
                    if X[colmn].nunique() < self.nuniqfit_pct*len(X)/100:
                        X = self.fit_col(X, colmn, col_y)
            else:
                self.X_drop.append(colmn)
        X = X.drop(columns=self.X_drop)
        return X.iloc[:self.len_X], X.iloc[self.len_X:], self.y
        
    def fit_col(self, X, col, col_y):
        """ 
        Преобразование одного столбца X[col] 
            col - имя преобразуемого столбца (признака)
        """
        b = X.iloc[:self.len_X].groupby([col], as_index=False)[col_y].mean().sort_values([col_y])
        b.index = list(range(len(b)))
        b[col+'_r'] = b.index
        X[col] = X[col].replace(list(b[col]), list(b[col+'_r']))
        return X
