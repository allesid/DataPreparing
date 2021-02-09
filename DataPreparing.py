class DataPreparing()
    """ Преобразование массива pandas.DataFrame для задач регрессии
        машинного обучения
        Требует import pandas as pd
        Категорийные данные преобразуются в целые числа, располагающиеся
        по возрастанию среднего значения чисел из столбца результата y = X[col_y],
        относящихся к данной категории.
        X - исходный массив типа pandas.DataFrame
        col - имя преобразуемого столбца (признака)
        col_y - имя столбца результата регрессии
    """
        
    def __init__(self, X, col_y)
        self.X = X
        self.col_y = col_y
        self.y = X[col_y]
    
        """ Преобразование X """
    def fit(self, X, col_y):
        X = X.drop([col_y])
        for colmn in X.columns:
           if X[colmn].dtype == 'object':
               return fit_col(self, col, col_y)
        
    def fit_col(self, col, col_y):
        """ Преобразование одного столбца X[col] """
        #col = 'MSZoning'
        print(X[col].unique())
        a = X.groupby([col], as_index=False)[col_y].mean()
        b = a.sort_values([col_y])
        b.index = pd.Series(range(len(b)))
        b[col+'_r'] = b.index
        X[col] = X[col].replace(list(b[col]), list(b[col+'_r']))
        X1 = pd.DataFrame()
        X1[col] = X[col]
        X1[col_y] = X[col_y]
        print(X1.corr())
        print(X1[col].unique())
        return X
