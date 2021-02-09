class DataPreparing():
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
        
    def __init__(self, X, col_y):
        self.X = X
        self.col_y = col_y
        self.y = X[col_y]
    
        """ Преобразование X """
    def fit(self, X, col_y):
        X = X.drop([col_y])
        for colmn in X.columns:
           if X[colmn].dtype == 'object':
               return self.fit_col(col, col_y)
        
    def fit_col(self, col, col_y):
        """ Преобразование одного столбца X[col] """
        print("Initial Category data in", col, "feature:", X[col].unique())
        b = X.groupby([col], as_index=False)[col_y].mean().sort_values([col_y])
        b[col+'_r'] = pd.Series(range(len(b)))
        X[col] = X[col].replace(list(b[col]), list(b[col+'_r']))
        print("Correlation matrix", X[col].corr(X[col_y]))
        print("Result Category data in", col, "feature:",X[col].unique())
        return X
