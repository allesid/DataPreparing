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
        
    def __init__(self, X, col_y, null_pct=10, nuniqfit_pct=10):
        self.X = X
        self.col_y = col_y
        self.y = X[col_y]
        self.null_pct = null_pct
        self.nuniqfit_pct = nuniqfit_pct
    def fit(self, X, col_y):
        """ Преобразование X """
        X_drop = [col_y]
        #print(type(X_drop))
        for colmn in X.columns:
            print("--------------------------------------")
            print(" Feature:", colmn, X[colmn].dtype)
            if len(X[X[colmn].isna()]) < self.null_pct*len(X)/100:
                if X[colmn].dtype == 'object':
                    X[colmn] = X[colmn].fillna('nofeature')
                    X = self.fit_col(X, colmn, col_y)
                elif X[colmn].dtype == 'int':
                    X[colmn] = X[colmn].fillna(0)
                    print(" Percent of uniquire data:", 100*X[colmn].nunique()/len(X))
                    if X[colmn].nunique() < self.nuniqfit_pct*len(X)/100:
                        X = self.fit_col(X, colmn, col_y)
            else:
                X_drop.append(colmn)
                print("   Drop column:", colmn, " Percent of NA data:", 100*len(X[X[colmn].isna()])/len(X))
        #print(X_drop, type(X_drop))
        X = X.drop(columns=X_drop)
        return X, self.y
        
    def fit_col(self, X, col, col_y):
        """ Преобразование одного столбца X[col] 
            col - имя преобразуемого столбца (признака)
        """
        print(" Percent of NA data:", 100*len(X[X[col].isna()])/len(X))
        print("Initial Category data in", col, "feature:", X[col].unique())
        b = X.groupby([col], as_index=False)[col_y].mean().sort_values([col_y])
        b.index = list(range(len(b)))
        b[col+'_r'] = b.index
        #print(b)
        X[col] = X[col].replace(list(b[col]), list(b[col+'_r']))
        print("Correlation matrix", X[col].corr(self.y))
        print("Result Category data in", col, "feature:",X[col].unique())
        return X
