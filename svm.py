import numpy as np
from cvxopt import solvers 
from cvxopt import matrix 
from sklearn.metrics.pairwise import pairwise_distances


class SVMSolver:
    """
    Класс с реализацией SVM через метод внутренней точки.
    """
    def __init__(self, C=1.0, method='primal', kernel='linear', gamma=2.0, degree=3):
        """
        C - float, коэффициент регуляризации
        
        method - строка, задающая решаемую задачу, может принимать значения:
            'primal' - соответствует прямой задаче
            'dual' - соответствует двойственной задаче
        kernel - строка, задающая ядро при решении двойственной задачи
            'linear' - линейное
            'polynomial' - полиномиальное
            'rbf' - rbf-ядро
        gamma - ширина rbf ядра, только если используется rbf-ядро
        d - степень полиномиального ядра, только если используется полиномиальное ядро
        Обратите внимание, что часть функций класса используется при одном методе решения,
        а часть при другом
        """
        self.C = C
        self.method = method
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.w = None
        self.dual = None
        self.X_train = None
    
    def compute_primal_objective(self, X, y):
        """
        Метод для подсчета целевой функции SVM для прямой задачи
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        margins = 1 - y[:, np.newaxis] * (X.dot(self.w) + self.w_0)
        margins[margins <= 0.0] = 0.0
        return 0.5 * (self.w * self.w).sum() + margins.mean() * self.C
        
    def compute_dual_objective(self, X, y):
        """
        Метод для подсчёта целевой функции SVM для двойственной задачи
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """ 
        P = None
        if self.kernel == 'linear':
            P = X.dot(X.T)
        elif self.kernel == 'polynomial':
            P = X.dot(X.T)
            P = np.power(P + 1, self.degree)
        elif self.kernel == 'rbf':
            P = pairwise_distances(X, X)
            P = np.exp(-self.gamma * P * P)
        P = P * y[:, np.newaxis] * y[np.newaxis, :]
        q = -np.ones(X.shape[0])
        
        return 0.5 * self.dual.T.dot(P.dot(self.dual)) + q.T.dot(self.dual)
        
    def fit(self, X, y, tolerance=1e-5, max_iter=20):
        """
        Метод для обучения svm согласно выбранной в method задаче
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        tolerance - требуемая точность для метода обучения
        max_iter - максимальное число итераций в методе
        
        """
        if self.method == 'primal':
            P = np.zeros((X.shape[1] + X.shape[0] + 1, X.shape[1] + X.shape[0] + 1))
            P[np.diag_indices(X.shape[1] + 1)] = 1.0
            P[0, 0] = 0.0
            q = np.zeros(X.shape[0] + X.shape[1] + 1)
            q[X.shape[1] + 1:] = self.C / X.shape[0]
            #print(P)
            #print(q)
            G = np.zeros((2 * X.shape[0], X.shape[1] + X.shape[0] + 1))
            G[: X.shape[0], 0] = y
            G[: X.shape[0], 1: X.shape[1] + 1] = X * y[:, np.newaxis]
            indexes = (np.arange(X.shape[0]), np.arange(X.shape[0]) + X.shape[1] + 1)
            G[indexes] = 1.0
            h = -np.ones(2 * X.shape[0])
            h[X.shape[0]:] = 0.0
            G = -G
            indexes = (np.arange(X.shape[0]) + X.shape[0], np.arange(X.shape[0]) + X.shape[1] + 1)
            G[indexes] = -1
            #print(G)
            #print(h)
            solvers.options['maxiters'] = max_iter
            solvers.options['reltol'] = tolerance
            solvers.options['show_progress'] = False
            result = np.array(solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))['x'])
            self.w = result[1: X.shape[1] + 1]
            self.w_0 = result[0]

        if self.method == 'dual':
            P = None
            if self.kernel == 'linear':
                P = X.dot(X.T)
            elif self.kernel == 'polynomial':
                P = X.dot(X.T)
                P = np.power(P + 1, self.degree)
            elif self.kernel == 'rbf':
                P = pairwise_distances(X, X)
                #print(self.gamma)
                #print (P)
                P = np.exp(-self.gamma * P * P)
            P = P * y[:, np.newaxis] * y[np.newaxis, :]
            q = -np.ones(X.shape[0])
            G = np.zeros((2 * X.shape[0], X.shape[0]))
            indexes = (np.arange(X.shape[0]), np.arange(X.shape[0]))
            G[indexes] = 1.0
            indexes = (np.arange(X.shape[0]) + X.shape[0], np.arange(X.shape[0]))
            G[indexes] = -1.0
            h = np.zeros(X.shape[0] * 2)
            h[: X.shape[0]] = self.C / X.shape[0]
            h[X.shape[0]:] = 0.0
            b = np.zeros(1)
            A = np.zeros((1, X.shape[0]))
            A[0, :] = y
            solvers.options['maxiters'] = max_iter
            solvers.options['reltol'] = tolerance
            solvers.options['show_progress'] = False
            result = np.array(solvers.qp(P=matrix(P), q=matrix(q), G=matrix(G), h=matrix(h), A=matrix(A), b=matrix(b))['x']).ravel()
            #print(result)
            self.dual = result
            self.X_train = X
            self.y_train = y
            self.w = self.get_w(X, y)
            self.w_0 = self.get_w0(X, y)

    def predict(self, X):
        """
        Метод для получения предсказаний на данных
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        """
        if self.method == 'primal' or self.kernel == 'linear':
            result = (X.dot(self.w) + self.w_0).ravel()
            result[result >= 0.0] = 1
            result[result < 0] = -1
            return result
        elif self.kernel == 'polynomial':
            result = np.power(1 + X.dot(self.X_train.T), self.degree).dot(self.dual * self.y_train) + self.w_0
            result[result >= 0.0] = 1
            result[result < 0] = -1
            return result.ravel()
        elif self.kernel == 'rbf':
            result = np.exp(-self.gamma * pairwise_distances(X, self.X_train)).dot(self.dual * self.y_train)
            result[result >= 0.0] = 1
            result[result < 0] = -1
            return result.ravel()
        
    def get_w(self, X=None, y=None):
        """
        Получить прямые переменные (без учёта w_0)
        
        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 
        
        return: одномерный numpy array
        """
        if self.method == 'primal':
            return self.w
        elif self.method == 'dual' and self.kernel == 'linear':
            if X is None:
                return self.w
            return X.T.dot(self.dual * y)
        
    def get_w0(self, X=None, y=None):
        """
        Получить вектор сдвига
        
        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 
        
        return: float
        """
        if self.method == 'primal':
            return self.w_0
        elif self.method == 'dual' and self.kernel == 'linear':
            if X is None:
                return self.w_0
            mask = self.dual != 0.0
            return -X[mask, :][0].dot(self.w) + y[mask][0]
        elif self.kernel == 'rbf':
            mask = self.dual > 1e-2
            obj = X[mask, :][0:1]
            ans = y[mask][0]
            return ans - np.exp(-self.gamma * pairwise_distances(obj, X)).dot(self.dual * y)
        elif self.kernel == 'polynomial':
            mask = self.dual != 0.0
            obj = X[mask, :][0:1]
            ans = y[mask][0]
            return ans - np.power(1 + obj.dot(X.T), self.degree).dot(self.dual * y)
        
    def get_dual(self):
        """
        Получить двойственные переменные
        
        return: одномерный numpy array
        """
        result = np.copy(self.dual)
        result[result < 0] *= -1
        return result

if __name__ == '__main__':
    svm = SVMSolver(method='dual')
    print(np.arange(2))
    print(svm.compute_primal_objective(np.ones((3, 2)), np.ones(3)))
    print(svm.compute_dual_objective(np.ones((3, 2)), np.array([1, -1, 1])))