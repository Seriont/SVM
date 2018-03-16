import numpy as np
from scipy.sparse import csr_matrix
import oracles
import time
# import utils
from scipy.special import expit
from importlib import reload
import math


oracles = reload(oracles)
# utils = reload(utils)


def accuracy_score(y, y_pred):
    return np.mean(y == y_pred)


class SubGDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """
        self.loss_function = oracles.BinaryHinge(**kwargs)
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        
    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is None:
            w = np.zeros(X.shape[1])
        else:
            w = np.copy(w_0)
        prev_w = w + 1
        iter_num = 0
        time_hist = []
        func_hist = [self.loss_function.func(X, y, w)]
        func_best = 100000
        self.w = np.copy(w)
        accuracy_hist = [accuracy_score(y, self.predict(X))]
        while iter_num < self.max_iter:
            
            time_start = time.time()
            prev_w = np.copy(w)
            grad_Q = self.loss_function.grad(X, y, w)
            w = w - (self.step_alpha / ((iter_num + 1) ** self.step_beta)) * grad_Q
            
            time_hist.append(time.time() - time_start)
            
            func_hist.append(self.loss_function.func(X, y, w))
            if func_hist[-1] < func_best:
                func_best = func_hist[-1]
                self.w = np.copy(w)
            accuracy_hist.append(accuracy_score(y, self.predict(X)))
            # if (iter_num + 1) % 1000 == 0:
            #     print('{0} iterations finished'.format(iter_num + 1))
            iter_num += 1
        if trace:
            history = {}
            history['time'] = np.array(time_hist)
            history['func'] = np.array(func_hist)
            history['accuracy'] = np.array(accuracy_hist)
            return history
        
    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        result = X.dot(self.w).ravel()
        result[result >= 0] = 1
        result[result < 0] = -1
        return result
        
    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        return self.loss_function.func(X, y, self.w)
        
    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        return self.loss_function.grad(X, y, self.w)
    
    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return np.copy(self.w)

      
class SSubGDClassifier(SubGDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, batch_size=1, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=10000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        
        max_iter - максимальное число итераций
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = oracles.BinaryHinge(**kwargs)
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.w = None
        
    def fit(self, X, y, w_0=None, trace=False, log_freq=0.01):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        if w_0 is None:
            w = np.zeros(X.shape[1])
        else:
            w = np.copy(w_0)
        prev_w = w + 1
        iter_num = 1
        epoch_hist = [0.0]
        time_hist = [0.0]
        time_start = time.time()
        weight_start = w
        func_hist = [self.loss_function.func(X, y, w)]
        weights_diff_hist = [0.0]
        epoch_num = 0
        self.w = np.copy(w)
        accuracy_hist = [accuracy_score(y, self.predict(X))]
        all_indexes = np.random.permutation(X.shape[0])
        ind_st = 0 
        time_start = time.time()
        func_best = 1000000
        while iter_num < self.max_iter:
            prev_w = np.copy(w)
            #indexes = np.random.choice(X.shape[0], self.batch_size, replace=False)
            indexes = all_indexes[ind_st: ind_st + self.batch_size]
            ind_st += self.batch_size
            if ind_st >= X.shape[0]:
                ind_st = 0
                indexes = np.random.permutation(X.shape[0])
            X_batch, y_batch = X[indexes, :], y[indexes]
            grad_Q = self.loss_function.grad(X_batch, y_batch, w)
            #print (grad_Q)
            w = w - (self.step_alpha / ((iter_num + 1) ** self.step_beta)) * grad_Q
            epoch_num = iter_num * self.batch_size / X.shape[0]
            
            if epoch_num - epoch_hist[-1] > log_freq:
                #print(epoch_num)
                epoch_hist.append(epoch_num)
                time_hist.append(time.time() - time_start)
                diff = w - weight_start
                weights_diff_hist.append(np.sum(np.sum(diff * diff)))
                weight_start = w
                accuracy_hist.append(accuracy_score(y, self.predict(X)))
                func_hist.append(self.loss_function.func(X, y, w))
                if func_hist[-1] < func_best:
                    func_best = func_hist[-1]
                    self.w = np.copy(w)
                # print('{0} epoch finished'.format(round(epoch_num)))
            iter_num += 1

        if trace:
            history = {}
            history['time'] = np.array(time_hist)
            history['func'] = np.array(func_hist)
            history['epoch_num'] = np.array(epoch_hist)
            history['weights_diff'] = np.array(weights_diff_hist)
            history['accuracy'] = np.array(accuracy_hist)
            return history


class PEGASOSMethod:
    """
    Реализация метода Pegasos для решения задачи svm.
    """
    def __init__(self, step_lambda=1.0, batch_size=1, max_iter=10000, random_seed=100):
        """
        step_lambda - величина шага, соответствует 
        
        batch_size - размер батча
        
        num_iter - число итераций метода, предлагается делать константное
        число итераций 
        """
        self.step_lambda = step_lambda
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.w = None
        
    def fit(self, X, y, trace=False, log_freq=0.01):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        w = np.zeros(X.shape[1])
        
        prev_w = w + 1
        iter_num = 1
        epoch_hist = [0.0]
        time_hist = [0.0]
        time_start = time.time()
        weight_start = w
        func_hist = [self.loss_function(X, y, w)]
        weights_diff_hist = [0.0]
        epoch_num = 0
        self.w = np.copy(w)
        accuracy_hist = [accuracy_score(y, self.predict(X))]
        all_indexes = np.random.permutation(X.shape[0])
        ind_st = 0 
        time_start = time.time()
        func_best = 1000000
        while iter_num < self.max_iter:
            prev_w = np.copy(w)
            #indexes = np.random.choice(X.shape[0], self.batch_size, replace=False)
            indexes = all_indexes[ind_st: ind_st + self.batch_size]
            ind_st += self.batch_size
            if ind_st >= X.shape[0]:
                ind_st = 0
                indexes = np.random.permutation(X.shape[0])
            X_batch, y_batch = X[indexes, :], y[indexes]
            alpha = 1.0 / (iter_num * self.step_lambda)
            mask = X_batch.dot(w) * y_batch < 1.0
            w = w * (1.0 - alpha * self.step_lambda) + \
                alpha * X_batch.T.dot(mask * y_batch) / self.batch_size
            if self.step_lambda * w.T.dot(w) > 1:
                w = w / math.sqrt(self.step_lambda * w.T.dot(w))
            epoch_num = iter_num * self.batch_size / X.shape[0]
            
            if epoch_num - epoch_hist[-1] > log_freq:
                #print(epoch_num)
                epoch_hist.append(epoch_num)
                time_hist.append(time.time() - time_start)
                diff = w - weight_start
                weights_diff_hist.append(np.sum(np.sum(diff * diff)))
                weight_start = w
                accuracy_hist.append(accuracy_score(y, self.predict(X)))
                func_hist.append(self.loss_function(X, y, w))
                if func_hist[-1] < func_best:
                    func_best = func_hist[-1]
                    self.w = np.copy(w)
            iter_num += 1

        if trace:
            history = {}
            history['time'] = np.array(time_hist)
            history['func'] = np.array(func_hist)
            history['epoch_num'] = np.array(epoch_hist)
            history['weights_diff'] = np.array(weights_diff_hist)
            history['accuracy'] = np.array(accuracy_hist)
            return history
        

    def loss_function(self, X, y, w):
        hinge = 1 - y * X.dot(w)
        hinge[hinge < 0.0] = 0.0
        return 0.5 * (w * w).sum() + (1 / self.step_lambda) * hinge.mean(axis=0)

    def predict(self, X):
        """
        Получить предсказания по выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        """
        result = X.dot(self.w).ravel()
        result[result >= 0] = 1
        result[result < 0] = -1
        return result
        