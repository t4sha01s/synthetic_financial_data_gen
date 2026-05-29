import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import iv, log1p
from datetime import datetime, timedelta

class DataSimulation():
    def __init__(self, vix_array: np.ndarray, treasures_array: np.ndarray, snp_array: np.ndarray,
                 vix_method: str = 'log_likelihood_article', treasures_method: str = 'ols',
                 forecast_days: int = 30, num_simulations: int = 10):
        """
        Инициализация симулятора
        
        Параметры:
        vix_array: исторические данные VIX
        treasures_array: исторические данные казначейских облигаций (в десятичной форме)
        snp_array: исторические данные S&P 500
        vix_method: метод оценки параметров для VIX
        treasures_method: метод оценки параметров для Treasuries
        forecast_days: количество дней для прогноза
        num_simulations: количество симуляций
        """
        self.vix_array = vix_array
        self.treasures_array = treasures_array
        self.snp_array = snp_array
        self.vix_method = vix_method
        self.treasures_method = treasures_method
        self.forecast_days = forecast_days  # Дни для прогноза
        self.num_simulations = num_simulations  # Количество симуляций
        self.dt = 1 / 252  # Дневные данные (252 торговых дня в году)
        self.start_date = datetime.today()  # Дата начала прогноза

    def correlated_data_stimulation(self) -> np.ndarray:
        """Генерация коррелированных случайных величин"""
        Z = np.random.normal(size=(3, self.forecast_days, self.num_simulations))
        L = self.get_cholesky_decomposition()
        return np.tensordot(L, Z, axes=(0, 0))

    def cholesky_decomposition(self, matrix: np.ndarray) -> np.ndarray:
        """Вычисление разложения Холецкого с обработкой положительной определенности"""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        positive_matrix = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T
        return np.linalg.cholesky(positive_matrix)

    def log_return(self, array: np.ndarray) -> np.ndarray:
        """Расчет логарифмической доходности"""
        return np.log(array[1:] / array[:-1])

    def get_cholesky_decomposition(self) -> np.ndarray:
        """Получение матрицы Холецкого для коррелированных доходностей"""
        returns_matrix = np.vstack([
            self.log_return(self.vix_array),
            self.log_return(self.treasures_array),
            self.log_return(self.snp_array)
        ])
        return self.cholesky_decomposition(np.corrcoef(returns_matrix))

    def neg_log_likelihood(self, params, array: np.ndarray) -> float:
        """Отрицательное логарифмическое правдоподобие (стандартная реализация)"""
        alpha, theta, sigma = params
        residuals = array[1:] - array[:-1] - alpha * (theta - array[:-1]) * self.dt
        log_likelihood = - (len(array) - 1) * np.log(sigma) - np.sum(residuals ** 2) / (2 * sigma ** 2)
        return -log_likelihood

    def neg_log_likelihood_article(self, params, array: np.ndarray) -> float:
        """Отрицательное логарифмическое правдоподобие (статья)"""
        alpha, theta, sigma = params
        # Проверка условия Феллера
        if 2 * alpha * theta < sigma**2:
            return 1e10  # Штраф за невалидные параметры
        
        N = len(array)
        dt = self.dt
        c = 2 * alpha / (sigma**2 * (1 - np.exp(-alpha * dt)))
        q = 2 * alpha * theta / sigma**2 - 1
        log_likelihood = 0
        
        for i in range(1, N):
            u = c * array[i-1] * np.exp(-alpha * dt)
            v = c * array[i]
            z = 2 * np.sqrt(u * v)
            
            # Численно устойчивая функция Бесселя
            if z < 1e-6:  # Аппроксимация для малых значений
                log_iv = q * np.log(z/2) - np.log(np.math.gamma(q+1))
            else:
                log_iv = np.log(iv(q, z)) + z - 0.5*np.log(2*np.pi*z)
            
            log_likelihood += -u - v + 0.5*q*np.log(v/u) + log_iv
        
        return -(log_likelihood + (N-1)*np.log(c))

    def sse(self, params, array: np.ndarray) -> float:
        """Сумма квадратов ошибок"""
        alpha, theta, sigma = params
        errors = (array[1:] - (array[:-1] + alpha * (theta - array[:-1]) * self.dt)) / np.sqrt(self.dt)
        return np.sum(np.square(errors))

    def params_estimation(self, array: np.ndarray, method: str) -> np.ndarray:
        """Оценка параметров модели CIR"""
        initial_guess = [0.1, np.mean(array), np.std(array)/np.sqrt(self.dt)]
        
        if method == 'log_likelihood':
            result = minimize(self.neg_log_likelihood, initial_guess, args=(array,), 
                             bounds=[(0, None), (None, None), (0, None)])
        elif method == 'log_likelihood_article':
            result = minimize(self.neg_log_likelihood_article, initial_guess, args=(array,), 
                             bounds=[(0, None), (None, None), (0, None)])
        elif method == 'sse':
            result = minimize(self.sse, initial_guess, args=(array,), 
                             bounds=[(0, None), (None, None), (0, None)])
        elif method == 'auto':
            alpha = 2 * np.corrcoef(array[:-1], array[1:])[0, 1] / np.mean(array)
            theta = np.mean(array)
            sigma = np.std(np.diff(array))/np.sqrt(self.dt)
            return alpha, theta, sigma
        elif method == 'ols':
            delta_r = np.diff(array)
            r_t = array[:-1]
            X = sm.add_constant(-r_t)
            model = sm.OLS(delta_r, X).fit()
            alpha = -model.params[1]
            theta = model.params[0] / alpha
            sigma = np.std(delta_r - model.predict(X))/np.sqrt(self.dt)
            return alpha, theta, sigma
        
        # Применение условия Феллера после оптимизации
        alpha, theta, sigma = result.x
        if 2*alpha*theta < sigma**2:
            sigma = np.sqrt(2*alpha*theta*0.99)  # Корректировка для удовлетворения условия
        
        return alpha, theta, sigma

    def cox_ingersoll_ross(self, array: np.ndarray, dW: np.ndarray, method: str) -> np.ndarray:
        """Моделирование процесса CIR"""
        alpha, theta, sigma = self.params_estimation(array, method)
        trajectories = np.zeros((self.forecast_days, self.num_simulations))
        trajectories[0, :] = array[-1]  # Начальное значение - последнее историческое
        
        # Схема полного усечения 
        for t in range(1, self.forecast_days):
            current = trajectories[t-1, :]
            positive_part = np.maximum(current, 0)
            drift = alpha * (theta - positive_part) * self.dt
            diffusion = sigma * np.sqrt(positive_part) * dW[t, :] * np.sqrt(self.dt)
            trajectories[t, :] = current + drift + diffusion
            
            # Отражение отрицательных значений
            trajectories[t, :] = np.maximum(trajectories[t, :], 1e-6)
            
        return trajectories

    def snp_component(self, array: np.ndarray, vix_stimulated: np.ndarray, 
                      treasures_stimulated: np.ndarray, dW: np.ndarray) -> np.ndarray:
        """Моделирование S&P 500 с использованием VIX как волатильности"""
        trajectories = np.zeros((self.forecast_days, self.num_simulations))
        trajectories[0, :] = array[-1]  # Начальное значение - последнее историческое
        
        # Расчет исторической коррекции дрейфа
        hist_log_returns = np.diff(np.log(self.snp_array))
        hist_vix = self.vix_array[:-1] / 100.0
        hist_treasuries = self.treasures_array[:-1]
        drift_adjustment = np.mean(hist_log_returns) - np.mean(hist_treasuries - 0.5 * hist_vix**2)
        
        for t in range(1, self.forecast_days):
            # Используем значения с предыдущего временного шага (t-1)
            vix_prev = np.clip(vix_stimulated[t-1, :], 10, 100) 
            vix_vol = vix_prev / 100.0 
            
            risk_free_rate = treasures_stimulated[t-1, :]
            
            # Дрейф с коррекцией для риск-премии
            drift = (risk_free_rate - 0.5 * vix_vol**2 + drift_adjustment) * self.dt
            
            # Диффузионный член с VIX как прокси волатильности
            diffusion = vix_vol * dW[2][t, :] * np.sqrt(self.dt)
            
            log_return = drift + diffusion
            trajectories[t, :] = trajectories[t-1, :] * np.exp(log_return)
            
        return trajectories

    def get_processes(self) -> np.ndarray:
        """Получение всех смоделированных траекторий"""
        dW = self.correlated_data_stimulation()
        vix_stimulated = self.cox_ingersoll_ross(self.vix_array, dW[0], self.vix_method)
        treasures_stimulated = self.cox_ingersoll_ross(self.treasures_array, dW[1], self.treasures_method)
        snp_simulated = self.snp_component(self.snp_array, vix_stimulated, treasures_stimulated, dW)
        return vix_stimulated, treasures_stimulated, snp_simulated

    def plot_simulations(self, vix_stimulated=None, treasures_stimulated=None, snp_simulated=None):
        """Визуализация результатов симуляции"""
        print(f"Методы оценки параметров: VIX: {self.vix_method}, Treasuries: {self.treasures_method}")
        print(f"Прогноз на {self.forecast_days} дней, {self.num_simulations} симуляций")
        
        if vix_stimulated is None or treasures_stimulated is None or snp_simulated is None:
            vix_stimulated, treasures_stimulated, snp_simulated = self.get_processes()
        
        # Генерация дат для оси X
        dates = [self.start_date + timedelta(days=i) for i in range(self.forecast_days)]
        date_labels = [date.strftime('%Y-%m-%d') for date in dates]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
        
        # Настройка основной информации
        main_title = (f"Прогноз на {self.forecast_days} дней | "
                     f"VIX: {self.vix_method.upper()}, Treasuries: {self.treasures_method.upper()}")
        plt.suptitle(main_title)
        
        # График VIX
        for i in range(vix_stimulated.shape[1]):
            ax1.plot(dates, vix_stimulated[:, i], alpha=0.7)
        ax1.set_title('Прогноз VIX')
        ax1.set_ylabel('Уровень VIX')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # График Treasuries
        for i in range(treasures_stimulated.shape[1]):
            ax2.plot(dates, treasures_stimulated[:, i]*100, alpha=0.7)
        ax2.set_title('Прогноз доходности казначейских облигаций')
        ax2.set_ylabel('Доходность (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # График S&P 500
        for i in range(snp_simulated.shape[1]):
            ax3.plot(dates, snp_simulated[:, i], alpha=0.7)
        ax3.set_title('Прогноз S&P 500')
        ax3.set_ylabel('Значение индекса')
        ax3.set_xlabel('Дата')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # Настройка формата дат
        plt.gcf().autofmt_xdate()
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.show()