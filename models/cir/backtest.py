import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import iv
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
sys.path.append(".")
from cir.simulation import DataSimulation

class ModelBacktester:
    def __init__(self, vix_data: pd.Series, treas_data: pd.Series, snp_data: pd.Series,
                 train_window: int = 252, test_periods: int = 12, forecast_days: int = 30,
                 vix_method: str = 'log_likelihood_article', treasures_method: str = 'ols'):
        """
        Инициализация бэктестера
        
        Параметры:
        vix_data: исторические данные VIX
        treas_data: исторические данные казначейских облигаций
        snp_data: исторические данные S&P 500
        train_window: размер обучающего окна в днях
        test_periods: количество тестовых периодов
        forecast_days: количество дней для прогноза
        vix_method: метод оценки параметров для VIX
        treasures_method: метод оценки параметров для Treasuries
        """
        self.vix_data = vix_data
        self.treas_data = treas_data
        self.snp_data = snp_data
        self.train_window = train_window
        self.test_periods = test_periods
        self.forecast_days = forecast_days
        self.vix_method = vix_method
        self.treasures_method = treasures_method
        self.results = []
        self.metrics = {}

    def run_backtest(self):
        """Запуск процедуры бэктестинга"""
        total_days = len(self.snp_data)
        start_idx = total_days - self.test_periods * (self.train_window + self.forecast_days)
        
        if start_idx < 0:
            required = self.test_periods * (self.train_window + self.forecast_days)
            available = total_days
            print(f"Недостаточно данных для бэктеста. Требуется: {required} дней, доступно: {available} дней.")
            print("Уменьшите test_periods или train_window.")
            return
        
        print(f"Запуск бэктеста на {self.test_periods} периодов...")
        print(f"Обучающее окно: {self.train_window} дней, Прогноз: {self.forecast_days} дней")
        
        for i in range(self.test_periods):
            # Определение границ данных
            train_start = start_idx + i * (self.train_window + self.forecast_days)
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = test_start + self.forecast_days
            
            # Проверка границ
            if test_end > total_days:
                print(f"Остановка на периоде {i+1}: выход за пределы данных")
                break
                
            # Извлечение данных
            train_vix = self.vix_data.iloc[train_start:train_end].values
            train_treas = self.treas_data.iloc[train_start:train_end].values
            train_snp = self.snp_data.iloc[train_start:train_end].values
            
            actual_snp = self.snp_data.iloc[test_start:test_end].values
            actual_dates = self.snp_data.index[test_start:test_end]
            
            try:
                # Создание и обучение модели
                model = DataSimulation(
                    vix_array=train_vix,
                    treasures_array=train_treas,
                    snp_array=train_snp,
                    vix_method=self.vix_method,
                    treasures_method=self.treasures_method,  # Исправлено здесь
                    forecast_days=self.forecast_days,
                    num_simulations=100
                )
                
                # Получение прогнозов
                _, _, snp_simulated = model.get_processes()
                median_forecast = np.median(snp_simulated, axis=1)
                
                # Сохранение результатов
                self.results.append({
                    'period': i+1,
                    'train_end_date': self.snp_data.index[train_end-1],
                    'test_start_date': self.snp_data.index[test_start],
                    'test_end_date': self.snp_data.index[test_end-1],
                    'actual': actual_snp,
                    'forecast': median_forecast,
                    'simulations': snp_simulated,
                    'actual_dates': actual_dates
                })
                
                print(f"Период {i+1}/{self.test_periods} завершен: обучение до {self.snp_data.index[train_end-1].strftime('%Y-%m-%d')}, прогноз на {self.forecast_days} дней")
                
            except Exception as e:
                print(f"Ошибка в периоде {i+1}: {str(e)}")
        
        if not self.results:
            print("Не удалось завершить ни одного периода бэктеста!")
            return
            
        print("\nБэктест завершен!")
        self.calculate_metrics()
        
    def calculate_metrics(self):
        """Расчет метрик качества прогноза"""
        all_actual = []
        all_forecast = []
        
        for result in self.results:
            all_actual.extend(result['actual'])
            all_forecast.extend(result['forecast'])
        
        # Преобразование в массивы numpy
        actual_array = np.array(all_actual)
        forecast_array = np.array(all_forecast)
        
        # Расчет метрик
        self.metrics = {
            'MAE': mean_absolute_error(actual_array, forecast_array),
            'RMSE': np.sqrt(mean_squared_error(actual_array, forecast_array)),
            'MAPE': np.mean(np.abs((actual_array - forecast_array) / actual_array)) * 100,
            'R2': 1 - np.sum((actual_array - forecast_array)**2) / np.sum((actual_array - np.mean(actual_array))**2),
            'Correlation': np.corrcoef(actual_array, forecast_array)[0, 1]
        }
        
        print("\nМетрики качества прогноза:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
    
    def plot_backtest_results(self, num_periods=3, all_periods=False):
        """Визуализация результатов бэктеста с возможностью вывода всех периодов"""
        if not self.results:
            print("Сначала запустите бэктест!")
            return
        
        # Определение периодов для визуализации
        if all_periods:
            plot_results = self.results
            print(f"Построение графиков для всех {len(plot_results)} периодов...")
        else:
            plot_results = self.results[-num_periods:]
            print(f"Построение графиков для последних {len(plot_results)} периодов...")
        
        # Разбиваем на несколько фигур при большом количестве периодов
        periods_per_figure = 5  # Максимум 5 периодов на одной фигуре
        num_figures = (len(plot_results) + periods_per_figure - 1) // periods_per_figure
        
        for fig_idx in range(num_figures):
            start_idx = fig_idx * periods_per_figure
            end_idx = min((fig_idx + 1) * periods_per_figure, len(plot_results))
            current_results = plot_results[start_idx:end_idx]
            num_current = len(current_results)
            
            plt.figure(figsize=(14, 5 * num_current))
            
            for idx, result in enumerate(current_results, 1):
                plt.subplot(num_current, 1, idx)
                
                # Даты для прогноза
                dates = result['actual_dates']
                
                # Фактические значения
                plt.plot(dates, result['actual'], 'bo-', label='Фактические значения', linewidth=2, markersize=5)
                
                # Прогноз (медиана)
                plt.plot(dates, result['forecast'], 'r--', label='Прогноз (медиана)', linewidth=2)
                
                # 90% доверительный интервал
                lower_bound = np.percentile(result['simulations'], 5, axis=1)
                upper_bound = np.percentile(result['simulations'], 95, axis=1)
                plt.fill_between(dates, lower_bound, upper_bound, color='gray', alpha=0.3, label='90% доверительный интервал')
                
                # Симуляции
                for i in range(min(10, result['simulations'].shape[1])):
                    plt.plot(dates, result['simulations'][:, i], alpha=0.15, color='green')
                
                plt.title(f"Прогноз S&P 500 (Период {result['period']})\nОбучение до: {result['train_end_date'].strftime('%Y-%m-%d')}", fontsize=14)
                plt.ylabel('Значение S&P 500', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    def plot_error_distribution(self):
        """Визуализация распределения ошибок"""
        if not self.results:
            print("Сначала запустите бэктест!")
            return
            
        errors = []
        for result in self.results:
            period_errors = (result['actual'] - result['forecast']) / result['actual']
            errors.extend(period_errors)
        
        plt.figure(figsize=(14, 3))
        sns.histplot(errors, kde=True, bins=30, color='skyblue')
        plt.title('Распределение относительных ошибок прогноза')
        plt.xlabel('Относительная ошибка (Факт - Прогноз)/Факт')
        plt.ylabel('Плотность')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Добавляем статистику
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        plt.axvline(x=mean_err, color='r', linestyle='--', label=f'Среднее: {mean_err:.4f}')
        plt.axvline(x=mean_err - std_err, color='g', linestyle=':', label=f'±1 STD: {std_err:.4f}')
        plt.axvline(x=mean_err + std_err, color='g', linestyle=':')
        plt.legend()
        plt.show()
    
    def plot_metrics_comparison(self):
        """Сравнение метрик по периодам"""
        if not self.results:
            print("Сначала запустите бэктест!")
            return None
            
        metrics_data = []
        for result in self.results:
            actual = result['actual']
            forecast = result['forecast']
            
            metrics_data.append({
                'Period': result['period'],
                'Start Date': result['test_start_date'],
                'End Date': result['test_end_date'],
                'MAE': mean_absolute_error(actual, forecast),
                'RMSE': np.sqrt(mean_squared_error(actual, forecast)),
                'MAPE': np.mean(np.abs((actual - forecast) / actual)) * 100
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        plt.figure(figsize=(14, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(metrics_df['Start Date'], metrics_df['MAE'], 'o-', color='blue')
        plt.title('Средняя абсолютная ошибка (MAE) по периодам')
        plt.ylabel('MAE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        plt.subplot(3, 1, 2)
        plt.plot(metrics_df['Start Date'], metrics_df['RMSE'], 'o-', color='green')
        plt.title('Корень из средней квадратичной ошибки (RMSE) по периодам')
        plt.ylabel('RMSE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        plt.subplot(3, 1, 3)
        plt.plot(metrics_df['Start Date'], metrics_df['MAPE'], 'o-', color='red')
        plt.title('Средняя абсолютная процентная ошибка (MAPE) по периодам')
        plt.ylabel('MAPE (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('3.png')
        plt.show()
        
        return metrics_df

    def print_summary_report(self):
        """Печать сводного отчета"""
        if not self.results or not self.metrics:
            print("Сначала запустите бэктест!")
            return
            
        print("\n" + "="*70)
        print("СВОДНЫЙ ОТЧЕТ ПО БЭКТЕСТУ МОДЕЛИ ПРОГНОЗИРОВАНИЯ")
        print("="*70)
        
        print(f"\nКонфигурация модели:")
        print(f"- Метод оценки VIX: {self.vix_method}")
        print(f"- Метод оценки Treasuries: {self.treasures_method}")
        print(f"- Окно обучения: {self.train_window} дней")
        print(f"- Прогнозный горизонт: {self.forecast_days} дней")
        print(f"- Количество тестовых периодов: {len(self.results)}")
        
        print(f"\nПериод данных:")
        print(f"- Начало: {self.snp_data.index[0].strftime('%Y-%m-%d')}")
        print(f"- Конец: {self.snp_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"- Всего дней: {len(self.snp_data)}")
        
        print("\nРезультаты бэктестинга:")
        print(f"- Общее количество прогнозов: {len(self.results) * self.forecast_days}")
        print(f"- Период первого прогноза: {self.results[0]['test_start_date'].strftime('%Y-%m-%d')} - {self.results[0]['test_end_date'].strftime('%Y-%m-%d')}")
        print(f"- Период последнего прогноза: {self.results[-1]['test_start_date'].strftime('%Y-%m-%d')} - {self.results[-1]['test_end_date'].strftime('%Y-%m-%d')}")
        
        print("\nМетрики качества прогноза:")
        for metric, value in self.metrics.items():
            print(f"- {metric}: {value:.4f}")
        
        print("\n" + "="*70)