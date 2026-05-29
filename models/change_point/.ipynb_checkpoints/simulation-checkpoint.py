import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class ChangePointInjector:
    def __init__(self, series, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.original_series = np.array(series)
        self.series = self.original_series.copy()
        self.change_points = {
            'mean_shift': [],
            'variance_shift': [],
            'trend_change': [],
            'combined': []
        }

    def poisson_jump_times(self, lam, size):
        return sorted(np.cumsum(np.random.poisson(lam=lam, size=size)).astype(int))

    def apply_mean_shift(self, lam=20, magnitude=50, alpha=0.1):
        """
        Реализация сдвига среднего.
        """
        series = self.series.copy()
        jump_times = self.poisson_jump_times(lam, size=3)
        jump_times = sorted([t for t in jump_times if 0 < t < len(series)])
        new_series = series.copy()
        # Накапливаем сдвиги
        for start in jump_times:
            self.change_points['mean_shift'].append(start)
            # величина изменения целевого уровня 
            delta_theta = np.random.choice([-1, 1]) * magnitude
            t_steps = np.arange(len(series) - start)
            # Математическое обоснование:
            # Решение уравнения dX = alpha(theta - X)dt дает переход вида:
            # Shift(t) = Delta_Theta * (1 - exp(-alpha * t))
            adaptation_curve = delta_theta * (1 - np.exp(-alpha * t_steps))
            new_series[start:] += adaptation_curve
        return new_series

    def apply_variance_shift(self, lam=30, magnitude=2.0):
            """
            Реализация сдвига волатильности.
            """
            series = self.series.copy()
            jump_times = self.poisson_jump_times(lam, size=3)
            jump_times = sorted([t for t in jump_times if 0 < t < len(series)])
            new_series = series.copy()
            for start in jump_times:
                self.change_points['variance_shift'].append(start)
                segment_len = len(series) - start
                std_past = np.std(series[:start])
                standard_noise = np.random.normal(0, (magnitude - 1) * std_past, segment_len)
                current_values = np.maximum(new_series[start:], 1e-6)
                relative_scaling = np.sqrt(current_values) / np.sqrt(np.mean(series[:start]))
                extra_diffusion = standard_noise * relative_scaling
                new_series[start:] += extra_diffusion   
            return new_series

    def apply_trend_change(self, lam=20, slope_change=0.5, dt=1/252):
        """
        Реализация изменения тренда.
        """
        series = self.series.copy()
        jump_times = self.poisson_jump_times(lam, size=3)
        jump_times = sorted([t for t in jump_times if 0 < t < len(series)])
        new_series = series.copy()
        for start in jump_times:
            self.change_points['trend_change'].append(start)
            beta = np.random.choice([-1, 1]) * slope_change
            t_steps = np.arange(len(series) - start)
            # Beta * t * dt
            trend_curve = beta * t_steps * dt 
            new_series[start:] += trend_curve
        return new_series
    def apply_trend_change(self, lam=20, slope_change=0.5, dt=1/252):
        """
        Реализация изменения тренда.
        """
        series = self.series.copy()
        jump_times = self.poisson_jump_times(lam, size=3)
        jump_times = sorted([t for t in jump_times if 0 < t < len(series)])
        new_series = series.copy()
        for start in jump_times:
            self.change_points['trend_change'].append(start)
            lookback = 15
            if start > lookback:
                prev_delta = new_series[start] - new_series[start - lookback]
            else:
                prev_delta = new_series[start] - new_series[0]
            beta_sign = -1 if prev_delta > 0 else 1
            beta = beta_sign * slope_change
            t_steps = np.arange(len(series) - start)
            trend_curve = beta * t_steps * dt * series[start] 
            new_series[start:] += trend_curve
        return new_series
    
    def apply_combined_shift(self, lam=25, mean_mag=50, vol_mag=2.5, slope_mag=0.1, alpha=0.1):
            """
            Реализация комбинированной разладки.
            """
            series = self.series.copy()
            jump_times = self.poisson_jump_times(lam, size=5)
            jump_times = sorted([t for t in jump_times if 0 < t < len(series)])
            new_series = series.copy()
            for t in jump_times:
                self.change_points['combined'].append(t)
                t_steps = np.arange(len(new_series) - t)
                # mean_shift
                delta_theta = np.random.choice([-1, 1]) * mean_mag
                adaptation = delta_theta * (1 - np.exp(-alpha * t_steps))
                # variance_shift
                std_past = np.std(new_series[:t]) if t > 0 else np.std(series)
                mean_past = np.mean(new_series[:t]) if t > 0 else np.mean(series)
                standard_noise = np.random.normal(0, (vol_mag - 1) * std_past, len(t_steps))
                scaling = np.sqrt(np.maximum(new_series[t:], 1e-6)) / np.sqrt(np.maximum(mean_past, 1e-6))
                diffusion_shock = standard_noise * scaling
                # variance_shift
                beta = np.random.choice([-1, 1]) * slope_mag
                drift = beta * t_steps * (1/252) # Масштабируем годовой тренд через dt
                # три эффекта 
                new_series[t:] += (adaptation + diffusion_shock + drift)
            return new_series

    
    def plot_all(self, ma_window=7):
        methods_map = {
            'mean_shift': 'apply_mean_shift',
            'variance_shift': 'apply_variance_shift',
            'trend_change': 'apply_trend_change',
            'combined': 'apply_combined_shift'
        }

        fig, axs = plt.subplots(len(methods_map), 3, figsize=(22, 15), sharex=True)
        for i, (label, method_name) in enumerate(methods_map.items()):

            self.change_points[label] = []
            if label == 'combined':
                modified_series = self.apply_combined_shift(lam=25, mean_mag=50, vol_mag=2.5, slope_mag=0.1)
            else:
                modified_series = getattr(self, method_name)()
    
            # Расчет метрик
            returns = np.diff(np.log(np.maximum(modified_series, 1e-8)))
            kernel = np.ones(ma_window) / ma_window
            rolling_mean = np.convolve(modified_series, kernel, mode='valid')
            ma_x_axis = range(ma_window - 1, len(modified_series))
    
            # Absolute 
            axs[i, 0].plot(modified_series, label='Price/Value', color='blue', alpha=0.8)
            cp_label_added = False
            for cp in self.change_points[label]:
                l = 'Change Point' if not cp_label_added else None
                axs[i, 0].axvline(cp, color='red', linestyle='--', alpha=0.7, label=l)
                cp_label_added = True
            axs[i, 0].set_title(f'TYPE {label} - Absolute')
            axs[i, 0].grid(True, alpha=0.3)
            axs[i, 0].legend(loc='best') 
    
            # Log returns 
            axs[i, 1].plot(returns, label='Log Returns', color='blue', alpha=0.7)
            for cp in self.change_points[label]:
                if cp < len(returns):
                    axs[i, 1].axvline(cp, color='red', linestyle='--', alpha=0.7)
            axs[i, 1].set_title(f'TYPE {label} - Log Returns')
            axs[i, 1].grid(True, alpha=0.3)
            axs[i, 1].legend(loc='best')
    
            #  Moving Average 
            axs[i, 2].plot(ma_x_axis, rolling_mean, label=f'{ma_window}-day MA', color='blue', linewidth=2, alpha=0.9)
            cps = sorted(self.change_points[label])
            boundaries = [0] + cps + [len(modified_series)]
            
            mean_label_added = False
            for j in range(len(boundaries) - 1):
                start_idx = int(boundaries[j])
                end_idx = int(boundaries[j+1])
                
                if start_idx < end_idx:
                    segment_mean = np.mean(modified_series[start_idx:end_idx])
                    l_mean = 'Segment Mean' if not mean_label_added else None
                    axs[i, 2].hlines(y=segment_mean, xmin=start_idx, xmax=end_idx, 
                                     color='green', linestyle='-', linewidth=3, label=l_mean)
                    mean_label_added = True
                    mid_point = (start_idx + end_idx) / 2
                    axs[i, 2].text(mid_point, segment_mean, f'{segment_mean:.2f}', 
                                   color='green', fontweight='bold', ha='center', va='bottom',
                                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
            for cp in self.change_points[label]:
                if cp < len(modified_series):
                    axs[i, 2].axvline(cp, color='red', linestyle='--', alpha=0.7)
            
            axs[i, 2].set_title(f'TYPE {label} - Mean Levels between CP')
            axs[i, 2].grid(True, alpha=0.3)
            axs[i, 2].legend(loc='best') 
    
        plt.suptitle("Analysis of managed structural breaks with Mean Levels", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()    