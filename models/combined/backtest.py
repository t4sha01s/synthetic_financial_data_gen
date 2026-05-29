import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson

def print_final_report(real, synth):
    comparison = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Skewness', 'Kurtosis'],
        'Real S&P 500': [np.mean(real), np.std(real), stats.skew(real), stats.kurtosis(real)],
        'Synthetic (GAN+CIR)': [np.mean(synth), np.std(synth), stats.skew(synth), stats.kurtosis(synth)]
    })
    
    print("\n" + "="*50)
    print("ОТЧЕТ")
    print("="*50)
    print(comparison.to_string(index=False))
    print("-"*50)
    print(f"Точность по Куртозису: {100 - abs(stats.kurtosis(real)-stats.kurtosis(synth))/stats.kurtosis(real)*100:.2f}%")
    print("="*50)

def plot_ACF(real, synth):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.rcParams.update({'font.size': 10})
    plot_acf(real, ax=axes[0, 0], title="ACF Real Returns (S&P 500)", lags=20)
    plot_acf(synth, ax=axes[0, 1], title="ACF Synthetic Returns (GAN-CIR)", lags=20)
    plot_acf(real**2, ax=axes[1, 0], title="ACF Squared Real Returns (Volatility Clustering)", lags=20)
    plot_acf(synth**2, ax=axes[1, 1], title="ACF Squared Synthetic Returns (Volatility Clustering)", lags=20)
    axes[0, 0].set_ylabel("Автокорреляция")
    axes[1, 0].set_ylabel("Автокорреляция (Квадраты)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def perform_statistical_validation(actual_prices, simulated_matrix):
    plt.figure(figsize=(14, 5))
    dates = np.arange(len(actual_prices))
    median = np.median(simulated_matrix, axis=1)
    lower_95 = np.percentile(simulated_matrix, 2.5, axis=1)
    upper_95 = np.percentile(simulated_matrix, 97.5, axis=1)
    lower_50 = np.percentile(simulated_matrix, 25, axis=1)
    upper_50 = np.percentile(simulated_matrix, 75, axis=1)

    plt.fill_between(dates, lower_95, upper_95, color='red', alpha=0.1, label='95% Доверительный интервал')
    plt.fill_between(dates, lower_50, upper_50, color='red', alpha=0.2, label='50% Доверительный интервал')
    
    plt.plot(dates, median, color='red', lw=2, label='Медианный прогноз (GAN+CIR)')
    plt.plot(dates, actual_prices, color='black', lw=3, marker='o', markersize=4, label='Реальный S&P 500 (Ground Truth)')
    
    plt.xlabel("Дни прогноза", fontsize=12)
    plt.ylabel("Значение индекса S&P 500", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    hits = np.logical_and(actual_prices >= lower_95, actual_prices <= upper_95)
    hit_ratio = np.mean(hits)
    
    ks_stat, p_value = stats.ks_2samp(actual_prices, simulated_matrix[-1, :])

    print("\n" + "="*40)
    print(f"ОТЧЕТ ПО ВАЛИДАЦИИ")
    print("="*40)
    print(f"Hit Ratio (95% CI): {hit_ratio:.2%}")
    print(f"KS-statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print("-"*40)
    
    if hit_ratio >= 0.90:
        print("Ура")
    else:
        print("Не ура")
    plt.show()

def check_residuals_stability(synth_returns):
    dw = durbin_watson(synth_returns)
    print(f"Durbin-Watson statistic: {dw:.4f}")
    if 1.8 <= dw <= 2.2:
        print("\nИТОГ: Значение близко к 2.0. Автокорреляция первого порядка отсутствует.")
        print("Это подтверждает, что модель воспроизводит свойство эффективности рынка.")
    elif dw < 1.8:
        print("\nИТОГ: Наблюдается положительная автокорреляция.")
    else:
        print("\nИТОГ: Наблюдается отрицательная автокорреляция.")