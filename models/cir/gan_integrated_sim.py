import torch
import numpy as np
from cir.simulation import DataSimulation

class GANIntegratedSimulation(DataSimulation):
    def __init__(self, generator, noise_dim, *args, **kwargs):
        """
        Параметры:
        generator: обученная модель Generator (WGAN-GP)
        noise_dim: размерность входного шума для GAN
        *args, **kwargs: параметры для базового DataSimulation
        """
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.noise_dim = noise_dim
        self.generator.eval() # Переводим GAN в режим оценки

    def generate_gan_noise(self, trend_conditions):
        """
        Генерирует шум через GAN для каждой симуляции.
        trend_conditions: средние значения тренда для каждой симуляции (batch_size, 1)
        """
        batch_size = self.num_simulations
        
        # Подготовка входных данных для GAN
        noise = torch.randn(batch_size, self.noise_dim)
        # В качестве условия (condition) используем текущий тренд
        condition = torch.tensor(trend_conditions, dtype=torch.float32).view(-1, 1)
        
        with torch.no_grad():
            # GAN генерирует последовательность остатков длиной seq_len
            # Если forecast_days > seq_len, нам нужно будет генерировать кусками или адаптировать
            fake_residuals = self.generator(noise, condition).numpy() # (batch_size, seq_len)
            
        return fake_residuals.T # Возвращаем (seq_len, batch_size)

    def snp_component(self, array, vix_stimulated, treasures_stimulated, dW):
        """
        Переопределенный метод: моделирование S&P 500 с GAN-шумом
        """
        trajectories = np.zeros((self.forecast_days, self.num_simulations))
        trajectories[0, :] = array[-1]
        
        # 1. Считаем drift_adjustment как в базе
        hist_log_returns = np.diff(np.log(self.snp_array))
        hist_vix = self.vix_array[:-1] / 100.0
        hist_treasuries = self.treasures_array[:-1]
        drift_adjustment = np.mean(hist_log_returns) - np.mean(hist_treasuries - 0.5 * hist_vix**2)

        # 2. Генерируем GAN шум
        # В качестве условия тренда берем последнее значение из истории
        last_trend = np.mean(hist_log_returns[-30:]) 
        # (На самом деле тренд можно обновлять динамически внутри цикла, но для начала возьмем так)
        
        # ВАЖНО: GAN генерирует остатки сразу на весь период (если forecast_days <= seq_len)
        gan_residuals = self.generate_gan_noise(np.full(self.num_simulations, last_trend))

        for t in range(1, self.forecast_days):
            vix_vol = np.clip(vix_stimulated[t-1, :], 10, 100) / 100.0
            risk_free_rate = treasures_stimulated[t-1, :]
            
            # Дрейф (классический)
            drift = (risk_free_rate - 0.5 * vix_vol**2 + drift_adjustment) * self.dt
            
            # Диффузия: вместо vix_vol * dW[2] мы берем шум из GAN
            # Но чтобы сохранить масштаб волатильности VIX, мы можем "подмешивать" их 
            # или использовать GAN-остатки напрямую, если они уже включают в себя волатильность
            
            # Вариант А: GAN шум как прямая замена случайности dW
            innovation = gan_residuals[t % gan_residuals.shape[0], :] 
            
            log_return = drift + innovation
            trajectories[t, :] = trajectories[t-1, :] * np.exp(log_return)
            
        return trajectories