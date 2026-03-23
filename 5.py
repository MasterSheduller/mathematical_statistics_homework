import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)
theta_true = 2.0
n = 100
alpha = 0.05

# Генерация выборки
X = np.random.uniform(0, 2*theta_true, n)

# Оценки
theta_mm = np.mean(X)
theta_mmle = np.max(X) / 2
theta_mmle_corr = (n+1)/n * theta_mmle

# Точный доверительный интервал
X_max = np.max(X)
q_low = 2 * (alpha/2)**(1/n)
q_high = 2 * (1 - alpha/2)**(1/n)
ci_exact = (X_max / q_high, X_max / q_low)

# Асимптотический интервал
z = stats.norm.ppf(1 - alpha/2)
se_asymp = theta_mm / np.sqrt(3*n)
ci_asymp = (theta_mm - z*se_asymp, theta_mm + z*se_asymp)

# Бутстраповский интервал (непараметрический)
B = 10000
boot_theta = []
for _ in range(B):
    boot_sample = np.random.choice(X, size=n, replace=True)
    boot_theta.append(np.mean(boot_sample))
ci_boot = np.percentile(boot_theta, [100*alpha/2, 100*(1-alpha/2)])

# Результаты
print(f"Истинное θ: {theta_true}")
print(f"Оценка ММ (x̄): {theta_mm:.4f}")
print(f"Оценка ММП: {theta_mmle:.4f}")
print(f"Исправленная ММП: {theta_mmle_corr:.4f}")
print("\nДоверительные интервалы (95%):")
print(f"Точный: [{ci_exact[0]:.4f}, {ci_exact[1]:.4f}]")
print(f"Асимптотический: [{ci_asymp[0]:.4f}, {ci_asymp[1]:.4f}]")
print(f"Бутстрап: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")