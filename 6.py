import numpy as np
import scipy.stats as stats

np.random.seed(42)
theta_true = 2.5
n = 100
alpha = 0.05

# Генерация выборки из распределения Парето
# Используем метод обратной функции
U = np.random.uniform(0, 1, n)
X = (1 - U) ** (1 / (1 - theta_true))  # F(x) = 1 - x^(1-theta), x>=1

# Оценка ММП
logX_sum = np.sum(np.log(X))
theta_hat = 1 + n / logX_sum

# Асимптотический интервал для theta
z = stats.norm.ppf(1 - alpha/2)
se_theta = (theta_hat - 1) / np.sqrt(n)
ci_theta_asymp = (theta_hat - z*se_theta, theta_hat + z*se_theta)

# Доверительный интервал для медианы через theta
def median(theta):
    return 2 ** (1/(theta-1))

med_hat = median(theta_hat)
ci_med_asymp = (median(ci_theta_asymp[1]), median(ci_theta_asymp[0]))

# Бутстрап (параметрический и непараметрический)
B = 1000
boot_theta_nonpar = []
boot_theta_par = []

for _ in range(B):
    # Непараметрический бутстрап
    boot_sample = np.random.choice(X, size=n, replace=True)
    boot_theta_nonpar.append(1 + n / np.sum(np.log(boot_sample)))
    
    # Параметрический бутстрап (генерируем из оценки)
    U_boot = np.random.uniform(0, 1, n)
    X_boot = (1 - U_boot) ** (1 / (1 - theta_hat))
    boot_theta_par.append(1 + n / np.sum(np.log(X_boot)))

ci_theta_boot_nonpar = np.percentile(boot_theta_nonpar, [100*alpha/2, 100*(1-alpha/2)])
ci_theta_boot_par = np.percentile(boot_theta_par, [100*alpha/2, 100*(1-alpha/2)])

# Доверительные интервалы для медианы через бутстрап
ci_med_boot_nonpar = (median(ci_theta_boot_nonpar[1]), median(ci_theta_boot_nonpar[0]))
ci_med_boot_par = (median(ci_theta_boot_par[1]), median(ci_theta_boot_par[0]))

print(f"Истинное θ: {theta_true}")
print(f"Оценка ММП θ̂: {theta_hat:.4f}")
print(f"Оценка медианы: {med_hat:.4f}")
print("\nДоверительные интервалы для θ (95%):")
print(f"Асимптотический: [{ci_theta_asymp[0]:.4f}, {ci_theta_asymp[1]:.4f}]")
print(f"Бутстрап (непар): [{ci_theta_boot_nonpar[0]:.4f}, {ci_theta_boot_nonpar[1]:.4f}]")
print(f"Бутстрап (пар): [{ci_theta_boot_par[0]:.4f}, {ci_theta_boot_par[1]:.4f}]")
print("\nДоверительные интервалы для медианы (95%):")
print(f"Асимптотический: [{ci_med_asymp[0]:.4f}, {ci_med_asymp[1]:.4f}]")
print(f"Бутстрап (непар): [{ci_med_boot_nonpar[0]:.4f}, {ci_med_boot_nonpar[1]:.4f}]")
print(f"Бутстрап (пар): [{ci_med_boot_par[0]:.4f}, {ci_med_boot_par[1]:.4f}]")