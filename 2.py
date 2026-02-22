import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde, skew, norm

np.random.seed(42)

# Генерация выборки (n = 25)
n = 25
sample = np.random.exponential(scale=1, size=n)

print("Выборка:\n", sample)

# a) Статистики выборки
mode_result = stats.mode(sample, keepdims=True)
mode = mode_result.mode[0]
median = np.median(sample)
data_range = np.max(sample) - np.min(sample)
skewness = skew(sample, bias=False)

print("\nМода:", mode)
print("Медиана:", median)
print("Размах:", data_range)
print("Оценка коэффициента асимметрии:", skewness)

# b) ЭФР, гистограмма, boxplot
# Эмпирическая функция распределения
sorted_sample = np.sort(sample)
ecdf = np.arange(1, n + 1) / n

plt.figure()
plt.step(sorted_sample, ecdf, where='post')
plt.title("Эмпирическая функция распределения")
plt.xlabel("x")
plt.ylabel("F_n(x)")
plt.grid()
plt.show()

# Гистограмма
plt.figure()
plt.hist(sample, bins='auto', density=True)
plt.title("Гистограмма")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.show()

# Boxplot
plt.figure()
plt.boxplot(sample, vert=False)
plt.title("Boxplot")
plt.show()

# c) Плотность распределения среднего
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)

# ЦПТ (нормальная аппроксимация)
x_vals = np.linspace(sample_mean - 3*sample_std/np.sqrt(n),
                     sample_mean + 3*sample_std/np.sqrt(n), 200)
clt_density = norm.pdf(x_vals, loc=sample_mean, scale=sample_std/np.sqrt(n))

# Bootstrap
B = 10000
bootstrap_means = np.array([
    np.mean(np.random.choice(sample, size=n, replace=True))
    for _ in range(B)
])

kde_mean = gaussian_kde(bootstrap_means)

plt.figure()
plt.plot(x_vals, clt_density)
plt.plot(x_vals, kde_mean(x_vals))
plt.title("Плотность среднего: ЦПТ vs Bootstrap")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.show()

# d) Bootstrap плотность асимметрии
bootstrap_skew = np.array([
    skew(np.random.choice(sample, size=n, replace=True), bias=False)
    for _ in range(B)
])

kde_skew = gaussian_kde(bootstrap_skew)

x_skew = np.linspace(min(bootstrap_skew), max(bootstrap_skew), 200)

plt.figure()
plt.plot(x_skew, kde_skew(x_skew))
plt.title("Bootstrap-плотность коэффициента асимметрии")
plt.xlabel("Skewness")
plt.ylabel("Плотность")
plt.show()

prob_skew_less_1 = np.mean(bootstrap_skew < 1)
print("\nP(асимметрия < 1) ≈", prob_skew_less_1)

# e) Плотность медианы (bootstrap)
bootstrap_medians = np.array([
    np.median(np.random.choice(sample, size=n, replace=True))
    for _ in range(B)
])

kde_median = gaussian_kde(bootstrap_medians)

x_median = np.linspace(min(bootstrap_medians), max(bootstrap_medians), 200)

plt.figure()
plt.plot(x_median, kde_median(x_median))
plt.title("Bootstrap-плотность медианы")
plt.xlabel("Median")
plt.ylabel("Плотность")
plt.show()