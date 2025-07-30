import numpy as np
import matplotlib.pyplot as plt
import time

# Точное значение площади
def exact_area():
    return np.sqrt(3)/4

# Оптимизированный метод прямоугольников (без создания больших массивов)
def optimized_rectangle_method(n):
    h = np.sqrt(3)/2
    dx = 1.0 / n
    dy = h / n
    area = 0.0
    
    # Разбиваем на блоки, чтобы избежать переполнения памяти
    block_size = 1000
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            # Создаем маленькие блоки
            x_blocks = min(block_size, n - i)
            y_blocks = min(block_size, n - j)
            
            x = np.linspace(i*dx, (i+x_blocks)*dx, x_blocks, endpoint=False) + dx/2
            y = np.linspace(j*dy, (j+y_blocks)*dy, y_blocks, endpoint=False) + dy/2
            
            X, Y = np.meshgrid(x, y)
            mask = (Y <= np.sqrt(3)*X) & (Y <= np.sqrt(3)*(1 - X)) & (Y <= h)
            area += np.sum(mask) * dx * dy
    
    return area

# Оптимизированный метод Монте-Карло
def optimized_monte_carlo(n):
    h = np.sqrt(3)/2
    x = np.random.rand(n)
    y = h * np.random.rand(n)
    inside = np.sum((y <= np.sqrt(3)*x) & (y <= np.sqrt(3)*(1 - x)))
    return (1 * h) * inside / n

# Параметры исследования (уменьшены для разумного времени выполнения)
n_values = [100, 500, 1000, 5000, 10000]
exact = exact_area()

# Измеряем производительность
def benchmark(method, n_values):
    times = []
    errors = []
    for n in n_values:
        start = time.perf_counter()
        area = method(n)
        times.append(time.perf_counter() - start)
        errors.append(abs(area - exact)/exact)
    return times, errors

print("Вычисление методом прямоугольников...")
rect_times, rect_errors = benchmark(optimized_rectangle_method, n_values)

print("Вычисление методом Монте-Карло...")
mc_times, mc_errors = benchmark(optimized_monte_carlo, n_values)

# Вывод результатов
print("\nТочное значение площади:", exact)
print("\nСравнение производительности:")
print(f"{'n':>8} {'Прям. время':>12} {'Прям. ошибка':>12} {'MC время':>12} {'MC ошибка':>12}")
for i, n in enumerate(n_values):
    print(f"{n:>8} {rect_times[i]:>12.4f} {rect_errors[i]:>12.2e} {mc_times[i]:>12.4f} {mc_errors[i]:>12.2e}")

# Построение графиков
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(n_values, rect_errors, 'o-', label='Метод прямоугольников')
plt.loglog(n_values, mc_errors, 's-', label='Метод Монте-Карло')
plt.xlabel('Количество операций')
plt.ylabel('Относительная ошибка')
plt.title('Точность методов')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.loglog(n_values, rect_times, 'o-', label='Метод прямоугольников')
plt.loglog(n_values, mc_times, 's-', label='Метод Монте-Карло')
plt.xlabel('Количество операций')
plt.ylabel('Время выполнения (с)')
plt.title('Производительность методов')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
