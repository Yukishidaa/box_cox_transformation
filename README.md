# Box-Cox Transformation Script

## Описание

Этот скрипт выполняет преобразование Бокса-Кокса (Box-Cox) для числового признака с наибольшей асимметрией (skewness) в наборе данных диабета (`diabetes dataset` из `sklearn`). Скрипт также строит гистограммы признака до и после преобразования и выводит показатели асимметрии.

## Функции

### `find_most_skewed_positive_feature(df)`
Находит наиболее асимметричный положительный признак.

**Аргументы:**
- `df` (`pandas.DataFrame`): DataFrame с числовыми признаками.

**Возвращает:**
- `str`: название наиболее асимметричного положительного признака.

### `shift_to_positive(df)`
Сдвигает все значения DataFrame, чтобы они стали положительными (необходимо для Box-Cox).

**Аргументы:**
- `df` (`pandas.DataFrame`): исходный DataFrame.

**Возвращает:**
- `pandas.DataFrame`: сдвинутый DataFrame.

### `apply_boxcox(df, feature)`
Применяет Box-Cox преобразование к выбранному признаку.

**Аргументы:**
- `df` (`pandas.DataFrame`): DataFrame с данными.
- `feature` (`str`): название признака.

**Возвращает:**
- `tuple`: (преобразованные значения как `numpy.ndarray`, оптимальное значение лямбда).

### `plot_before_after(original, transformed, feature)`
Строит гистограммы признака до и после Box-Cox.

**Аргументы:**
- `original` (array-like): исходные значения признака.
- `transformed` (array-like): значения после Box-Cox.
- `feature` (`str`): название признака.

**Возвращает:**
- None

## Использование

1. Создать виртуальное окружение и активировать его:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Запустить скрипт:

```bash
python boxcox_script.py
```

- Скрипт автоматически найдет наиболее асимметричный положительный признак.
- Выведет его название, оптимальное лямбда для Box-Cox, асимметрию до и после преобразования.
- Построит гистограммы до и после Box-Cox.

## Пример работы

![Две гистограммы для сравнения распределения](/box_cox_transformation/res.png)

## Зависимости

- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

## Замечания

- Box-Cox работает только с положительными числами. Скрипт автоматически сдвигает данные для этого.