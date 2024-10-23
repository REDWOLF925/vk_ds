solution.ipynb включает визуализацию трейна и обучение модели

===========================================================================

prod.ipynb применяет модель на тестовых данных

===========================================================================

в /models лежит сама модель, в /data train и test датасеты, в /other лежит скрипт, содержащий функции, 
необходимые для обработки данных

===========================================================================

submission.csv - ответ на задачу

===========================================================================

сгенерированные признаки:

•Sum -- сумма всех значений ряда

•Max -- максимальное значение ряда

•Var -- дисперсия

•Mean -- среднее значение

•Len -- количество значений ряда

•Inflow -- общий "приток", сумма положительных разностей 

между двумя соседними значениям

•Outlow -- общий "отток", модуль суммы отрицательных разностей 

между двумя соседними значениям

•Static_count -- количество разностей между соседними значениями <= 1

•Trend -- коэффициент угла наклона линейной регрессии

•Bias -- смещение линии регрессии

•A, D -- коэффициенты аппроксимирующего полинома 3 степени:

A + B*x + C*x^2 + D*x^3, оставшиеся коэффициенты были отброшены

после отбора по SHAP_values
