"""Общие ориентиры М2; фактические точки русла записаны в data/s1/index.csv."""
LENA_POINTS = {
    "lensk": ("Ленск", 60.72, 114.92),
    "olekminsk": ("Олёкминск", 60.38, 120.42),
    "pokrovsk": ("Покровск", 61.48, 129.15),
    "yakutsk_crossing": ("Переправа Якутск — Нижний Бестях", 62.00, 129.82),
    "sangar": ("Сангар", 63.92, 127.47),
    "zhigansk": ("Жиганск", 66.77, 123.37),
    "kyusyur": ("Кюсюр", 70.68, 127.37),
}
LENA_NAMES = {key: point[0] for key, point in LENA_POINTS.items()}
