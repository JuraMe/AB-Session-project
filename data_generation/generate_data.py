"""
Генерация данных для сессионного задания: внедрение нового алгоритма
назначения курьеров и расчёта обещанного времени доставки (ETA)
в сервисе экспресс-доставки продуктов.

Сценарий
--------
* 20 городов трёх страт: large (6), medium (7), small (7).
* 2 марта – 28 июня 2026 (17 недель). Алгоритм включают 4 мая 2026
  в 8 пилотных городах (3 large, 3 medium, 2 small). Пилотные города выбирал
  бизнес, а не случайный механизм: крупных городов среди них больше.
* Рандомизировать пользователей нельзя: курьеры общие на весь город,
  и алгоритм для части пользователей влияет на остальных (нарушение SUTVA).

Что заложено в данные (это и нужно найти в анализе)
---------------------------------------------------
1. Недельная сезонность (пятница–воскресенье выше), общий весенний рост,
   небольшой летний спад в июне.
2. Майские праздники (1–3 и 9–11 мая): спад конверсии во всех городах,
   сильнее в крупных. Вторые праздники попадают в пост-период, а крупных
   городов больше среди пилотных — простой DiD без учёта этого смещён.
3. Истинный эффект на конверсию разный по стратам:
   large +1.2 п.п., medium +0.8 п.п., small +0.2 п.п.,
   с «разгоном» в первую неделю (от 30% до 100% эффекта).
4. Нарушение параллельных трендов: в пилотном городе M02 за 14 дней до запуска
   стартовала локальная промо-акция (+0.6 п.п. к конверсии до конца периода).
5. Guardrail: алгоритм обещает более короткий ETA, но в малых городах курьеров
   мало, поэтому растут время доставки, доля опозданий и отмен.
6. Проблемы качества данных: в контрольном городе S07 нет логов за 5 дней,
   в контрольном городе L04 за один день заказы задвоены (дубли в логах).

Файлы
-----
data/city_daily.csv   — панель «город × день»
data/users_sample.csv — выборка пользователей по городам до и после запуска
                        (для проверки баланса и стабильности состава)

Запуск:  python data_generation/generate_data.py [seed]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 4051
rng = np.random.default_rng(SEED)

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# ----- Календарь -----
START, END = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-06-28")
LAUNCH = pd.Timestamp("2026-05-04")
dates = pd.date_range(START, END, freq="D")
HOLIDAYS = pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03",
                           "2026-05-09", "2026-05-10", "2026-05-11"])

# ----- Города -----
cities = (
    [("L%02d" % i, "large") for i in range(1, 7)]
    + [("M%02d" % i, "medium") for i in range(1, 8)]
    + [("S%02d" % i, "small") for i in range(1, 8)]
)
TREATED = {"L01", "L02", "L03", "M01", "M02", "M03", "S01", "S02"}

STRATUM = {
    #          DAU (тыс.)   база конв.  доставка, мин  опоздания  отмены
    "large":  dict(dau=(40, 80), conv=0.130, time=28.0, late=0.065, cancel=0.024,
                   holiday=-0.015, effect=0.012, d_time=-0.5, d_late=0.002, d_cancel=0.000),
    "medium": dict(dau=(10, 25), conv=0.110, time=25.0, late=0.070, cancel=0.025,
                   holiday=-0.010, effect=0.008, d_time=0.6, d_late=0.006, d_cancel=0.001),
    "small":  dict(dau=(2, 6), conv=0.090, time=22.0, late=0.075, cancel=0.027,
                   holiday=-0.006, effect=0.002, d_time=4.0, d_late=0.035, d_cancel=0.008),
}
# Прирост конверсии по дням недели (пн=0 ... вс=6)
WEEKDAY = np.array([-0.004, -0.003, -0.002, 0.000, 0.008, 0.012, 0.010])

t = np.arange(len(dates))
common_trend = 0.00012 * t                                   # весенний рост
june_dip = np.where(dates >= "2026-06-01", -0.004, 0.0)      # дачный сезон
is_holiday = dates.isin(HOLIDAYS)
days_since = (dates - LAUNCH).days.values
ramp = np.clip(0.3 + 0.1 * days_since, 0.3, 1.0)             # разгон эффекта за ~7 дней

rows = []
for city, stratum in cities:
    p = STRATUM[stratum]
    treated = city in TREATED
    post = (dates >= LAUNCH).astype(int)
    d = treated * post * ramp

    dau_base = rng.uniform(*p["dau"]) * 1000
    city_conv = p["conv"] + rng.normal(0, 0.008)
    city_slope = rng.normal(0, 0.00001)                      # маленький собственный тренд

    # AR(1)-шум конверсии: дни внутри города зависимы
    eps = np.zeros(len(dates))
    for i in range(1, len(dates)):
        eps[i] = 0.5 * eps[i - 1] + rng.normal(0, 0.0015)

    conv = (city_conv + common_trend + june_dip + city_slope * t
            + WEEKDAY[dates.weekday] + is_holiday * p["holiday"] + eps
            + d * p["effect"])
    if city == "M02":   # локальная промо-акция, не связанная с алгоритмом
        conv += np.where(dates >= LAUNCH - pd.Timedelta(days=14), 0.006, 0.0)
    conv = np.clip(conv, 0.01, 0.5)

    dau = (dau_base * (1 + 0.0008 * t) * (1 + 0.06 * (dates.weekday >= 4))
           * (1 - 0.12 * is_holiday) * rng.normal(1, 0.03, len(dates)))
    dau = np.round(dau).astype(int)
    orders = rng.binomial(dau, conv)

    aov = rng.normal(1450 if stratum == "large" else 1300, 25, len(dates))

    time_eps = rng.normal(0, 0.8, len(dates))
    delivery = (p["time"] + rng.normal(0, 1.0) + 1.5 * (dates.weekday >= 4)
                + 2.0 * is_holiday + time_eps + d * p["d_time"])
    late_p = np.clip(p["late"] + rng.normal(0, 0.006) + 0.01 * (dates.weekday >= 4)
                     + d * p["d_late"], 0.005, 0.5)
    late_orders = rng.binomial(orders, late_p)
    cancel_p = np.clip(p["cancel"] + rng.normal(0, 0.002) + d * p["d_cancel"], 0.001, 0.3)
    cancelled = rng.binomial(orders, cancel_p)

    rows.append(pd.DataFrame({
        "date": dates, "city": city, "stratum": stratum, "treated": int(treated),
        "active_users": dau, "orders": orders, "revenue": np.round(orders * aov),
        "avg_delivery_min": np.round(delivery, 2), "late_orders": late_orders,
        "cancelled_orders": cancelled,
    }))

panel = pd.concat(rows, ignore_index=True)

# ----- Проблемы качества данных -----
outage = (panel["city"] == "S07") & panel["date"].between("2026-03-23", "2026-03-27")
panel = panel[~outage]
dup = (panel["city"] == "L04") & (panel["date"] == "2026-04-15")
for col in ["orders", "revenue", "late_orders", "cancelled_orders"]:
    panel.loc[dup, col] *= 2

panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
panel.to_csv(OUT_DIR / "city_daily.csv", index=False)

# ----- Выборка пользователей (pre: март–апрель, post: май–июнь) -----
users = []
uid = 0
for city, stratum in cities:
    for period in ["pre", "post"]:
        n = 1500
        big = stratum == "large"
        age = rng.normal(33 if big else 37, 9, n).clip(18, 75).round()
        tenure = rng.gamma(2.0, 6 if big else 5, n).round(1)
        ios = rng.binomial(1, 0.48 if big else 0.36, n)
        orders_30d = rng.poisson(3.2 if big else 2.6, n)
        new_user = rng.binomial(1, 0.12, n)
        users.append(pd.DataFrame({
            "user_id": np.arange(uid, uid + n), "city": city, "stratum": stratum,
            "treated": int(city in TREATED), "period": period, "age": age,
            "tenure_months": tenure, "is_ios": ios,
            "orders_last_30d": orders_30d, "is_new": new_user,
        }))
        uid += n
pd.concat(users, ignore_index=True).to_csv(OUT_DIR / "users_sample.csv", index=False)

print(f"seed={SEED}: {len(panel)} строк панели, {uid} пользователей -> {OUT_DIR}")
