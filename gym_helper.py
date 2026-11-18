"""
Gym helper для Escape from Tarkov.

Читает пиксели экрана (без инжекта и без доступа к памяти игры), находит
"зону" (фигуру) и "маркер", и нажимает кнопку, когда маркер входит в зону.
Точка нажатия каждый раз случайная: не идеальный центр, но и не край.

Установка (Windows, игра в Borderless/Windowed):
    pip install mss numpy opencv-python pydirectinput keyboard

Использование:
    1) python gym_helper.py calibrate
       - откройте игру на экране с мини-игрой, у вас 5 секунд
       - выделите область мини-игры мышью (ENTER), затем кликните по цвету зоны,
         потом по цвету маркера
    2) python gym_helper.py run [--debug]
       - F7 включить/выключить, F9 выход
"""
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).with_name("gym_config.json")

DEFAULTS = {
    "roi": None,              # [left, top, width, height]
    "zone_bgr": None,         # цвет зоны (фигуры)
    "marker_bgr": None,       # цвет движущегося маркера
    "tolerance": 40,          # допуск по каждому каналу цвета
    "min_area": 15,           # минимум пикселей для детекта
    "hit_depth_min": 0.20,    # насколько глубоко в зону жать: доля радиуса от центра
    "hit_depth_max": 0.60,    # 0 = ровно центр, 1 = самый край
    "latency_ms": 30,         # компенсация задержки (захват + ввод + игра)
    "cooldown_s": 0.8,        # пауза после нажатия
    "action": "key:space",    # "key:space", "key:e" или "mouse"
}


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def grab_full():
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[1]
        img = np.array(sct.grab(mon))[:, :, :3]
    return img


def calibrate():
    print("Переключитесь в игру, мини-игра должна быть на экране. Скриншот через 5 сек...")
    time.sleep(5)
    full = grab_full()

    print("Выделите область мини-игры мышью и нажмите ENTER.")
    x, y, w, h = cv2.selectROI("ROI", full, showCrosshair=False)
    cv2.destroyWindow("ROI")
    if w == 0 or h == 0:
        print("Область не выбрана.")
        return
    crop = full[y:y + h, x:x + w].copy()

    clicks = []

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append(tuple(int(c) for c in crop[my, mx]))

    cv2.namedWindow("pick")
    cv2.setMouseCallback("pick", on_mouse)
    for label in ("ЗОНЫ (фигуры)", "МАРКЕРА"):
        print(f"Кликните по цвету {label}.")
        n = len(clicks)
        while len(clicks) == n:
            cv2.imshow("pick", crop)
            if cv2.waitKey(30) == 27:
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()

    cfg = load_config()
    cfg["roi"] = [int(x), int(y), int(w), int(h)]
    cfg["zone_bgr"], cfg["marker_bgr"] = list(clicks[0]), list(clicks[1])
    save_config(cfg)
    print("Сохранено в", CONFIG_PATH)


def make_mask(img, bgr, tol):
    c = np.array(bgr, dtype=np.int16)
    lo = np.clip(c - tol, 0, 255).astype(np.uint8)
    hi = np.clip(c + tol, 0, 255).astype(np.uint8)
    return cv2.inRange(img, lo, hi)


def centroid(mask):
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return None, 0
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), m["m00"]


def press(action):
    import pydirectinput
    pydirectinput.PAUSE = 0
    if action == "mouse":
        pydirectinput.click()
    else:
        key = action.split(":", 1)[1]
        pydirectinput.press(key)


def run(debug=False):
    import keyboard
    import mss

    cfg = load_config()
    if not (cfg["roi"] and cfg["zone_bgr"] and cfg["marker_bgr"]):
        print("Сначала выполните: python gym_helper.py calibrate")
        return
    left, top, w, h = cfg["roi"]
    region = {"left": left, "top": top, "width": w, "height": h}

    enabled = False
    print("F7 вкл/выкл, F9 выход.")

    def toggle():
        nonlocal enabled
        enabled = not enabled
        print("ВКЛ" if enabled else "ВЫКЛ")

    keyboard.add_hotkey("f7", toggle)

    depth = random.uniform(cfg["hit_depth_min"], cfg["hit_depth_max"])
    last_pos, last_t, last_press = None, None, 0.0

    with mss.mss() as sct:
        while not keyboard.is_pressed("f9"):
            if not enabled:
                time.sleep(0.02)
                continue

            img = np.array(sct.grab(region))[:, :, :3]
            now = time.perf_counter()
            zmask = make_mask(img, cfg["zone_bgr"], cfg["tolerance"])
            mmask = make_mask(img, cfg["marker_bgr"], cfg["tolerance"])
            zc, zarea = centroid(zmask)
            mc, marea = centroid(mmask)

            if zc is None or mc is None or zarea < cfg["min_area"] or marea < cfg["min_area"]:
                last_pos = None
            else:
                # прогноз положения маркера с учётом задержки
                pred = mc
                if last_pos is not None and now > last_t:
                    vel = (mc - last_pos) / (now - last_t)
                    pred = mc + vel * (cfg["latency_ms"] / 1000.0)
                last_pos, last_t = mc, now

                radius = (zarea / np.pi) ** 0.5
                dist = float(np.linalg.norm(pred - zc))

                if dist <= depth * radius and now - last_press > cfg["cooldown_s"]:
                    press(cfg["action"])
                    last_press = now
                    depth = random.uniform(cfg["hit_depth_min"], cfg["hit_depth_max"])
                    print(f"hit: dist/radius={dist / radius:.2f}, следующая глубина={depth:.2f}")

            if debug:
                vis = np.hstack([img, cv2.cvtColor(zmask, cv2.COLOR_GRAY2BGR),
                                 cv2.cvtColor(mmask, cv2.COLOR_GRAY2BGR)])
                cv2.imshow("debug (img | zone | marker)", vis)
                cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "calibrate":
        calibrate()
    elif cmd == "run":
        run(debug="--debug" in sys.argv)
    else:
        print(__doc__)
