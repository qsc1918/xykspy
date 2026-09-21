#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes, json, math, os, time, sys, re, random
import numpy as np, cv2

MUMU_PATH = r"C:\Program Files\Netease\MuMu"
MUMU_DLL = MUMU_PATH + r"\nx_device\12.0\shell\sdk\external_renderer_ipc.dll"
MUMU_INSTANCE = int(os.environ.get("MATH_AUTO_INSTANCE", "1"))
DEVICE = os.environ.get("MATH_AUTO_DEVICE", "127.0.0.1:16416")
WORKDIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(WORKDIR, "config.json")

def _load_config():
    cfg = {
        "left_roi": [240, 455, 455, 605],
        "right_roi": [625, 455, 855, 605],
        "write_center": [540, 1300],
        "interval": 1.0,
        "wait_after_write": 0.25,
        "buttons": {
            "start_practice": [742, 1347],
            "collect": [539, 1678],
            "collect_upper": [539, 1576],
            "continue": [805, 1833],
            "dialog_decline": [217, 1745],
            "dialog_claim": [725, 1745],
            "continue_pk": [568, 1657],
        },
        "bottom_band_y": 1150,
        "circle_center": 540,
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

CFG = _load_config()
LEFT_ROI = tuple(CFG["left_roi"])
RIGHT_ROI = tuple(CFG["right_roi"])
WRITE_CENTER_X, WRITE_CENTER_Y = CFG["write_center"]
WAIT_AFTER_WRITE = CFG.get("wait_after_write", 0.5)

# ==================== 截图（MuMu 原生 IPC）====================
class MuMuScreen:
    def __init__(self, instance=1):
        from mmumu.api import MuMuApi
        from mmumu.base import get_mumu_path
        self.api = MuMuApi(MUMU_DLL)
        self.instance = instance
        self.handle = None
        self.emulator_path = self._get_emulator_path()
        # 修复：原代码 if self.handle 恒为 False（handle 刚被设为 None），
        # 导致永远走 else 分支用写死的 MUMU_PATH，_get_emulator_path() 白算了。
        self.handle = self.api.connect(self.emulator_path, self.instance)

    def _get_emulator_path(self):
        from mmumu.base import get_mumu_path
        path = get_mumu_path()
        if path:
            return path
        # 尝试常见路径
        common_paths = [
            r"C:\Program Files\Netease\MuMu",
            r"C:\Program Files\Netease\MuMuPlayer-12.0",
            r"D:\Program Files\Netease\MuMuPlayer-12.0",
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
        return MUMU_PATH

    def grab(self):
        if self.handle is None:
            self.handle = self.api.connect(self.emulator_path, self.instance)
        w, h = ctypes.c_int(0), ctypes.c_int(0)
        result = self.api.capture_display(self.handle, 0, 0, ctypes.byref(w), ctypes.byref(h), None)
        if result != 0:
            return None
        W, H = w.value, h.value
        buf = (ctypes.c_ubyte * (W * H * 4))()
        result = self.api.capture_display(self.handle, 0, W * H * 4, ctypes.byref(w), ctypes.byref(h), buf)
        if result != 0:
            return None
        return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)[:, :, :3])

# ==================== 数字定位（连通分量 ~2ms）====================
def locate_digits(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw = (gray < 115).astype(np.uint8)
    # 只扫黄色卡片区域 y 23%~37%：数字在卡片内（y≈460~590），
    # 卡片下方那行灰色"上一题"文字（y≈720 起）被排除在外。
    zone = np.zeros_like(bw)
    zone[int(h*0.23):int(h*0.37), int(w*0.05):int(w*0.95)] = bw[int(h*0.23):int(h*0.37), int(w*0.05):int(w*0.95)]
    n, lab, stats, cents = cv2.connectedComponentsWithStats(zone, 8)
    glyphs = [(int(stats[i,0]), int(stats[i,1]), int(stats[i,2]), int(stats[i,3]), int(stats[i,4]))
              for i in range(1,n) if 45<=stats[i,3]<=150 and 10<=stats[i,2]<=95 and stats[i,4]>250]
    if not glyphs:
        return None, None
    glyphs.sort(key=lambda g: g[0])
    # 锚点 = 中间白圈中心（固定 x=540）。? 可见时它在圈内，? 消失时空圈位置不变，
    # 所以固定锚点两种情况都成立，不需要再去猜"哪个字是?"。
    ax = CFG.get("circle_center", 540)
    left  = [g for g in glyphs if (g[0]+g[2]/2) < ax - 110]
    right = [g for g in glyphs if (g[0]+g[2]/2) > ax + 110]
    if not left or not right:
        return None, None
    def bbox(gs):
        return min(g[0] for g in gs), min(g[1] for g in gs), max(g[0]+g[2] for g in gs), max(g[1]+g[3] for g in gs)
    lx1,ly1,lx2,ly2 = bbox(left)
    rx1,ry1,rx2,ry2 = bbox(right)
    p=22; q=16
    return ((max(0,lx1-p),max(0,ly1-q),min(w,lx2+p),min(h,ly2+q)),
            (max(0,rx1-p),max(0,ry1-q),min(w,rx2+p),min(h,ry2+q)))

def fix_one_seven(crop, value):
    """1/7 几何交叉校验（纯 cv2，几毫秒）：
    该字体下 '1' 右侧是整高竖笔 → 右下角有墨迹；'7' 顶横+斜线 → 右下角空白。
    OCR 把两者搞混时用它纠偏；其他数字一律原样返回。"""
    if value is None or not (0 <= value <= 9) or crop is None or crop.size == 0:
        return value
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    bw = (gray < 120).astype(np.uint8)
    ys, xs = np.where(bw > 0)
    if ys.size == 0:
        return value
    g = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = g.shape
    if hh / max(1, ww) < 1.5:          # 矮胖的不可能是 1/7，跳过
        return value
    br = g[int(hh * 0.65):, int(ww * 0.55):]
    br_ink = br.mean() if br.size else 0.0
    if value == 7 and br_ink > 0.30:   # 右下满墨 → 其实是 1
        return 1
    if value == 1 and br_ink < 0.15:   # 右下空白 → 其实是 7
        return 7
    return value

# ==================== PaddleOCR 识别（CPU/GPU 可选）====================
class PaddleDigitOCR:
    """使用 PaddleOCR 识别数字，支持 CPU/GPU 切换"""
    def __init__(self, use_gpu=False):
        self._ocr = None
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # CPU/GPU 配置
        if use_gpu:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            os.environ.setdefault("PADDLEFLAGS_GPU", "1")
            os.environ.setdefault("PADDLEFLAGS_USE_GLOBAL_POOL", "1")
            log_gpu = "[GPU] 已启用 GPU 加速 (CUDA)"
            device_param = "gpu"
        else:
            log_gpu = "[CPU] 使用 CPU 运行"
            device_param = "cpu"
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                lang="en", device=device_param, enable_mkldnn=False,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv4_mobile_det",
                text_recognition_model_name="PP-OCRv4_mobile_rec",
            )
            print(log_gpu)
        except Exception as e:
            print(f"[OCR] PaddleOCR 加载失败：{e}")

    def read_pair(self, img, roi1, roi2):
        """读取两个数字区域"""
        def crop(roi):
            x1,y1,x2,y2 = [int(v) for v in roi]
            h,w = img.shape[:2]
            return img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        c1, c2 = crop(roi1), crop(roi2)
        n1 = self._read_one(c1)
        n2 = self._read_one(c2)
        # 单位数做 1/7 几何交叉校验（防止把 1 写成 7，题就错了）
        return fix_one_seven(c1, n1), fix_one_seven(c2, n2)

    def _read_one(self, crop):
        """识别单个数字（健壮版）

        原代码直接 int(texts[0])：OCR 只要多认出一个空格、小数点或杂字
        （如 "1 0 0"、"100."）就抛异常 → 返回 None → 该题卡死。
        现在先剥掉所有非数字字符再校验位数（本题目数字范围 0~99，放宽到 3 位）。
        """
        if crop is None or crop.size == 0:
            return None
        try:
            res = self._ocr.predict(crop)
            if res:
                texts = res[0].get("rec_texts", [])
                if texts:
                    digits = re.sub(r"\D", "", "".join(texts))
                    if 1 <= len(digits) <= 3:
                        return int(digits)
        except Exception:
            pass
        return None

    def read_boxes(self, img, roi):
        """读取区域里每行文字及其位置框，返回 [(text, cx, cy), ...]（坐标是整图坐标）。
        PaddleOCR 3.x 的 rec_boxes 是 [x1, y1, x2, y2]（左上角 + 右下角）。"""
        out = []
        try:
            x1, y1, x2, y2 = [int(v) for v in roi]
            H, W = img.shape[:2]
            crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            res = self._ocr.predict(crop)
            if res:
                d = res[0]
                texts = d.get("rec_texts", [])
                boxes = d.get("rec_boxes", [])
                for t, b in zip(texts, boxes):
                    t = str(t).strip()
                    if t and len(b) == 4:
                        bx1, by1, bx2, by2 = [float(v) for v in b]
                        cx = (bx1 + bx2) / 2          # ← 关键：用 (x1+x2)/2
                        cy = (by1 + by2) / 2          # ← 关键：用 (y1+y2)/2
                        out.append((t, x1 + cx, y1 + cy))
        except Exception:
            pass
        return out

    def read_text(self, img, roi):
        """读取区域文字（用于弹窗识别）"""
        try:
            x1,y1,x2,y2 = [int(v) for v in roi]
            h,w = img.shape[:2]
            crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            res = self._ocr.predict(crop)
            if res:
                texts = "".join(res[0].get("rec_texts", []))
                if texts:
                    return texts
        except Exception:
            pass
        return ""

    def read_pair_text(self, img, roi1, roi2):
        """读取两个区域的文字"""
        def crop(roi):
            x1,y1,x2,y2 = [int(v) for v in roi]
            h,w = img.shape[:2]
            return img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        t1 = self._read_text(crop(roi1))
        t2 = self._read_text(crop(roi2))
        return t1, t2

    def _read_text(self, crop):
        """读取区域文字"""
        try:
            res = self._ocr.predict(crop)
            if res:
                texts = "".join(res[0].get("rec_texts", []))
                return texts if texts else None
        except Exception:
            pass
        return None

# ==================== 有机手写（仿真笔迹，防检测）====================
# 原理: MaaTouch 通过 adb socket 注入, swipe(点列表) 一次完成
#       down -> move*N -> up, 中间不抬笔, 画出连续一笔。
# 仿真要素: 起笔偏移±14px / 尺寸随机 / 二次弧度 / 逐点抖动 / 时长随机。
class SymbolWriter:

    def __init__(self, device=None, font_file=None, speed=4.0):
        from mtc.maatouch import MaaTouch
        if device is None:
            device = DEVICE
        # MuMu 的 adb 端口需主动 connect 才会出现在设备列表
        try:
            from adbutils import adb
            if device not in [d.serial for d in adb.device_list()]:
                adb.connect(device)
        except Exception:
            pass
        self.touch = MaaTouch(device)
        self.cx, self.cy = WRITE_CENTER_X, WRITE_CENTER_Y
        self.SPEED = speed

    # ---- 轨迹生成 ----
    @staticmethod
    def _jline(p1, p2, bow=0.0, steps=9):
        """两点间 steps 个插值点 + 二次弧度 bow + 逐点抖动(±14px)。
        抖动让线条不是完美直线——太直会被判机器。"""
        pts = []
        for i in range(1, steps):
            t = i / steps
            x = p1[0] + (p2[0] - p1[0]) * t + random.uniform(-14, 14)
            y = p1[1] + (p2[1] - p1[1]) * t + bow * 4.0 * t * (1 - t) + random.uniform(-14, 14)
            pts.append((x, y))
        return pts

    def _stroke(self, pts, duration):
        """一笔画完: MaaTouch.swipe = down -> move*N -> up, 末尾自动抬笔。
        抬笔+停顿是 app 识别并进入下一题的关键，所以每个符号/横笔都单独 swipe。"""
        path = [(int(x), int(y)) for (x, y) in pts]
        self.touch.swipe(path, duration)

    # ---- 对外接口 ----
    def write(self, sym):
        cx = self.cx + random.randint(-14, 14)   # 起笔位置随机 ±14px
        cy = self.cy + random.randint(-14, 14)
        if sym in (">", "<"):
            L = random.randint(200, 240)                 # 每条边 200~240px（太小会被判机器）
            alpha = random.uniform(25, 45)               # 半张角 25°~45°（张角随机）
            d = int(L * math.cos(math.radians(alpha)))   # 水平深度
            h = int(L * math.sin(math.radians(alpha)))   # 半高（上下各 h）
            if sym == ">":
                A, B, V = (cx - d, cy - h), (cx - d, cy + h), (cx + d, cy)
            else:
                A, B, V = (cx + d, cy - h), (cx + d, cy + h), (cx - d, cy)
            if random.random() < 0.5:                   # 随机从哪端起笔
                A, B = B, A
            pts = [A] + self._jline(A, V, random.uniform(-10, 10)) \
                  + [V] + self._jline(V, B, random.uniform(-10, 10)) + [B]
            # 原速 ≈ 150ms，乘 SPEED 后 ≈ 600ms（SPEED=4.0）
            dur = int((25 + len(pts) * 6 * random.uniform(0.85, 1.25)) * self.SPEED)
            self._stroke(pts, dur)
        elif sym == "=":
            L = random.randint(190, 230)                 # 两横各 190~230px
            y1 = cy + random.randint(-200, -50)           # 上下两横间距随机
            y2 = y1 + random.randint(50, 200)
            for y in (y1, y2):
                if random.random() < 0.5:
                    p1, p2 = (cx - L, y), (cx + L, y)    # 随机从左或从右起笔
                else:
                    p1, p2 = (cx + L, y), (cx - L, y)
                pts = [p1] + self._jline(p1, p2, random.uniform(-8, 8)) + [p2]
                # 原速 ≈ 310ms/横，乘 SPEED 后 ≈ 1.2s/横（SPEED=4.0）
                self._stroke(pts, int((180 + len(pts) * 15 * random.uniform(0.85, 1.25)) * self.SPEED))
                time.sleep(0.12 * self.SPEED)            # 两横之间抬笔停顿，同步放大
# 本文件是核心模块（截图/数字定位/OCR/手写），由 run_final.py（练习）
# 和 run_pk.py（PK）两个入口脚本调用，不再单独运行。