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
        "wait_after_write": 0.1,
        "buttons": {
            "start_practice": [742, 1347],
            "collect": [539, 1678],
            "collect_upper": [539, 1576],
            "continue": [1000, 145],
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
        self.handle = self.api.connect(self.emulator_path, self.instance)
        self._last_w = 0
        self._last_h = 0
        self._buf = None
        self._c_w = ctypes.c_int(0)
        self._c_h = ctypes.c_int(0)

    def _get_emulator_path(self):
        from mmumu.base import get_mumu_path
        path = get_mumu_path()
        if path:
            return path
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

        if self._buf is not None and self._last_w > 0 and self._last_h > 0:
            W, H = self._last_w, self._last_h
            self._c_w.value = W
            self._c_h.value = H
            result = self.api.capture_display(
                self.handle, 0, W * H * 4,
                ctypes.byref(self._c_w), ctypes.byref(self._c_h), self._buf
            )
            if result == 0 and self._c_w.value == W and self._c_h.value == H:
                return np.flipud(
                    np.frombuffer(self._buf, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
                )
            self._buf = None
            self._last_w = 0
            self._last_h = 0

        return self._grab_slow()

    def _grab_slow(self):
        w, h = ctypes.c_int(0), ctypes.c_int(0)
        result = self.api.capture_display(self.handle, 0, 0, ctypes.byref(w), ctypes.byref(h), None)
        if result != 0:
            return None
        W, H = w.value, h.value
        if W <= 0 or H <= 0:
            return None
        buf = (ctypes.c_ubyte * (W * H * 4))()
        result = self.api.capture_display(self.handle, 0, W * H * 4, ctypes.byref(w), ctypes.byref(h), buf)
        if result != 0:
            return None
        self._last_w = W
        self._last_h = H
        self._buf = buf
        return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)[:, :, :3])

# ==================== 数字定位（连通分量 ~2ms）====================
def locate_digits(img):
    h, w = img.shape[:2]
    y1 = int(h * 0.23)
    y2 = int(h * 0.37)
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)
    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    bw = (gray < 115).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
    glyphs = []
    for i in range(1, n):
        gx = int(stats[i, 0]); gy = int(stats[i, 1])
        gw = int(stats[i, 2]); gh = int(stats[i, 3])
        ga = int(stats[i, 4])
        if 45 <= gh <= 150 and 10 <= gw <= 95 and ga > 250:
            glyphs.append((gx + x1, gy + y1, gw, gh, ga))
    if not glyphs:
        return None, None
    glyphs.sort(key=lambda g: g[0])
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
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop
    bw = (gray < 120).astype(np.uint8)
    ys, xs = np.where(bw > 0)
    if ys.size == 0:
        return value
    g = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = g.shape
    if hh / max(1, ww) < 1.5:
        return value
    br = g[int(hh * 0.65):, int(ww * 0.55):]
    br_ink = br.mean() if br.size else 0.0
    if value == 7 and br_ink > 0.30:
        return 1
    if value == 1 and br_ink < 0.15:
        return 7
    return value

# ==================== PaddleOCR 识别（CPU/GPU 可选）====================
class PaddleDigitOCR:
    """使用 PaddleOCR 识别数字，支持 CPU/GPU 切换"""
    def __init__(self, use_gpu=False):
        self._ocr = None
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        if use_gpu:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            os.environ.setdefault("PADDLEFLAGS_GPU", "1")
            os.environ.setdefault("PADDLEFLAGS_USE_GLOBAL_POOL", "1")
            log_gpu = "[GPU] 已启用 GPU 加速 (CUDA)"
            device_param = "gpu"
            mkldnn = False
        else:
            log_gpu = "[CPU] 使用 CPU 运行"
            device_param = "cpu"
            mkldnn = True
        try:
            from paddleocr import PaddleOCR
            kwargs = dict(
                lang="en", device=device_param,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv4_mobile_det",
                text_recognition_model_name="PP-OCRv4_mobile_rec",
            )
            try:
                self._ocr = PaddleOCR(enable_mkldnn=mkldnn, **kwargs)
            except TypeError:
                self._ocr = PaddleOCR(**kwargs)
            print(log_gpu + (" [mkldnn]" if mkldnn else ""))
        except Exception as e:
            print(f"[OCR] PaddleOCR 加载失败：{e}")

    # ---------- 主入口 ----------
    def read_pair(self, img, roi1, roi2, strict_glyph=True):
        """读取两个数字区域。

        修复 17→17 误判：
          1. 先尝试合并预测（快路径）；
          2. 若合并结果的位数和字形数不匹配（例如右边只有 1 个字形却返回 17），
             则走"逐字形 OCR"兜底：把 ROI 里的每个字形单独送 OCR 再拼接。
          3. 若逐字形也失败，回退到单块识别。
        """
        def crop(roi):
            x1,y1,x2,y2 = [int(v) for v in roi]
            h,w = img.shape[:2]
            return img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        c1, c2 = crop(roi1), crop(roi2)

        # 候选 A：合并预测
        merged = self._read_pair_merged(c1, c2)

        if merged is not None:
            m1, m2 = merged
            if not strict_glyph:
                return fix_one_seven(c1, m1), fix_one_seven(c2, m2)
            if self._digits_match_glyphs(c1, m1) and self._digits_match_glyphs(c2, m2):
                return fix_one_seven(c1, m1), fix_one_seven(c2, m2)

        # 候选 B：逐字形 OCR（慢但准，专治单字形被读成多位数）
        if strict_glyph:
            g1 = self._read_by_glyphs(c1)
            g2 = self._read_by_glyphs(c2)
            if g1 is not None and g2 is not None:
                return fix_one_seven(c1, g1), fix_one_seven(c2, g2)

        # 候选 C：单块识别（最原始路径）
        n1 = self._read_one(c1)
        n2 = self._read_one(c2)

        # 如果合并也有结果，比较合理性，挑更好的
        if merged is not None:
            m1, m2 = merged
            m_score = int(self._digits_match_glyphs(c1, m1)) + int(self._digits_match_glyphs(c2, m2))
            s_score = int(self._digits_match_glyphs(c1, n1)) + int(self._digits_match_glyphs(c2, n2))
            if m_score > s_score:
                n1, n2 = m1, m2

        return fix_one_seven(c1, n1), fix_one_seven(c2, n2)

    # ---------- 合理性 & 逐字形 ----------
    def _digits_match_glyphs(self, crop, value):
        """OCR 结果的位数，和 ROI 中检测到的字形数是否吻合。"""
        if value is None:
            return False
        g = len(self._find_glyphs(crop))
        if g <= 0:
            return True   # 无法判断，放行
        return len(str(value)) == g

    @staticmethod
    def _find_glyphs(crop):
        """返回 crop 里"像数字"的连通分量 bbox 列表 [(x, y, w, h), ...]，按 x 排序。"""
        try:
            if crop is None or crop.size == 0:
                return []
            if crop.ndim == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            else:
                gray = crop
            bw = (gray < 120).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
            out = []
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                if h >= 20 and w >= 5 and a >= 50:
                    out.append((int(x), int(y), int(w), int(h)))
            out.sort(key=lambda g: g[0])
            return out
        except Exception:
            return []

    def _read_by_glyphs(self, crop):
        """逐字形 OCR：把 crop 按连通分量切成单字，逐个识别后拼成整数。

        单个字形不可能被识别成两位数，所以这条路径能根治
        "1 → 17" / "7 → 17" 之类的误判。
        """
        try:
            glyphs = self._find_glyphs(crop)
            if not glyphs:
                return None
            # 相邻字形若 x 范围重叠较多，视为同一数字的多连通分量（如 4、5）
            merged = []
            for g in glyphs:
                if merged:
                    px, py, pw, ph = merged[-1]
                    overlap_thr = min(px + pw, g[0] + g[2]) - max(px, g[0])
                    if overlap_thr > 0 and overlap_thr > 0.5 * min(pw, g[2]):
                        nx = min(px, g[0]); ny = min(py, g[1])
                        nx2 = max(px + pw, g[0] + g[2])
                        ny2 = max(py + ph, g[1] + g[3])
                        merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
                        continue
                merged.append(g)
            if not merged or len(merged) > 3:
                return None

            H, W = crop.shape[:2]
            digits = []
            for (x, y, w, h) in merged:
                x1 = max(0, x - 3); y1 = max(0, y - 3)
                x2 = min(W, x + w + 3); y2 = min(H, y + h + 3)
                sub = crop[y1:y2, x1:x2]
                d = self._read_one(sub)
                if d is None or not (0 <= d <= 9):
                    return None
                digits.append(str(d))
            return int("".join(digits))
        except Exception:
            return None

    # ---------- 底层 ----------
    def _read_pair_merged(self, c1, c2):
        """把左右两个 crop 水平拼接后一次 OCR，成功返回 (n1, n2)，否则 None。"""
        if c1 is None or c2 is None or c1.size == 0 or c2.size == 0:
            return None
        try:
            h1, _ = c1.shape[:2]
            h2, _ = c2.shape[:2]
            h = max(h1, h2)

            def pad_to(c, h):
                ch, _ = c.shape[:2]
                if ch == h:
                    return c
                pt = (h - ch) // 2
                pb = h - ch - pt
                if c.ndim == 3:
                    return np.pad(c, ((pt, pb), (0, 0), (0, 0)),
                                  mode="constant", constant_values=255)
                return np.pad(c, ((pt, pb), (0, 0)),
                              mode="constant", constant_values=255)

            c1p = pad_to(c1, h)
            c2p = pad_to(c2, h)
            if c1p.ndim == 3:
                gap = np.full((h, 40, c1p.shape[2]), 255, dtype=c1p.dtype)
            else:
                gap = np.full((h, 40), 255, dtype=c1p.dtype)
            merged = np.concatenate([c1p, gap, c2p], axis=1)

            res = self._ocr.predict(merged)
            if not res:
                return None
            d = res[0]
            texts = d.get("rec_texts", [])
            boxes = d.get("rec_boxes", [])
            if len(texts) != 2:
                return None
            nums = []
            for t in texts:
                digits = re.sub(r"\D", "", str(t))
                if not (1 <= len(digits) <= 3):
                    return None
                nums.append(int(digits))
            if len(boxes) == 2:
                order = sorted(
                    range(2),
                    key=lambda i: (float(boxes[i][0]) + float(boxes[i][2])) / 2,
                )
                nums = [nums[i] for i in order]
            return nums[0], nums[1]
        except Exception:
            return None

    def _read_one(self, crop):
        """识别单个数字（健壮版）"""
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
                        cx = (bx1 + bx2) / 2
                        cy = (by1 + by2) / 2
                        out.append((t, x1 + cx, y1 + cy))
        except Exception:
            pass
        return out

    def read_text(self, img, roi):
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
        def crop(roi):
            x1,y1,x2,y2 = [int(v) for v in roi]
            h,w = img.shape[:2]
            return img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        t1 = self._read_text(crop(roi1))
        t2 = self._read_text(crop(roi2))
        return t1, t2

    def _read_text(self, crop):
        try:
            res = self._ocr.predict(crop)
            if res:
                texts = "".join(res[0].get("rec_texts", []))
                return texts if texts else None
        except Exception:
            pass
        return None

# ==================== 有机手写（仿真笔迹，防检测）====================
class SymbolWriter:

    def __init__(self, device=None, font_file=None, speed=4.0):
        from mtc.maatouch import MaaTouch
        if device is None:
            device = DEVICE
        try:
            from adbutils import adb
            if device not in [d.serial for d in adb.device_list()]:
                adb.connect(device)
        except Exception:
            pass
        self.touch = MaaTouch(device)
        self.cx, self.cy = WRITE_CENTER_X, WRITE_CENTER_Y
        self.SPEED = speed

    @staticmethod
    def _jline(p1, p2, bow=0.0, steps=9):
        pts = []
        for i in range(1, steps):
            t = i / steps
            x = p1[0] + (p2[0] - p1[0]) * t + random.uniform(-14, 14)
            y = p1[1] + (p2[1] - p1[1]) * t + bow * 4.0 * t * (1 - t) + random.uniform(-14, 14)
            pts.append((x, y))
        return pts

    def _stroke(self, pts, duration):
        path = [(int(x), int(y)) for (x, y) in pts]
        self.touch.swipe(path, duration)

    def write(self, sym):
        cx = self.cx + random.randint(-14, 14)
        cy = self.cy + random.randint(-14, 14)
        if sym in (">", "<"):
            L = random.randint(200, 240)
            alpha = random.uniform(25, 45)
            d = int(L * math.cos(math.radians(alpha)))
            h = int(L * math.sin(math.radians(alpha)))
            if sym == ">":
                A, B, V = (cx - d, cy - h), (cx - d, cy + h), (cx + d, cy)
            else:
                A, B, V = (cx + d, cy - h), (cx + d, cy + h), (cx - d, cy)
            if random.random() < 0.5:
                A, B = B, A
            pts = [A] + self._jline(A, V, random.uniform(-10, 10)) \
                  + [V] + self._jline(V, B, random.uniform(-10, 10)) + [B]
            dur = int((25 + len(pts) * 6 * random.uniform(0.85, 1.25)) * self.SPEED)
            self._stroke(pts, dur)
        elif sym == "=":
            L = random.randint(190, 230)
            y1 = cy + random.randint(-200, -50)
            y2 = y1 + random.randint(50, 200)
            for y in (y1, y2):
                if random.random() < 0.5:
                    p1, p2 = (cx - L, y), (cx + L, y)
                else:
                    p1, p2 = (cx + L, y), (cx - L, y)
                pts = [p1] + self._jline(p1, p2, random.uniform(-8, 8)) + [p2]
                self._stroke(pts, int((180 + len(pts) * 15 * random.uniform(0.85, 1.25)) * self.SPEED))
                time.sleep(0.12 * self.SPEED)
# 本文件是核心模块（截图/数字定位/OCR/手写），由 run.py 调用，不再单独运行。
