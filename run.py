#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import sys
import threading
import time

import cv2

import xyks as m

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

WORKDIR = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_instances(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _roi_hash(img, L, R):
    parts = []
    H, W = img.shape[:2]
    for roi in (L, R):
        try:
            x1, y1, x2, y2 = [int(v) for v in roi]
        except Exception:
            parts.append(b"")
            continue
        x1 = max(0, min(W, x1)); y1 = max(0, min(H, y1))
        x2 = max(0, min(W, x2)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            parts.append(b"")
            continue
        crop = img[y1:y2, x1:x2]
        try:
            small = cv2.resize(crop, (4, 4), interpolation=cv2.INTER_AREA)
            parts.append(small.tobytes())
        except Exception:
            parts.append(b"")
    return b"|".join(parts)


class Stats:
    __slots__ = ("t_start", "grab_n", "grab_t", "ocr_n", "ocr_t",
                 "sleep_n", "sleep_t", "write_n", "write_t")

    def __init__(self):
        self.reset()

    def reset(self):
        self.t_start = time.perf_counter()
        self.grab_n = 0
        self.grab_t = 0.0
        self.ocr_n = 0
        self.ocr_t = 0.0
        self.sleep_n = 0
        self.sleep_t = 0.0
        self.write_n = 0
        self.write_t = 0.0

    def elapsed(self):
        return time.perf_counter() - self.t_start

    def other(self):
        return max(0.0, self.elapsed() - self.grab_t - self.ocr_t
                   - self.sleep_t - self.write_t)

    def format(self):
        return (
            f"总 {self.elapsed() * 1000:.0f}ms | "
            f"截图 {self.grab_n}次 {self.grab_t * 1000:.0f}ms | "
            f"OCR {self.ocr_n}次 {self.ocr_t * 1000:.0f}ms | "
            f"Sleep {self.sleep_n}次 {self.sleep_t * 1000:.0f}ms | "
            f"落笔 {self.write_n}次 {self.write_t * 1000:.0f}ms | "
            f"其他 {self.other() * 1000:.0f}ms"
        )


class Runner:
    def __init__(self, instance, device, use_gpu=False, log_questions=False,
                 strict_glyph=True):
        self.instance = instance
        self.device = device
        self.screen = m.MuMuScreen(instance=instance)
        self.ocr = m.PaddleDigitOCR(use_gpu=use_gpu)
        self.writer = m.SymbolWriter(device=device, speed=1.25)
        self.touch = self.writer.touch
        self.last_screenshot = time.time()
        self.round = 1
        self.round_done = 0
        self.unknown_streak = 0
        self._last_back = 0.0
        self.stats = Stats()
        self.log_questions = log_questions
        self.strict_glyph = strict_glyph
        try:
            from adbutils import adb
            self._adb = adb.device(device)
        except Exception as e:
            log(f"[实例 {instance}] adb 连接失败：{e}")
            self._adb = None

    # ---- 带统计的包装 ----
    def grab(self):
        t0 = time.perf_counter()
        img = self.screen.grab()
        self.stats.grab_t += time.perf_counter() - t0
        self.stats.grab_n += 1
        return img

    def read_pair(self, img, L, R):
        t0 = time.perf_counter()
        r = self.ocr.read_pair(img, L, R, strict_glyph=self.strict_glyph)
        self.stats.ocr_t += time.perf_counter() - t0
        self.stats.ocr_n += 1
        return r

    def read_boxes(self, img, roi):
        t0 = time.perf_counter()
        r = self.ocr.read_boxes(img, roi)
        self.stats.ocr_t += time.perf_counter() - t0
        self.stats.ocr_n += 1
        return r

    def write_sym(self, sym):
        t0 = time.perf_counter()
        self.writer.write(sym)
        self.stats.write_t += time.perf_counter() - t0
        self.stats.write_n += 1

    def _sleep(self, seconds):
        self.stats.sleep_n += 1
        t0 = time.perf_counter()
        time.sleep(seconds)
        self.stats.sleep_t += time.perf_counter() - t0

    # ---- 原方法 ----
    def snap(self, tag=""):
        img = self.grab()
        p = os.path.join(WORKDIR, f"diag_{time.strftime('%H%M%S')}_{tag}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img

    def _is_live_question(self, img):
        L, R = m.locate_digits(img)
        return L is not None and R is not None

    def _click(self, x, y, off=17):
        self.touch.click(x + random.randint(-off, off), y + random.randint(-off, off), 120)

    def _band_boxes(self, img):
        y0 = int(m.CFG.get("bottom_band_y", 1150))
        return self.read_boxes(img, (0, y0, img.shape[1], img.shape[0]))

    def _collect_reward(self):
        log(f"[实例 {self.instance}] ===== 第 {self.round} 轮完成（本轮作答 {self.round_done} 题）=====")
        self.round += 1
        self.round_done = 0
        btn = m.CFG["buttons"]
        self._click(*btn.get("collect_upper", [539, 1576]))
        self._sleep(1.2)
        if any("开心收下" in t for t, _, _ in self._band_boxes(self.grab())):
            self._click(*btn.get("collect", [539, 1678]))
            self._sleep(0.8)

    def handle_non_question(self, img):
        boxes = self._band_boxes(img)
        texts = "".join(t for t, _, _ in boxes)
        h, w = img.shape[:2]
        btn = m.CFG.get("buttons", {})

        if "不启用" in texts:
            log(f"[实例 {self.instance}]   [弹窗] 启用背包道具 → 点 不启用")
            self._click(*btn.get("dialog_decline", [217, 1745]))
            return True
        if "领取" in texts:
            log(f"[实例 {self.instance}]   [弹窗] 收到道具 → 点 领取道具")
            self._click(*btn.get("dialog_claim", [725, 1745]))
            return True
        if "放弃" in texts:
            log(f"[实例 {self.instance}]   [弹窗] → 点 放弃/不启用")
            self._click(*btn.get("dialog_decline", [217, 1745]))
            return True
        if "开心收下" in texts:
            self._collect_reward()
            return True
        if "继续PK" in texts:
            log(f"[实例 {self.instance}]   [弹窗] → 点 继续PK")
            self._click(*btn.get("continue_pk", [568, 1657]))
            return True
        if "再练一次" in texts or "继续" in texts:
            log(f"[实例 {self.instance}]   [排行/任务] 点 再练一次/继续")
            self._click(*btn.get("continue", [805, 1833]))
            return True
        if "开始" in texts:
            log(f"[实例 {self.instance}]   [主界面] 点 开始练习")
            self._click(*btn.get("start_practice", [742, 1347]))
            return True

        top = img[0:int(h * 0.15), int(w * 0.1):int(w * 0.9)]
        self.unknown_streak += 1
        if self.unknown_streak == 6:
            self.snap(f"unknown_{texts[:6]}")
            log(f"[实例 {self.instance}]   [未知页面] 连续 6 帧未识别：{texts[:40]!r}（已存截图）")
        self._sleep(0.4)
        return False


def worker(instance, device, stop_event, use_gpu=False,
           log_questions=False, fast_poll=True, strict_glyph=True):
    try:
        r = Runner(instance=instance, device=device,
                   use_gpu=use_gpu, log_questions=log_questions,
                   strict_glyph=strict_glyph)
    except Exception as e:
        log(f"[实例 {instance}] 初始化失败: {e}")
        return

    wait_after = m.WAIT_AFTER_WRITE

    def _wait(duration):
        r.stats.sleep_n += 1
        t0 = time.perf_counter()
        stop_event.wait(duration)
        r.stats.sleep_t += time.perf_counter() - t0

    total_done = 0
    last_sig = None
    last_hash = None
    sig_seen_at = 0.0
    rewrites = {}
    last_write_at = -10.0
    ocr_fail_at = None

    log(f"[实例 {instance}] 练习开始，设备 {device}"
        + ("（fast-poll 开）" if fast_poll else "（fast-poll 关）")
        + ("（严格字形校验开）" if strict_glyph else "（严格字形校验关）"))

    while not stop_event.is_set():
        try:
            r.stats.reset()
            img = r.grab()
            if img is None:
                _wait(0.5)
                continue

            L, R = m.locate_digits(img)

            if L is None and time.time() - last_write_at < 1.5:
                _wait(wait_after)
                continue

            if L is None:
                acted = r.handle_non_question(img)
                _wait(0.8 if acted else 0.3)
                continue

            # ---------- 题目页 ----------
            ocr_skipped = False
            if fast_poll and last_hash is not None and last_sig is not None:
                h = _roi_hash(img, L, R)
                if h == last_hash:
                    n1, n2 = last_sig
                    ocr_skipped = True

            if not ocr_skipped:
                n1, n2 = r.read_pair(img, L, R)
                if n1 is not None and n2 is not None and fast_poll:
                    last_hash = _roi_hash(img, L, R)

            if n1 is None or n2 is None:
                now = time.time()
                if ocr_fail_at is None:
                    ocr_fail_at = now
                elif now - ocr_fail_at > 10:
                    r.snap("ocr_fail")
                    log(f"[实例 {instance}]   [跳过] OCR 连续失败 10s，点右上角跳过")
                    r._click(985, 138, off=6)
                    ocr_fail_at = None
                    _wait(0.5)
                _wait(0.25)
                continue
            ocr_fail_at = None

            sig = (n1, n2)
            now = time.time()
            sym = ">" if n1 > n2 else ("<" if n1 < n2 else "=")

            if sig == last_sig:
                if now - last_write_at < 1.0:
                    _wait(wait_after)
                    continue
                if now - sig_seen_at > 3.5:
                    if rewrites.get(sig, 0) >= 2:
                        if r.log_questions:
                            log(f"[实例 {instance}]   [放弃] {n1} {sym} {n2} 重写 2 次仍未识别")
                        r._click(985, 138, off=6)
                        ocr_fail_at = None
                        _wait(0.5)
                        rewrites[sig] = 0
                        last_sig = None
                        last_hash = None
                        sig_seen_at = 0.0
                        _wait(0.6)
                        continue
                    rewrites[sig] = rewrites.get(sig, 0) + 1
                    r.write_sym(sym)
                    last_write_at = now
                    if r.log_questions:
                        log(f"[实例 {instance}] [重试] {n1} {sym} {n2}（重写 {rewrites[sig]}）")
                    _wait(wait_after)
                    continue
                _wait(0.3)
                continue

            total_done += 1
            r.round_done += 1
            last_sig = sig
            sig_seen_at = now
            rewrites = {sig: 0}
            r.write_sym(sym)
            last_write_at = time.time()
            _wait(wait_after)

            if r.log_questions:
                log(f"[实例 {instance}] [统计] {n1} {sym} {n2} {r.stats.format()}")

        except Exception as e:
            log(f"[实例 {instance}] 异常: {e}")
            _wait(0.5)

    log(f"[实例 {instance}] 已停止，共作答 {total_done} 题")


class InstanceManager:
    """管理多个 worker 线程：可独立启停，互不影响。"""
    def __init__(self, use_gpu, log_questions, fast_poll, strict_glyph):
        self.lock = threading.Lock()
        self.workers = {}
        self.use_gpu = use_gpu
        self.log_questions = log_questions
        self.fast_poll = fast_poll
        self.strict_glyph = strict_glyph

    def start(self, instance):
        with self.lock:
            w = self.workers.get(instance)
            if w and w["thread"].is_alive():
                log(f"[管理] 实例 {instance} 已在运行")
                return
            stop_event = threading.Event()
            device = f"127.0.0.1:{16384 + 32 * instance}"
            t = threading.Thread(
                target=worker,
                args=(instance, device, stop_event,
                      self.use_gpu, self.log_questions,
                      self.fast_poll, self.strict_glyph),
                daemon=True,
                name=f"inst-{instance}",
            )
            t.start()
            self.workers[instance] = {"thread": t, "stop_event": stop_event}
            log(f"[管理] 已启动实例 {instance}")

    def stop(self, instance):
        with self.lock:
            w = self.workers.get(instance)
            if not w or not w["thread"].is_alive():
                log(f"[管理] 实例 {instance} 未在运行")
                return
            w["stop_event"].set()
            log(f"[管理] 已发送停止信号给实例 {instance}")

    def stop_all(self):
        with self.lock:
            for w in self.workers.values():
                w["stop_event"].set()

    def join_all(self, timeout=3):
        with self.lock:
            for w in self.workers.values():
                w["thread"].join(timeout=timeout)

    def list_running(self):
        with self.lock:
            running = sorted(i for i, w in self.workers.items()
                             if w["thread"].is_alive())
        log(f"[管理] 运行中：{running if running else '（无）'}")

    def any_alive(self):
        with self.lock:
            return any(w["thread"].is_alive() for w in self.workers.values())


def command_loop(mgr, exit_event):
    try:
        if not (sys.stdin and sys.stdin.isatty()):
            return
        while not exit_event.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                exit_event.set()
                return
            parts = line.strip().split()
            if not parts:
                continue
            cmd = parts[0].lower()
            try:
                if cmd == "start" and len(parts) >= 2:
                    for inst in parse_instances(parts[1]):
                        mgr.start(inst)
                elif cmd == "stop" and len(parts) >= 2:
                    for inst in parse_instances(parts[1]):
                        mgr.stop(inst)
                elif cmd == "list":
                    mgr.list_running()
                elif cmd in ("quit", "exit", "q"):
                    exit_event.set()
                    return
                elif cmd in ("help", "?", "h"):
                    log("命令：start <实例> | stop <实例> | list | quit")
                else:
                    log(f"[管理] 未知命令：{line.strip()}（输入 help 查看）")
            except Exception as e:
                log(f"[管理] 命令执行失败：{e}")
    except Exception as e:
        log(f"[管理] 控制台异常：{e}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=str, default="1",
                    help="实例号，逗号分隔或区间，如 1,2 或 1-3 或 1,3-5")
    ap.add_argument("--use-gpu", action="store_true", help="使用 GPU 加速 OCR")
    ap.add_argument("--log-questions", action="store_true",
                    help="显示每道题的详细日志（默认只显示每题统计）")
    ap.add_argument("--fast-poll", dest="fast_poll", action="store_true", default=True,
                    help="启用 ROI 哈希快速轮询（默认开启）")
    ap.add_argument("--no-fast-poll", dest="fast_poll", action="store_false",
                    help="禁用 ROI 哈希快速轮询")
    ap.add_argument("--strict-glyph", dest="strict_glyph",
                    action="store_true", default=True,
                    help="启用严格字形校验 + 逐字形 OCR 兜底（默认开启）")
    ap.add_argument("--no-strict-glyph", dest="strict_glyph",
                    action="store_false",
                    help="关闭严格字形校验（速度优先，可能重新出现 1→17 误判）")
    args = ap.parse_args()

    instances = parse_instances(args.instances)
    if not instances:
        log("没有指定任何实例")
        return

    mgr = InstanceManager(args.use_gpu, args.log_questions,
                          args.fast_poll, args.strict_glyph)
    for inst in instances:
        mgr.start(inst)

    log(f"===== 已启动 {len(instances)} 个实例：{instances} =====")
    log("命令：start <实例> / stop <实例> / list / quit / help")

    exit_event = threading.Event()
    cmd_thread = threading.Thread(
        target=command_loop, args=(mgr, exit_event),
        daemon=True, name="cmd",
    )
    cmd_thread.start()

    interactive = bool(sys.stdin and sys.stdin.isatty())
    try:
        while not exit_event.is_set():
            if not interactive and not mgr.any_alive():
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("收到 Ctrl+C，正在停止所有实例…")
        exit_event.set()

    mgr.stop_all()
    mgr.join_all(timeout=3)
    log("全部停止")


if __name__ == "__main__":
    main()