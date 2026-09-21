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


class Runner:
    def __init__(self, instance, device, use_gpu=False):
        # Pass the specific instance for IPC capture and device serial for MaaTouch
        self.instance = instance
        self.device = device
        self.screen = m.MuMuScreen(instance=instance)
        self.ocr = m.PaddleDigitOCR(use_gpu=use_gpu)
        self.writer = m.SymbolWriter(device=device, speed=1.5)
        self.touch = self.writer.touch
        self.last_screenshot = time.time()
        self.round = 1          # 当前第几轮（每轮 100 题）
        self.round_done = 0     # 本轮已作答题数
        self.unknown_streak = 0
        self._last_back = 0.0        # 上次按返回键的时间（冷却用，防转场时连按）
        # adb 连接（用于按返回键关闭弹窗）
        try:
            from adbutils import adb
            self._adb = adb.device(device)
        except Exception as e:
            log(f"[实例 {instance}] adb 连接失败：{e}")
            self._adb = None

    def grab(self):
        return self.screen.grab()

    def snap(self, tag=""):
        img = self.grab()
        p = os.path.join(WORKDIR, f"diag_{time.strftime('%H%M%S')}_{tag}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img

    def _is_live_question(self, img):
        L, R = m.locate_digits(img)
        return L is not None and R is not None

    def _click(self, x, y, off=17):
        """固定坐标 + 随机 15~20px 偏移（像真人手指落点，防固定坐标指纹）"""
        self.touch.click(x + random.randint(-off, off), y + random.randint(-off, off), 120)

    def _band_boxes(self, img):
        """底部条（y 从 bottom_band_y 到屏底）OCR 出每行文字和位置，返回 [(text,cx,cy),...]。
        各页面（主界面/奖励/任务/排行榜/弹窗）的按钮全都落在这个条里，
        所以一次 OCR 就能判断"我在哪个页面、该点哪"。"""
        y0 = int(m.CFG.get("bottom_band_y", 1150))
        return self.ocr.read_boxes(img, (0, y0, img.shape[1], img.shape[0]))

    def _collect_reward(self):
        """奖励页「开心收下」：点固定坐标；1.2s 后页面没走，
        再补点另一个位置（当天前 1~2 轮按钮偏下，之后偏上，两个位置点一遍最稳）。"""
        log(f"[实例 {self.instance}] ===== 第 {self.round} 轮完成（本轮作答 {self.round_done} 题）=====")
        self.round += 1
        self.round_done = 0
        btn = m.CFG["buttons"]
        # 先点偏上位置（1 个奖励=稳态），没走再补点偏下（3 个奖励=当天前 1~2 轮）
        self._click(*btn.get("collect_upper", [539, 1576]))
        time.sleep(1.2)
        if any("开心收下" in t for t, _, _ in self._band_boxes(self.grab())):
            self._click(*btn.get("collect", [539, 1678]))
            time.sleep(0.8)

    def handle_non_question(self, img):
        """非题目页：底部条 OCR 认字判断页面，点击一律固定坐标（±17px 偏移）。
        OCR 只当"眼睛"，坐标当"手"——框坐标在 PaddleOCR 3.x 里是内部坐标系，不可靠。"""
        boxes = self._band_boxes(img)
        texts = "".join(t for t, _, _ in boxes)
        h, w = img.shape[:2]
        btn = m.CFG.get("buttons", {})

        if "不启用" in texts:             # 启用背包道具弹窗 → 左按钮「不启用」（保护背包）
            log(f"[实例 {self.instance}]   [弹窗] 启用背包道具 → 点 不启用")
            self._click(*btn.get("dialog_decline", [217, 1745]))
            return True
        if "领取" in texts:               # 收到免费道具弹窗 → 右按钮「领取道具」
            log(f"[实例 {self.instance}]   [弹窗] 收到道具 → 点 领取道具")
            self._click(*btn.get("dialog_claim", [725, 1745]))
            return True
        if "放弃" in texts:               # 其它只有「放弃」的弹窗 → 左按钮
            log(f"[实例 {self.instance}]   [弹窗] → 点 放弃/不启用")
            self._click(*btn.get("dialog_decline", [217, 1745]))
            return True
        if "开心收下" in texts:           # 奖励弹窗（盖在排行榜上！必须先关，
            self._collect_reward()        # 否则点的是被挡住的「再练一次」）
            return True
        if "继续PK" in texts:
            log(f"[实例 {self.instance}]   [弹窗] → 点 继续PK")
            self._click(*btn.get("continue_pk", [568, 1657]))
            return True
        if "再练一次" in texts or "继续" in texts:   # 排行榜 / 任务奖励页
            log(f"[实例 {self.instance}]   [排行/任务] 点 再练一次/继续")
            self._click(*btn.get("continue", [805, 1833]))
            return True
        if "开始" in texts:           # 主界面
            log(f"[实例 {self.instance}]   [主界面] 点 开始练习")
            self._click(*btn.get("start_practice", [742, 1347]))
            return True

        # 未知页面：顶部压暗 = 广告类弹窗，连续 6 帧存诊断截图
        top = img[0:int(h * 0.15), int(w * 0.1):int(w * 0.9)]
        self.unknown_streak += 1
        if self.unknown_streak == 6:
            self.snap(f"unknown_{texts[:6]}")
            log(f"[实例 {self.instance}]   [未知页面] 连续 6 帧未识别：{texts[:40]!r}（已存截图）")
        time.sleep(0.4)
        return False


def worker(instance, device, stop_event, use_gpu=False):
    """单个模拟器实例的主循环（每个线程一份 Runner / OCR / MaaTouch）。"""
    try:
        r = Runner(instance=instance, device=device, use_gpu=use_gpu)
    except Exception as e:
        log(f"[实例 {instance}] 初始化失败: {e}")
        return

    wait_after = m.WAIT_AFTER_WRITE   # 写完停顿 0.25s（在 config.json 里改）

    total_done = 0
    last_sig = None          # 上一题签名 (n1, n2)
    sig_seen_at = 0.0        # 该题首次出现时间（卡题判断用）
    rewrites = {}            # sig -> 该题已重写次数
    last_write_at = -10.0    # 最后一次落笔时间（转场保护用）
    ocr_fail_at = None       # 当前题连续 OCR 失败的起始时间

    log(f"[实例 {instance}] 练习开始，设备 {device}")

    while not stop_event.is_set():
        try:
            img = r.grab()
            if img is None:
                stop_event.wait(0.5)
                continue

            L, R = m.locate_digits(img)

            # 写完 1.5s 内屏幕还在转场：不做任何 UI 处理
            if L is None and time.time() - last_write_at < 1.5:
                stop_event.wait(wait_after)
                continue

            if L is None:
                # 非题目页（主界面/奖励/任务/排行榜/弹窗/广告）
                acted = r.handle_non_question(img)
                stop_event.wait(0.8 if acted else 0.3)
                continue

            # ---------- 题目页 ----------
            n1, n2 = r.ocr.read_pair(img, L, R)
            if n1 is None or n2 is None:
                now = time.time()
                if ocr_fail_at is None:
                    ocr_fail_at = now
                elif now - ocr_fail_at > 10:       # OCR 连续 10s 失败 → 点「跳过」救场
                    r.snap("ocr_fail")
                    log(f"[实例 {instance}]   [跳过] OCR 连续失败 10s，点右上角跳过")
                    r._click(985, 138, off=6)
                    ocr_fail_at = None
                    stop_event.wait(0.5)
                stop_event.wait(0.25)
                continue
            ocr_fail_at = None

            sig = (n1, n2)
            now = time.time()
            sym = ">" if n1 > n2 else ("<" if n1 < n2 else "=")

            if sig == last_sig:
                # 和上一步是同一题：刚写完等 app 前进；同一题拖过 3.5s = 没被识别 → 重写
                if now - last_write_at < 1.0:
                    stop_event.wait(wait_after)
                    continue
                if now - sig_seen_at > 3.5:
                    if rewrites.get(sig, 0) >= 2:
                        log(f"[实例 {instance}]   [放弃] {n1} {sym} {n2} 重写 2 次仍未识别")
                        r._click(985, 138, off=6)
                        ocr_fail_at = None
                        stop_event.wait(0.5)
                        rewrites[sig] = 0
                        last_sig = None
                        sig_seen_at = 0.0
                        stop_event.wait(0.6)
                        continue
                    rewrites[sig] = rewrites.get(sig, 0) + 1
                    r.writer.write(sym)
                    last_write_at = now
                    log(f"[实例 {instance}] [重试] {n1} {sym} {n2}（重写 {rewrites[sig]}）")
                    stop_event.wait(wait_after)
                    continue
                stop_event.wait(0.3)
                continue

            # 新题 → 计 1 题（只有题面真的换了才计数，不会把重写作双算）
            total_done += 1
            r.round_done += 1
            last_sig = sig
            sig_seen_at = now
            rewrites = {sig: 0}
            t0 = time.time()
            r.writer.write(sym)
            last_write_at = time.time()
            dt = (time.time() - t0) * 1000
            log(f"[实例 {instance}] [{total_done}] {n1} {sym} {n2}  (落笔 {dt:.0f}ms)")
            stop_event.wait(wait_after)

        except Exception as e:
            log(f"[实例 {instance}] 异常: {e}")
            stop_event.wait(0.5)

    log(f"[实例 {instance}] 已停止，共作答 {total_done} 题")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=str, default="1",
                    help="实例号，逗号分隔或区间，如 1,2 或 1-3 或 1,3-5")
    ap.add_argument("--use-gpu", action="store_true", help="使用 GPU 加速 OCR")
    args = ap.parse_args()

    # 解析实例号列表
    instances = []
    for part in args.instances.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            instances.extend(range(int(a), int(b) + 1))
        else:
            instances.append(int(part))

    if not instances:
        log("没有指定任何实例")
        return

    stop_event = threading.Event()
    threads = []
    for inst in instances:
        device = f"127.0.0.1:{16384 + 32 * inst}"
        t = threading.Thread(
            target=worker,
            args=(inst, device, stop_event, args.use_gpu),
            daemon=True,
            name=f"inst-{inst}",
        )
        t.start()
        threads.append(t)

    log(f"===== 已启动 {len(instances)} 个实例：{instances}，Ctrl+C 停止 =====")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("收到 Ctrl+C，正在停止所有实例…")
        stop_event.set()
        for t in threads:
            t.join(timeout=3)
        log("全部停止")


if __name__ == "__main__":
    main()