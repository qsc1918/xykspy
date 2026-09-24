# 数学练习自动答题助手

一个基于 MuMu 模拟器 + PaddleOCR + MaaTouch 的自动答题脚本。

模拟人工操作：截图 → OCR 识别屏幕上两个数字 → 判断大小 → 手写对应比较符号（`>` / `<` / `=`）→ 自动进入下一题。支持多开、GPU 加速、仿真笔迹。

---

## 特性

- **多实例并行**：一条命令同时驱动多个 MuMu 模拟器实例
- **GPU 加速**：PaddleOCR 支持 CUDA，大幅降低 OCR 耗时
- **仿真手写**：MaaTouch 注入，笔迹带随机起笔位置、角度、抖动、时长，不易被判定为机器
- **低延迟**：单题平均 **300ms** 左右（6 开，i7-14700K + RTX 4070 实测）
- **运行时管理**：程序运行中可动态启动 / 停止任意实例，无需重启

---

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 |
| 模拟器 | MuMu 模拟器 12（其他版本未测试） |
| Python | 3.13（我使用的） 或更高 |
| 显卡 | NVIDIA GPU（可选，但强烈推荐） |
| 依赖管理 | [uv](https://github.com/astral-sh/uv)（推荐）或 pip |

> 没有 GPU 也能跑，把 `--use-gpu` 去掉即可，但速度会明显下降，多开可能不够用。

---

## 安装

### 1. 安装 MuMu 模拟器

- 下载并安装 [MuMu 模拟器 12](https://mumu.163.com/)
- 在 MuMu 多开器中创建你需要的实例数量（例如 6 个）
- 分别启动每个实例，并确保它们**都已经进入练习页面**
- 记下每个实例的编号（多开器里会显示：1、2、3……）

### 2. 安装 Python 3.13+

从 [python.org](https://www.python.org/downloads/) 下载安装，安装时勾选 **Add Python to PATH**。

### 3. 安装 uv（推荐）

```powershell
pip install uv
```

### 4. 拉取项目并安装依赖

```powershell
git clone https://github.com/qsc1918/xykspy.git
cd xykspy
uv sync
```

`uv sync` 会自动读取 `pyproject.toml` 并安装所有依赖，包括从 GitHub 拉取的 `mtc` 和从 Paddle 官方源安装的 `paddlepaddle-gpu`。

如果你不用 uv，也可以用 pip：

```powershell
pip install mtc@git+https://github.com/NakanoSanku/mtc
pip install numpy opencv-contrib-python paddleocr pillow pynput
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

paddlepaddle版本选择：CPU 还是 GPU？
本项目需要用到 `paddlepaddle` 进行 OCR 识别，它分为 CPU 版本 和 GPU 版本。请根据你的硬件情况选择安装：

#### 方案一：GPU 版本（推荐，需要 NVIDIA 显卡）
如果你有 NVIDIA 显卡，并希望获得最快的识别速度，请安装 GPU 版本。这需要你的电脑满足以下条件：
- NVIDIA 驱动：建议版本 >= 550（或较新的驱动版本）
- CUDA：12.9（本项目使用的版本）
- cuDNN：网上说>= 9.5，我也不知道，较新的版本应该都行
----

#### 方案二：CPU 版本（不需要 NVIDIA 显卡）

如果你的电脑没有 NVIDIA 显卡，或者你不想折腾驱动环境，可以直接安装 CPU 版本。它不需要安装 CUDA、cuDNN 或特定的 NVIDIA 驱动，开箱即用。

安装命令如下（Windows 系统）：

```powershell
pip install paddlepaddle==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

> 说明：CPU 版本运行速度会明显慢于 GPU 版本。如果你只是单开或少开使用，CPU 版本通常也能满足基本需求；如果是多开，强烈建议使用 GPU 版本。

使用方式

- 安装 CPU 版本后，启动程序时去掉 --use-gpu 参数即可：

```powershell
python run.py --instances 1-2
```
----

---

## 使用

### 基本命令

```powershell
python run.py --instances 1-6 --use-gpu
```

- `--instances 1-6`：启动实例 1 到 6
- `--use-gpu`：使用 GPU 加速 OCR

启动后会看到类似输出：

```
[18:51:47] [实例 1] 练习开始，设备 127.0.0.1:16416（fast-poll 开）（严格字形校验开）
[18:51:47] [实例 2] 练习开始，设备 127.0.0.1:16448
...
```

程序会自动开始答题，`Ctrl+C` 停止全部。

### 实例号写法

`--instances` 支持三种写法，可以组合使用：

| 写法 | 含义 |
|---|---|
| `1` | 只启动实例 1 |
| `1,2,3` | 启动实例 1、2、3 |
| `1-6` | 启动实例 1 到 6 |
| `1,3-5,7` | 启动实例 1、3、4、5、7 |

### 运行时命令

程序启动后，**在当前终端直接输入命令并回车**即可管理实例：

| 命令 | 作用 |
|---|---|
| `start 7` | 启动实例 7 |
| `start 1,3-5` | 启动实例 1、3、4、5 |
| `stop 2` | 停止实例 2 |
| `list` | 查看当前运行中的实例 |
| `help` | 显示命令帮助 |
| `quit` | 停止全部实例并退出 |

> 这个交互功能只在**直接在终端运行**时生效。如果你把输出重定向到文件或作为后台服务运行，会跳过交互，不影响主流程。

### 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--instances` | `1` | 要启动的实例号，支持逗号、区间 |
| `--use-gpu` | 关 | 启用 GPU 加速 OCR |
| `--log-questions` | 关 | 显示每道题的详细日志和耗时统计 |
| `--fast-poll` | 开 | 用 ROI 哈希快速轮询，减少无谓 OCR |
| `--no-fast-poll` | — | 关闭快速轮询（每次循环都做完整 OCR） |
| `--strict-glyph` | 开 | 严格字形校验 + 逐字形 OCR 兜底 |
| `--no-strict-glyph` | — | 关闭严格校验（速度优先，可能重新出现 1 → 17 的误判） |

一般推荐直接用默认参数：

```powershell
python run.py --instances 1-6 --use-gpu
```

只有在排查问题时才需要动其他开关。

---

## 配置文件

项目根目录下的 `config.json` 是**可选**的，如果不存在就用内置默认值。需要微调时，可以在里面覆盖你想改的项。常见可改项：

```json
{
  "left_roi":  [240, 455, 455, 605],
  "right_roi": [625, 455, 855, 605],
  "write_center": [540, 1300],
  "wait_after_write": 0.1,
  "bottom_band_y": 1150,
  "circle_center": 540,
  "buttons": {
    "start_practice":  [742, 1347],
    "collect":         [539, 1678],
    "collect_upper":   [539, 1576],
    "continue":        [1000, 145],
    "dialog_decline":  [217, 1745],
    "dialog_claim":    [725, 1745],
    "continue_pk":     [568, 1657]
  }
}
```

| 键 | 说明 |
|---|---|
| `left_roi` / `right_roi` | 左侧 / 右侧数字所在的矩形区域 `[x1, y1, x2, y2]` |
| `write_center` | 手写符号的落笔中心点 |
| `wait_after_write` | 写完符号后等待 app 反应的时间（秒） |
| `bottom_band_y` | 底部按钮识别条的上边界 y 坐标 |
| `circle_center` | 中间白圈的 x 坐标（用于左右划分） |
| `buttons.*` | 各页面按钮的固定点击坐标 |

> 如果你的分辨率和默认布局一致，通常不需要改任何东西。

---

## 性能参考

实测环境：**i7-14700K + RTX 4070**，MuMu 12，6 开，GPU 加速。

| 项目 | 数据 |
|---|---|
| 单题耗时 | 约 **300ms** |
| 30 题（一轮） | 约 **10 秒** |
| 单题循环拆解 | 截图 ~15ms，OCR ~40ms，落笔 ~210ms，等待 ~100ms |

不同硬件、不同模拟器负载会有差异，但 6 开时 CPU 占用和 GPU 占用都不会打满，属于轻负载运行。

---

## 常见问题

**Q：启动时报 `adb 连接失败`？**
A：检查 MuMu 是否已经启动，以及对应实例的 ADB 端口是否可用。默认端口是 `16384 + 32 × 实例号`（实例 1 = 16416，实例 2 = 16448，以此类推）。

**Q：OCR 一直识别失败，程序点了「跳过」？**
A：大概率是 `left_roi` / `right_roi` 和当前分辨率不匹配。可以打开一张实时截图，量一下数字所在的区域，然后改 `config.json`。

**Q：手写符号没有被 app 识别，反复重写？**
A：可能是落笔速度太快或太慢。可以在 `run.py` 的 `Runner.__init__` 里调整 `speed` 参数（默认 `1.25`，越大越慢、越稳）。

**Q：6 开时 CPU 或 GPU 占用很高？**
A：可以试着用 `--no-fast-poll` 看是否缓解（会牺牲一点速度）。如果 GPU 显存吃紧，也可以去掉 `--use-gpu` 改用 CPU，但速度会明显下降。

---

## 免责声明

本项目仅供**个人学习与技术研究**使用。请勿用于任何违反 app 服务条款、影响其他用户正常使用、或谋取不当利益的场景。使用本项目产生的一切后果由使用者自行承担。