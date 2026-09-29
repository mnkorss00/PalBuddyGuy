# Pal Buddy Guy: The anipal's best friend
This project improves on the tracking of the Vive Pro Eye and the Vive Facial Tracker. You make an expression, record it, and train a small network that drives a VRCFaceTracking parameter from it.

It now comes with a GUI, and one process handles receiving data, recording, training, calibration and output.

> 한국어 요약은 [아래](#한국어-빠른-시작)에 있습니다.

# System requirements
* An NVIDIA GPU with CUDA and at least 4 GB of VRAM is recommended. The code also runs on CPU (and Apple MPS), but training will be slow.
* Works with **both trackers, the facial tracker only, or the Pro Eye only**. See *Input mode* below.
* Python 3.8+ with PyTorch.

# Installation
1. Replace the tvm runtime inside SRanipal. Copy the two .DLL files from the `tvm runtime` folder into `C:\Program Files\VIVE\SRanipal`, replacing the existing files. Back up the old files first in case you want to revert.
2. **Double-click `PalBuddyGuy.bat`.** That's it.

On the first run the launcher:
* finds Python 3.9+, or offers to install it with winget,
* creates a private environment in `.venv`, so your system Python is left untouched,
* installs PyTorch (the CUDA build when an NVIDIA GPU is detected, otherwise the CPU build) and the other packages. This is a 2–3 GB download and takes a few minutes,
* offers to create a **Pal Buddy Guy** shortcut on the desktop.

After that, double-clicking the .bat or the shortcut opens the GUI directly, without a console window. Startup errors (such as a port already in use) appear in a dialog, and everything is logged to `palbuddy.log`. To reinstall from scratch, delete the `.venv` folder and run the .bat again.

# Running
Start Pal Buddy Guy (double-click) **before** SRanipalRuntime.

Manual / advanced use, with your own Python environment:

```
pip install -r requirements.txt   # install a GPU build of torch first: https://pytorch.org/get-started/locally/
python -m palbuddy          # GUI  (same as: python script.py)
python -m palbuddy --cli    # text commands, like the old script
python -m palbuddy --infer  # start tracking right away
```
Arguments given to the .bat are passed on as well, for example `PalBuddyGuy.bat --infer`.

You no longer need to start `tvm_proxy.py` separately, because the app receives the SRanipal streams itself. See *Legacy proxy mode* below if you still want a separate proxy.

All settings are saved in `config.json`, which is created on first start. You can edit them from the GUI; nothing has to be edited in the source code.

## 1. Check the streams (Live tab)
The status bar shows whether the eye tracker, face tracker and VRCFaceTracking are connected, and their frame rates.
The camera preview should show the **eyes on top and the face below**. If they're reversed, click **Swap eye/face**. The choice is saved.

### Input mode
The **Input** box on the Live tab selects which trackers feed the network:

| Input | Uses |
|---|---|
| Eye + face trackers | both feature maps (the original behaviour) |
| Face tracker only | only the facial tracker's features |
| Eye tracker only | only the Pro Eye's features |

* If the connected trackers don't match the selected mode, a suggestion appears. When only one tracker streams, check the preview and click *Face only* or *Eye only*. With a single tracker, it doesn't matter which SRanipal port its stream arrives on, so no swapping is needed.
* A model only works in the input mode it was trained in, and the mode is stored in the model file. After switching, train again.
* Recordings always store both halves, with a missing tracker saved as zeros. That means **recordings made with both trackers can be reused** for face-only or eye-only training. A recording made with one tracker can't be used for a mode that needs the other tracker, and training tells you which file is the problem.
* In every mode the outputs drive VRCFT *lip* shapes, because that's what the VRCFT module accepts. Normal eye tracking (gaze, blink) keeps working as before. In eye-only mode you can still train something like a frown seen by the eye cameras and map it to a lip shape.

## 2. Record (Record tab)
Enter a name and click **Record**. After the countdown, 2048 frames are captured (about 30 s) and written directly to `<dataset folder>/<name>-em.mmap`.
* You *must* also record a **neutral** set. It doesn't have to be a truly neutral face. It should cover any faces you **don't** want to track: talking, looking around, blinking. That way the network learns what *not* to fire on.
* While recording, keep the target expression dominant but move around a bit (adjust the headset, turn your head). The goal is diverse data where the target expression is the one constant.
* Each recording is about 400 MB, so point the dataset folder (Settings tab) at a drive with space.
* Old `.pkl` recordings can be converted with **Convert old .pkl**.

## 3. Configure classes and train (Train tab)
Each class is one network output. The **first class must be neutral**. For each class, choose its recordings and, optionally, the SRanipal lip shape it should drive ("Drives shape"). You can also select recordings in the Record tab and click **Add to class…**.
Click **Train**. The loss curve should drop below the 0.001 line by the end. If it doesn't, something is wrong with the recordings or the class list. Then click **Save model**.

## 4. Track and calibrate (Live tab)
Click **Start tracking**. Each output is shown as its raw value and the value sent to VRCFT.
**FastCal** puppets each shape on your avatar one after another. Copy it with your face. The measured strengths become each class's *max power*, which you can also edit by hand in the Train tab. **Smoothing** reduces jitter at the cost of a little latency.

## 5. Inference performance (Settings tab → Inference performance)
Tracking runs the network on every new frame, about 60 times a second. These settings trade GPU, CPU, memory and latency against each other:

| Setting | What it does |
|---|---|
| **Engine** | *ONNX Runtime* is the fastest on the CPU and never loads PyTorch while tracking, which saves hundreds of MB of RAM. *auto* uses it when it's installed; the launcher installs it. When you save a model, an `.onnx` copy (plus an int8 version) is written next to the `.pt`, and it is refreshed automatically if the `.pt` is newer. |
| **Run on** | *auto* uses the GPU if there is one. With ONNX Runtime that means DirectML (`onnxruntime-directml`), which works on NVIDIA, **AMD** and Intel GPUs. *CPU* leaves the GPU entirely to VR. *GPU* forces the GPU. |
| **CPU threads** | Default 1. More threads cut latency but **increase** total CPU use. |
| **int8 quantization** | On the CPU, uses int8 weights for the big linear layer: about 2–3× faster, with the same model and no retraining. The output difference is about 0.003. |
| **Max tracking rate** | For example 30 Hz. Skips frames in between and halves the work. Combine it with *Smoothing* on the Live tab. |

**Benchmark this PC** times every available backend with your model, so you can pick what works best on your machine. The Live tab shows the running backend, the time per frame and the whole process's CPU use.

### Lite model (Train tab → Model)
*lite* has the same layout with fewer channels: about 8× fewer weights and about 5× less compute. Whether it tracks as well as *standard* depends on your recordings. To compare them, train both and look at the **validation accuracy**.

### Validation accuracy
Training holds out the last 10 % of every recording (`validation_split` in `config.json`) and reports validation loss and accuracy after each epoch, plus the worst classes at the end. The hold-out is taken from the end of each recording rather than as random frames, because neighbouring frames are nearly identical and a random split would overstate accuracy.

### Measurements
One 2.8 GHz Xeon core, frames arriving at 60 Hz, whole-process CPU as % of one core, memory measured while tracking:

| Model | Backend | ms/frame | CPU | RAM |
|---|---|---|---|---|
| standard | PyTorch fp32, all cores (old default) | 8.0 | 105 % | |
| standard | PyTorch int8, 1 thread | 7.6 | 50 % | 761 MB |
| standard | **ONNX int8, 1 thread** | 5.3 | 31 % | **102 MB** |
| lite | PyTorch fp32, 1 thread | 3.4 | 22 % | 576 MB |
| lite | **ONNX int8, 1 thread** | 1.4 | **11 %** | **72 MB** |

Idle worker threads are also told not to busy-wait (`OMP_WAIT_POLICY=PASSIVE`, and ONNX Runtime spinning is off), so they don't burn CPU between frames.

### Tracking on a PC without PyTorch
Copy `config.json`, `buddyguy.onnx`, `buddyguy.int8.onnx` and `buddyguy.json` next to the app, then `pip install numpy onnxruntime`. Tracking works without PyTorch; only training needs it.

## 6. VRCFaceTracking (Settings tab → VRCFaceTracking module)
For **VRCFaceTracking v6**, install the module from the app with **Install / update**, then restart VRCFaceTracking. The status bar shows *VRCFaceTracking: connected (v6 module)*.

* VRCFaceTracking v6 lets only one module provide eye tracking and one provide expressions, so the Pal Buddy Guy module **runs the SRanipal module inside itself**. The installer disables the standalone SRanipal module by renaming its `module.json`, and **Remove** restores it. Install the SRanipal module in VRCFaceTracking first.
* A class's *Drives shape* can be an SRanipal lip shape (as before) or any **Unified Expression**, including eye-area shapes like `BrowLowererLeft` or `EyeSquintRight` that SRanipal can't express. For example, you can train a frown seen by the Pro Eye cameras.
* **v1 (legacy SRanipal) avatar parameters work.** VRCFaceTracking v6 computes the v1 parameters (`JawOpen`, `MouthSmileLeft`, `SmileSadLeft`, `JawX`, …) from Unified Expressions. SRanipal targets are translated so that this round trip gives the v1 parameter **exactly** the trained value, which is verified against VRCFaceTracking's own parameter functions for every shape. The v1 eye/brow parameters can also be chosen as targets directly: `LeftEyeWiden`, `LeftEyeSqueeze`, `BrowDownLeft`, `BrowsInnerUp`, and so on. Limitations that come from VRCFaceTracking itself:
  * The v1 `Tongue*Morph` parameters are always 0 in v6. Pal Buddy Guy drives the two tongue directions instead.
  * `MouthUpperOverturn` / `MouthLowerOverturn` also raise v1 `MouthUpperUp*` / `MouthLowerDown*`. This follows VRCFaceTracking's conversion formula, and every Unified-based module behaves the same way.
  * `MouthSadLeft/Right` are sent as `MouthStretch*`, because that reproduces v1 exactly without leaking into the other side. For a Unified (v2) avatar, pick `MouthFrownLeft/Right` instead.
* By default Pal Buddy Guy's value replaces SRanipal's for the shapes it drives. *Keep SRanipal's value when it is larger* switches to the maximum of the two.
* When tracking stops, the module falls back to plain SRanipal within 0.5 s.
* The original VRCFaceTracking module (≤ v4) still works; it only understands SRanipal lip shapes.

Details, the protocol and how to build or test the module: [`vrcft-module/README.md`](vrcft-module/README.md).

## 7. Process priority (Settings tab → Inference performance)
Tracking needs very little CPU, so by default the process runs at **below normal** priority and yields to VR and the game when the CPU is busy. On hybrid Intel CPUs (12th gen and later, e.g. the i7-12700K), **Efficiency cores only** keeps it off the P-cores entirely. On Windows 11, **Efficiency mode** (EcoQoS) lets the scheduler run it on slow, low-power cores. *low* priority is available too, but tracking may stutter when the CPU is fully loaded.

# What changed compared to the original scripts
**GUI**
* Tkinter GUI (no extra dependencies) with live connection status, camera preview, output meters, recording with progress, a class/dataset editor, training with a live loss chart and ETA, FastCal, and settings. Available in English and Korean.

**Performance**
* SRanipal is received in-process instead of through the proxy, which removes one TCP hop and one process.
* Frames are handed over through a latest-sample slot with a condition variable. The old code used a list queue with `pop(0)` and polled in 1 ms sleep loops.
* Inference runs once per *new* frame. The old loop recomputed the same frame and then slept 10 ms even when a new frame was waiting. It uses a pinned input buffer and `inference_mode`.
* VRCFT updates go out as one packet per frame. The old code made 1 + 3·N tiny `send` calls with Nagle enabled.
* Training: batches are built with one fancy-indexed read per recording and prepared by background threads while the GPU trains. Mixed precision (AMP) and cudnn autotuning are available. Recording writes straight to `.mmap`, so the `pkl → mmap` conversion step is gone.
* Fixed the `batch_size` global hack: inference used to set it to 1 and swap the dropout functions out.

**Connectivity**
* Disconnects are detected. Before, `recv()` returned `b""` forever after a peer left, and the thread spun at 100 % CPU without ever reconnecting.
* A restarted SRanipal or VRCFT replaces its stale connection automatically. There's no need to restart the script.
* The VRCFT connection uses TCP_NODELAY and keepalive. A send timeout keeps a frozen VRCFT from stalling tracking.
* All ports listen on `127.0.0.1` by default instead of on every network interface. This is configurable.
* Ports are configurable, and there's a clear error message when a port is already in use.
* The proxy client (legacy mode) reconnects with backoff.
* The status bar shows each stream as connected, no data (stalled) or disconnected.

The network architecture, the `buddyguy.pt` checkpoint format, the recording format and the VRCFT wire protocol are unchanged, so existing models, recordings and the VRCFT module keep working.

# Legacy proxy mode
If you want the receiver in its own process, run `python tvm_proxy.py` and set the input source to *proxy* in Settings (or `"source": "proxy"` in `config.json`).

# Tests
`python -m unittest discover tests` runs offline tests. They use fake SRanipal and VRCFT peers over loopback, and a tiny record → train → infer run on CPU.

---

# 한국어 빠른 시작
1. `tvm runtime` 폴더의 DLL 두 개를 `C:\Program Files\VIVE\SRanipal`에 덮어씁니다 (원본은 백업해 두세요).
2. **`PalBuddyGuy.bat`을 더블클릭합니다.** 처음 실행할 때는 Python(없으면 winget으로 설치할지 묻습니다), PyTorch(NVIDIA GPU가 있으면 CUDA 버전)와 나머지 패키지를 `.venv` 폴더에 자동으로 설치하고, 바탕화면 바로가기를 만들지 묻습니다. 다운로드가 약 2~3GB라 몇 분 걸립니다. 그다음부터는 더블클릭하면 콘솔 창 없이 바로 GUI가 뜹니다. 다시 설치하려면 `.venv` 폴더를 지우고 다시 실행하세요.
3. **SRanipalRuntime보다 먼저** 실행해야 합니다. 이제 `tvm_proxy.py`는 따로 실행할 필요가 없습니다.
4. **실시간** 탭: *입력 방식*을 고릅니다. **눈+입 / 입만(페이셜 트래커) / 눈만(Pro Eye)** 중 하나입니다. 트래커가 하나만 연결되어 있으면 미리보기를 보고 *입만* 또는 *눈만*을 누르면 됩니다. 두 트래커를 모두 쓸 때는 위에 눈, 아래에 얼굴이 보이는지 확인하고, 반대로 보이면 *눈/얼굴 뒤바꾸기*를 누르세요. 입력 방식을 바꾸면 다시 학습해야 합니다. 눈+입으로 녹화한 파일은 입만/눈만 학습에도 그대로 쓸 수 있습니다.
5. **녹화** 탭: 이름을 입력하고 녹화합니다 (약 30초, 파일 하나에 약 400MB). `neutral`(무표정) 녹화는 반드시 있어야 합니다.
6. **학습** 탭: 클래스별로 녹화 파일과 대상 파라미터를 지정합니다 (첫 번째 클래스는 neutral). *학습 시작*을 누르고, 손실이 0.001 아래로 내려가면 *모델 저장*을 누릅니다.
7. **실시간** 탭: *트래킹 시작*을 누른 뒤 *빠른 보정(FastCal)*으로 아바타를 따라 하며 보정합니다.

8. **설정 탭 → 추론 성능**: 기본값(자동: ONNX Runtime, CPU int8)으로도 가볍게 돌아갑니다. *이 PC에서 비교 측정*을 누르면 CPU/GPU/ONNX 중 어느 쪽이 이 PC에 맞는지 바로 확인할 수 있습니다. 학습 탭에서 *모델 크기 → 경량*을 고르면 추론이 약 5배 빨라집니다. 표준 모델과 검증 정확도를 비교해 보고 결정하세요.
9. **설정 탭 → VRCFaceTracking 모듈 → 설치**: VRCFaceTracking v6용 모듈을 설치합니다. 먼저 VRCFaceTracking에서 SRanipal 모듈을 설치해 두세요. 설치하면 SRanipal 모듈은 Pal Buddy Guy 모듈 안에서 실행됩니다. v6는 눈과 표정을 각각 모듈 하나만 담당할 수 있기 때문입니다. 설치 후 VRCFaceTracking을 다시 시작하세요. *제거*를 누르면 원래대로 돌아갑니다. 대상 파라미터로 `BrowLowererLeft` 같은 **눈썹/눈 주변 표정**도 고를 수 있습니다.
10. **프로세스 우선순위**: 기본값은 "보통 이하"라서 VR/게임에 CPU를 양보합니다. 12세대 이후 Intel CPU에서는 *E코어만 사용*, Windows 11에서는 *효율 모드*도 쓸 수 있습니다.

언어는 설정 탭에서 바꿀 수 있습니다 (auto / en / ko).
