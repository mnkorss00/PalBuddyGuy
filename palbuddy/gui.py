"""Tkinter GUI for recording, training, calibrating and running the tracker.

Tk is not thread safe: worker threads only put messages on `self.events`,
which the Tk main loop drains every 50 ms.
"""

import locale
import logging
import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .config import ExpressionClass
from .datasets import frame_count
from .engine import dataset_files
from .frames import decode_camera
from .i18n import Translator
from .params import LIP_SHAPES

log = logging.getLogger(__name__)

STATE_COLORS = {"ok": "#2e9d4f", "stalled": "#d69a00", "disconnected": "#c0392b"}
PREVIEW_SCALE = 2


def detect_language(setting):
    if setting in ("en", "ko"):
        return setting
    try:
        loc = (locale.getlocale()[0] or "") + (os.environ.get("LANG") or "")
    except ValueError:
        loc = ""
    return "ko" if "ko" in loc.lower() or "korean" in loc.lower() else "en"


class QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.q.put(("log", self.format(record)))
        except Exception:
            pass


def gray_to_photo(img):
    """numpy uint8 (H, W) -> tk.PhotoImage via an in-memory PGM (no PIL needed)."""
    h, w = img.shape
    data = b"P5 %d %d 255\n" % (w, h) + np.ascontiguousarray(img).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


class ClassDialog(tk.Toplevel):
    def __init__(self, app, cls=None):
        super().__init__(app.root)
        self.app = app
        t = app.t
        self.title(t("class_dialog"))
        self.transient(app.root)
        self.result = None
        cls = cls or ExpressionClass("", [], None, 0.9)

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=t("col_name")).grid(row=0, column=0, sticky="w")
        self.name = tk.StringVar(value=cls.name)
        ttk.Entry(frm, textvariable=self.name, width=30).grid(row=0, column=1, sticky="ew")

        ttk.Label(frm, text=t("col_target")).grid(row=1, column=0, sticky="w")
        self.none_label = t("none")
        shapes = [self.none_label] + sorted(LIP_SHAPES, key=LIP_SHAPES.get)
        self.target = tk.StringVar(value=cls.target or self.none_label)
        ttk.Combobox(frm, textvariable=self.target, values=shapes, state="readonly", width=28).grid(
            row=1, column=1, sticky="ew")

        ttk.Label(frm, text=t("col_power")).grid(row=2, column=0, sticky="w")
        self.power = tk.StringVar(value=str(cls.max_power))
        ttk.Entry(frm, textvariable=self.power, width=10).grid(row=2, column=1, sticky="w")

        ttk.Label(frm, text=t("files_hint")).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.files = tk.Listbox(frm, selectmode="multiple", height=12, exportselection=False)
        self.files.grid(row=4, column=0, columnspan=2, sticky="nsew")
        available = dataset_files(app.engine.cfg)
        for f in cls.files:  # keep entries whose file is missing (e.g. on another drive)
            if f not in available:
                available.append(f)
        for i, f in enumerate(available):
            self.files.insert("end", f)
            if f in cls.files:
                self.files.selection_set(i)

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="OK", command=self.ok).pack(side="left", padx=4)
        ttk.Button(btns, text=t("cancel"), command=self.destroy).pack(side="left")
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(4, weight=1)
        self.grab_set()
        self.wait_window()

    def ok(self):
        try:
            power = float(self.power.get())
            if power <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(self.app.t("error"), "max power > 0", parent=self)
            return
        name = self.name.get().strip()
        if not name:
            return
        target = self.target.get()
        files = [self.files.get(i) for i in self.files.curselection()]
        self.result = ExpressionClass(name, files, None if target == self.none_label else target, power)
        self.destroy()


class App:
    def __init__(self, engine):
        self.engine = engine
        self.cfg = engine.cfg
        self.t = Translator(detect_language(self.cfg.language))
        self.events = queue.Queue()
        logging.getLogger().addHandler(QueueLogHandler(self.events))

        self.root = tk.Tk()
        self.root.title("%s" % self.t("title"))
        self.root.geometry("1000x720")
        self.root.minsize(860, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names() and os.name != "nt":
            style.theme_use("clam")

        self._photo = None
        self.loss_history = []
        self.bars = {}

        self._build_status_bar()
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._build_live_tab()
        self._build_record_tab()
        self._build_train_tab()
        self._build_settings_tab()
        self._build_log_tab()

        self.refresh_classes()
        self.refresh_recordings()
        self.root.after(50, self._pump_events)
        self.root.after(200, self._refresh_status)
        self.root.after(66, self._refresh_preview)

    # ================================================================ layout
    def _build_status_bar(self):
        t = self.t
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")
        self.status_labels = {}
        for key in ("eye", "face", "vrcft"):
            f = ttk.Frame(bar)
            f.pack(side="left", padx=(0, 18))
            dot = tk.Canvas(f, width=12, height=12, highlightthickness=0)
            dot.create_oval(1, 1, 11, 11, fill=STATE_COLORS["disconnected"], outline="", tags="dot")
            dot.pack(side="left", padx=(0, 4))
            lbl = ttk.Label(f, text=t(key))
            lbl.pack(side="left")
            self.status_labels[key] = (dot, lbl)
        self.device_label = ttk.Label(bar, text="")
        self.device_label.pack(side="right")

    def _build_live_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text=t("tab_live"))

        left = ttk.LabelFrame(tab, text=t("camera"), padding=6)
        left.pack(side="left", fill="y")
        size = 200 * PREVIEW_SCALE
        self.preview = tk.Canvas(left, width=size, height=size, bg="black", highlightthickness=0)
        self.preview.pack()
        self.preview_image = self.preview.create_image(0, 0, anchor="nw")
        row = ttk.Frame(left)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text=t("swap"), command=self.on_swap).pack(side="left")
        self.show_preview = tk.BooleanVar(value=self.cfg.show_preview)
        ttk.Checkbutton(row, text=t("preview"), variable=self.show_preview,
                        command=self.on_preview_toggle).pack(side="left", padx=8)

        right = ttk.Frame(tab, padding=(10, 0))
        right.pack(side="left", fill="both", expand=True)
        ctl = ttk.Frame(right)
        ctl.pack(fill="x")
        self.infer_btn = ttk.Button(ctl, text=t("start_infer"), command=self.on_toggle_infer)
        self.infer_btn.pack(side="left")
        ttk.Button(ctl, text=t("fastcal"), command=self.on_fastcal).pack(side="left", padx=6)
        ttk.Label(ctl, text=t("smoothing")).pack(side="left", padx=(16, 4))
        self.smoothing = tk.DoubleVar(value=self.cfg.smoothing)
        ttk.Scale(ctl, from_=0.0, to=0.9, variable=self.smoothing, length=140,
                  command=self.on_smoothing).pack(side="left")
        self.infer_stats = ttk.Label(right, text=t("not_tracking"))
        self.infer_stats.pack(anchor="w", pady=(6, 0))
        self.fastcal_label = ttk.Label(right, text=t("fastcal_help"), foreground="#666")
        self.fastcal_label.pack(anchor="w", pady=(2, 6))

        self.bars_frame = ttk.LabelFrame(right, text=t("outputs"), padding=6)
        self.bars_frame.pack(fill="both", expand=True)

    def _rebuild_bars(self):
        for w in self.bars_frame.winfo_children():
            w.destroy()
        self.bars = {}
        for i, c in enumerate(self.cfg.classes):
            ttk.Label(self.bars_frame, text=c.name, width=14).grid(row=i, column=0, sticky="w")
            cv = tk.Canvas(self.bars_frame, height=16, width=260, bg="#e6e6e6", highlightthickness=0)
            cv.grid(row=i, column=1, sticky="ew", pady=2)
            raw = cv.create_rectangle(0, 0, 0, 16, fill="#8aa4c8", outline="")
            sent = cv.create_rectangle(0, 10, 0, 16, fill="#2e6fd1", outline="")
            txt = ttk.Label(self.bars_frame, text="", width=18)
            txt.grid(row=i, column=2, sticky="w", padx=6)
            target = ttk.Label(self.bars_frame, text=c.target or "", foreground="#666")
            target.grid(row=i, column=3, sticky="w")
            self.bars[i] = (cv, raw, sent, txt)
        self.bars_frame.columnconfigure(1, weight=1)

    def _build_record_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text=t("tab_record"))
        form = ttk.Frame(tab)
        form.pack(fill="x")
        ttk.Label(form, text=t("rec_name")).grid(row=0, column=0, sticky="w")
        self.rec_name = tk.StringVar()
        ttk.Entry(form, textvariable=self.rec_name, width=24).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(form, text=t("rec_frames")).grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.rec_frames = tk.IntVar(value=self.cfg.record_frames)
        ttk.Spinbox(form, from_=256, to=16384, increment=256, textvariable=self.rec_frames, width=8).grid(
            row=0, column=3, padx=6)
        ttk.Label(form, text=t("rec_countdown")).grid(row=0, column=4, sticky="w", padx=(12, 0))
        self.rec_countdown = tk.DoubleVar(value=self.cfg.record_countdown)
        ttk.Spinbox(form, from_=0, to=30, increment=1, textvariable=self.rec_countdown, width=5).grid(
            row=0, column=5, padx=6)
        self.rec_btn = ttk.Button(form, text=t("record"), command=self.on_record)
        self.rec_btn.grid(row=0, column=6, padx=(12, 4))
        ttk.Button(form, text=t("cancel"), command=self.engine.cancel_record).grid(row=0, column=7)
        ttk.Label(tab, text=t("rec_help"), foreground="#666", wraplength=900).pack(anchor="w", pady=6)
        self.rec_progress = ttk.Progressbar(tab, maximum=1.0)
        self.rec_progress.pack(fill="x")
        self.rec_status = ttk.Label(tab, text="")
        self.rec_status.pack(anchor="w", pady=(2, 8))

        box = ttk.LabelFrame(tab, text=t("recordings"), padding=6)
        box.pack(fill="both", expand=True)
        cols = ("frames", "size")
        self.rec_tree = ttk.Treeview(box, columns=cols, show="tree headings", selectmode="extended")
        self.rec_tree.heading("#0", text=t("col_name"))
        self.rec_tree.heading("frames", text=t("rec_frames"))
        self.rec_tree.heading("size", text="MB")
        self.rec_tree.column("frames", width=90, anchor="e")
        self.rec_tree.column("size", width=90, anchor="e")
        self.rec_tree.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(box)
        side.pack(side="left", fill="y", padx=6)
        for key, cmd in (("refresh", self.refresh_recordings), ("add_to_class", self.on_add_to_class),
                         ("delete", self.on_delete_recording), ("convert_pkl", self.on_convert)):
            ttk.Button(side, text=t(key), command=cmd).pack(fill="x", pady=2)

    def _build_train_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text=t("tab_train"))
        box = ttk.LabelFrame(tab, text=t("classes"), padding=6)
        box.pack(fill="both", expand=True)
        cols = ("target", "power", "files")
        self.class_tree = ttk.Treeview(box, columns=cols, show="tree headings", height=8, selectmode="browse")
        self.class_tree.heading("#0", text=t("col_name"))
        self.class_tree.heading("target", text=t("col_target"))
        self.class_tree.heading("power", text=t("col_power"))
        self.class_tree.heading("files", text=t("col_files"))
        self.class_tree.column("#0", width=130)
        self.class_tree.column("target", width=150)
        self.class_tree.column("power", width=95, anchor="e")
        self.class_tree.column("files", width=400)
        self.class_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(box, command=self.class_tree.yview)
        self.class_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="left", fill="y")
        self.class_tree.bind("<Double-1>", lambda e: self.on_edit_class())
        side = ttk.Frame(box)
        side.pack(side="left", fill="y", padx=6)
        for key, cmd in (("add", self.on_add_class), ("edit", self.on_edit_class),
                         ("remove", self.on_remove_class), ("up", lambda: self.on_move_class(-1)),
                         ("down", lambda: self.on_move_class(1))):
            ttk.Button(side, text=t(key), command=cmd).pack(fill="x", pady=2)

        opts = ttk.Frame(tab)
        opts.pack(fill="x", pady=6)
        self.epochs = tk.IntVar(value=self.cfg.epochs)
        self.batch = tk.IntVar(value=self.cfg.batch_size)
        self.lr = tk.StringVar(value=str(self.cfg.learning_rate))
        self.amp = tk.BooleanVar(value=self.cfg.mixed_precision)
        self.cache = tk.BooleanVar(value=self.cfg.cache_datasets_in_ram)
        self.resume = tk.BooleanVar(value=False)
        ttk.Label(opts, text=t("epochs")).pack(side="left")
        ttk.Spinbox(opts, from_=1, to=500, textvariable=self.epochs, width=5).pack(side="left", padx=(4, 12))
        ttk.Label(opts, text=t("batch")).pack(side="left")
        ttk.Spinbox(opts, from_=8, to=1024, increment=8, textvariable=self.batch, width=6).pack(
            side="left", padx=(4, 12))
        ttk.Label(opts, text=t("lr")).pack(side="left")
        ttk.Entry(opts, textvariable=self.lr, width=9).pack(side="left", padx=(4, 12))
        opts2 = ttk.Frame(tab)
        opts2.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(opts2, text=t("amp"), variable=self.amp).pack(side="left")
        ttk.Checkbutton(opts2, text=t("cache"), variable=self.cache).pack(side="left", padx=12)
        ttk.Checkbutton(opts2, text=t("resume"), variable=self.resume).pack(side="left")

        ctl = ttk.Frame(tab)
        ctl.pack(fill="x")
        self.train_btn = ttk.Button(ctl, text=t("train"), command=self.on_train)
        self.train_btn.pack(side="left")
        ttk.Button(ctl, text=t("stop"), command=self.engine.cancel_training).pack(side="left", padx=4)
        ttk.Button(ctl, text=t("save_model"), command=self.on_save_model).pack(side="left", padx=(16, 4))
        ttk.Button(ctl, text=t("load_model"), command=self.on_load_model).pack(side="left")
        self.train_progress = ttk.Progressbar(tab, maximum=1.0)
        self.train_progress.pack(fill="x", pady=(8, 2))
        self.train_status = ttk.Label(tab, text="")
        self.train_status.pack(anchor="w")
        chart = ttk.LabelFrame(tab, text=t("loss"), padding=4)
        chart.pack(fill="both", expand=True, pady=(6, 0))
        self.loss_canvas = tk.Canvas(chart, height=160, bg="white", highlightthickness=0)
        self.loss_canvas.pack(fill="both", expand=True)
        self.loss_canvas.bind("<Configure>", lambda e: self._draw_loss())

    def _build_settings_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=12)
        self.nb.add(tab, text=t("tab_settings"))
        c = self.cfg
        self.settings = {
            "dataset_folder": tk.StringVar(value=c.dataset_folder),
            "model_path": tk.StringVar(value=c.model_path),
            "bind_host": tk.StringVar(value=c.bind_host),
            "face_port": tk.IntVar(value=c.face_port),
            "eye_port": tk.IntVar(value=c.eye_port),
            "proxy_port": tk.IntVar(value=c.proxy_port),
            "vrcft_port": tk.IntVar(value=c.vrcft_port),
            "max_send_rate": tk.DoubleVar(value=c.max_send_rate),
        }
        self.source_var = tk.StringVar(value=c.source)
        self.lang_var = tk.StringVar(value=c.language)
        row = 0

        def line(label, widget_factory):
            nonlocal row
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 10))
            widget_factory().grid(row=row, column=1, sticky="w", pady=3)
            row += 1

        def path_row(label, key, is_dir):
            def make():
                f = ttk.Frame(tab)
                ttk.Entry(f, textvariable=self.settings[key], width=50).pack(side="left")

                def browse():
                    if is_dir:
                        p = filedialog.askdirectory(initialdir=self.settings[key].get() or ".")
                    else:
                        p = filedialog.asksaveasfilename(defaultextension=".pt",
                                                         filetypes=[("PyTorch", "*.pt"), ("*", "*")])
                    if p:
                        self.settings[key].set(p)
                ttk.Button(f, text=t("browse"), command=browse).pack(side="left", padx=4)
                return f
            line(label, make)

        path_row(t("dataset_folder"), "dataset_folder", True)
        path_row(t("model_path"), "model_path", False)

        def source_widget():
            f = ttk.Frame(tab)
            ttk.Radiobutton(f, text=t("source_direct"), value="direct", variable=self.source_var).pack(anchor="w")
            ttk.Radiobutton(f, text=t("source_proxy"), value="proxy", variable=self.source_var).pack(anchor="w")
            return f
        line(t("source"), source_widget)
        line(t("bind_host"), lambda: ttk.Entry(tab, textvariable=self.settings["bind_host"], width=16))
        for key in ("face_port", "eye_port", "proxy_port", "vrcft_port"):
            line(t(key), lambda key=key: ttk.Entry(tab, textvariable=self.settings[key], width=8))
        line(t("max_rate"), lambda: ttk.Entry(tab, textvariable=self.settings["max_send_rate"], width=8))
        line(t("language"), lambda: ttk.Combobox(tab, textvariable=self.lang_var, values=("auto", "en", "ko"),
                                                 state="readonly", width=8))
        ttk.Button(tab, text=t("save"), command=self.on_save_settings).grid(row=row, column=1, sticky="w", pady=12)

    def _build_log_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text=self.t("tab_log"))
        self.log_text = tk.Text(tab, height=20, state="disabled", wrap="word")
        sb = ttk.Scrollbar(tab, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

    # ================================================================ refresh
    def _pump_events(self):
        try:
            while True:
                kind, *payload = self.events.get_nowait()
                handler = getattr(self, "_ev_" + kind, None)
                if handler:
                    handler(*payload)
        except queue.Empty:
            pass
        self.root.after(50, self._pump_events)

    def _ev_log(self, line):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        if int(self.log_text.index("end-1c").split(".")[0]) > 2000:
            self.log_text.delete("1.0", "500.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _ev_rec_progress(self, done, total, phase):
        self.rec_progress["value"] = done / max(1, total)
        self.rec_status.configure(text="%s  %d / %d" % (phase, done, total))

    def _ev_rec_done(self, ok, message, filename):
        self.rec_status.configure(text=message)
        self.rec_btn.state(["!disabled"])
        self.refresh_recordings()

    def _ev_train_progress(self, info):
        total = info["epochs"] * info["steps"]
        done = min(info["epoch"], info["epochs"] - 1) * info["steps"] + info["step"]
        if info.get("epoch_done"):
            done = info["epoch"] * info["steps"]
        else:
            self.loss_history.append(info["loss"])
        self.train_progress["value"] = done / max(1, total)
        elapsed = info["elapsed"]
        eta = elapsed / max(done, 1) * (total - done)
        self.train_status.configure(text="epoch %d/%d   step %d/%d   loss %.6f   avg %.6f   %.0fs (ETA %.0fs)" % (
            min(info["epoch"] + 1, info["epochs"]), info["epochs"], info["step"], info["steps"],
            info["loss"], info["avg"], elapsed, eta))
        self._draw_loss()

    def _ev_train_done(self, ok, message, history):
        self.train_btn.state(["!disabled"])
        self.train_status.configure(text=message)
        if ok:
            self._rebuild_bars()

    def _refresh_status(self):
        s = self.engine.status()
        t = self.t
        for key in ("eye", "face"):
            snap = s["streams"].get(key, {"state": "disconnected", "fps": 0})
            dot, lbl = self.status_labels[key]
            dot.itemconfigure("dot", fill=STATE_COLORS[snap["state"]])
            lbl.configure(text="%s: %s%s" % (t(key), t(snap["state"]),
                                             " (%.0f fps)" % snap["fps"] if snap["state"] == "ok" else ""))
        v = s["vrcft"]
        dot, lbl = self.status_labels["vrcft"]
        state = "ok" if v.get("connected") else "disconnected"
        dot.itemconfigure("dot", fill=STATE_COLORS[state])
        lbl.configure(text="%s: %s" % (t("vrcft"), t(state)))
        model = "model ✓" if s["model_loaded"] else "model ✗"
        if s["model_dirty"]:
            model += "*"
        self.device_label.configure(text="%s: %s   %s" % (t("device"), s["device"], model))

        if s["inferring"]:
            self.infer_btn.configure(text=t("stop_infer"))
            self.infer_stats.configure(text=t("infer_stats") % (s["infer_fps"], s["latency_ms"]))
        else:
            self.infer_btn.configure(text=t("start_infer"))
            self.infer_stats.configure(text=t("not_tracking"))
        self.fastcal_label.configure(text=s["fastcal"] or t("fastcal_help"),
                                     foreground="#c05000" if s["fastcal"] else "#666")
        if s["inferring"]:
            self._update_bars()
        self.root.after(100 if s["inferring"] else 250, self._refresh_status)

    def _update_bars(self):
        raw = self.engine.last_raw
        if raw is None:
            return
        out = self.engine.last_out
        for i, (cv, r_rect, s_rect, txt) in self.bars.items():
            if i >= len(raw):
                continue
            w = max(cv.winfo_width(), 1)
            c = self.cfg.classes[i]
            r = float(raw[i])
            cv.coords(r_rect, 0, 0, w * min(1.0, r / max(c.max_power, 1e-6)), 16)
            if i in out:
                cv.coords(s_rect, 0, 10, w * (out[i] + 1) / 2, 16)
                txt.configure(text="%.3f → %+.2f" % (r, out[i]))
            else:
                cv.coords(s_rect, 0, 10, 0, 16)
                txt.configure(text="%.3f" % r)

    def _refresh_preview(self):
        if self.show_preview.get() and self.nb.index("current") == 0:
            cams = self.engine.hub.cameras
            img = np.zeros((200, 200), dtype=np.uint8)
            if cams.get("eye") is not None:
                img[:100] = decode_camera(cams["eye"])
            if cams.get("face") is not None:
                img[100:] = decode_camera(cams["face"], flipped=True)
            if PREVIEW_SCALE > 1:
                img = img.repeat(PREVIEW_SCALE, 0).repeat(PREVIEW_SCALE, 1)
            self._photo = gray_to_photo(img)
            self.preview.itemconfigure(self.preview_image, image=self._photo)
        self.root.after(66, self._refresh_preview)

    def _draw_loss(self):
        cv = self.loss_canvas
        cv.delete("all")
        hist = self.loss_history
        w, h = cv.winfo_width(), cv.winfo_height()
        if len(hist) < 2 or w < 10:
            return
        vals = np.log10(np.maximum(np.asarray(hist[-2000:]), 1e-7))
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-6:
            hi = lo + 1
        pad = 30
        # 0.001 target line
        target = np.log10(1e-3)
        if lo <= target <= hi:
            y = h - 8 - (target - lo) / (hi - lo) * (h - 16)
            cv.create_line(pad, y, w, y, fill="#2e9d4f", dash=(4, 3))
            cv.create_text(2, y, text="1e-3", anchor="w", fill="#2e9d4f", font=("TkDefaultFont", 7))
        xs = np.linspace(pad, w - 4, len(vals))
        ys = h - 8 - (vals - lo) / (hi - lo) * (h - 16)
        cv.create_line(*np.column_stack((xs, ys)).ravel().tolist(), fill="#2e6fd1")
        cv.create_text(2, 8, text="%.0e" % 10 ** hi, anchor="w", font=("TkDefaultFont", 7))
        cv.create_text(2, h - 8, text="%.0e" % 10 ** lo, anchor="w", font=("TkDefaultFont", 7))

    def refresh_classes(self):
        self.class_tree.delete(*self.class_tree.get_children())
        for i, c in enumerate(self.cfg.classes):
            self.class_tree.insert("", "end", iid=str(i), text=c.name,
                                   values=(c.target or "", "%.4g" % c.max_power, ", ".join(c.files)))
        self._rebuild_bars()

    def refresh_recordings(self):
        self.rec_tree.delete(*self.rec_tree.get_children())
        for f in dataset_files(self.cfg):
            path = self.cfg.dataset_path(f)
            try:
                size = os.path.getsize(path)
                self.rec_tree.insert("", "end", iid=f, text=f, values=(frame_count(path), "%.0f" % (size / 2**20)))
            except OSError:
                pass

    # ================================================================ actions
    def _error(self, e):
        messagebox.showerror(self.t("error"), str(e), parent=self.root)

    def on_swap(self):
        self.engine.set_swapped(not self.cfg.swapped)

    def on_preview_toggle(self):
        self.cfg.show_preview = self.show_preview.get()
        self.engine.save_config()

    def on_smoothing(self, _value=None):
        self.cfg.smoothing = round(float(self.smoothing.get()), 2)

    def on_toggle_infer(self):
        try:
            if self.engine.inferring:
                self.engine.stop_inference()
            else:
                self.engine.start_inference()
        except Exception as e:
            self._error(e)

    def on_fastcal(self):
        try:
            self.engine.request_fastcal()
        except Exception as e:
            self._error(e)

    def on_record(self):
        try:
            self.cfg.record_frames = int(self.rec_frames.get())
            self.cfg.record_countdown = float(self.rec_countdown.get())
            self.engine.record(
                self.rec_name.get(),
                on_progress=lambda d, n, p: self.events.put(("rec_progress", d, n, p)),
                on_done=lambda ok, m, f: self.events.put(("rec_done", ok, m, f)))
            self.rec_btn.state(["disabled"])
            self.engine.save_config()
        except Exception as e:
            self._error(e)

    def _selected_class(self):
        sel = self.class_tree.selection()
        return int(sel[0]) if sel else None

    def on_add_to_class(self):
        files = list(self.rec_tree.selection())
        if not files or not self.cfg.classes:
            return
        names = [c.name for c in self.cfg.classes]
        win = tk.Toplevel(self.root)
        win.title(self.t("pick_class"))
        win.transient(self.root)
        var = tk.StringVar(value=names[0])
        ttk.Label(win, text=self.t("pick_class"), padding=8).pack()
        ttk.Combobox(win, values=names, textvariable=var, state="readonly").pack(padx=8)

        def ok():
            c = self.cfg.classes[names.index(var.get())]
            for f in files:
                if f not in c.files:
                    c.files.append(f)
            self.engine.save_config()
            self.refresh_classes()
            win.destroy()
        ttk.Button(win, text="OK", command=ok).pack(pady=8)
        win.grab_set()

    def on_delete_recording(self):
        for f in self.rec_tree.selection():
            if messagebox.askyesno(self.t("delete"), self.t("confirm_delete") % f, parent=self.root):
                try:
                    os.remove(self.cfg.dataset_path(f))
                except OSError as e:
                    self._error(e)
        self.refresh_recordings()

    def on_convert(self):
        try:
            done = self.engine.convert_pickles()
            log.info("Converted %d legacy recordings", len(done))
        except Exception as e:
            self._error(e)
        self.refresh_recordings()

    def on_add_class(self):
        d = ClassDialog(self)
        if d.result:
            self.cfg.classes.append(d.result)
            self._classes_changed()

    def on_edit_class(self):
        i = self._selected_class()
        if i is None:
            return
        d = ClassDialog(self, self.cfg.classes[i])
        if d.result:
            self.cfg.classes[i] = d.result
            self._classes_changed()

    def on_remove_class(self):
        i = self._selected_class()
        if i is not None:
            del self.cfg.classes[i]
            self._classes_changed()

    def on_move_class(self, delta):
        i = self._selected_class()
        if i is None:
            return
        j = i + delta
        if 0 <= j < len(self.cfg.classes):
            cl = self.cfg.classes
            cl[i], cl[j] = cl[j], cl[i]
            self._classes_changed()
            self.class_tree.selection_set(str(j))

    def _classes_changed(self):
        # a model trained for a different class list can't be used any more
        m = self.engine.model
        if m is not None and m.num_outputs != self.cfg.num_classes:
            self.engine.stop_inference()
        self.engine.save_config()
        self.refresh_classes()

    def on_train(self):
        try:
            self.cfg.epochs = int(self.epochs.get())
            self.cfg.batch_size = int(self.batch.get())
            self.cfg.learning_rate = float(self.lr.get())
            self.cfg.mixed_precision = bool(self.amp.get())
            self.cfg.cache_datasets_in_ram = bool(self.cache.get())
            self.engine.save_config()
            self.loss_history = []
            self.engine.train_async(
                on_progress=lambda **info: self.events.put(("train_progress", info)),
                on_done=lambda ok, m, h: self.events.put(("train_done", ok, m, h)),
                resume=bool(self.resume.get()))
            self.train_btn.state(["disabled"])
        except Exception as e:
            self._error(e)

    def on_save_model(self):
        try:
            self.engine.save_model()
        except Exception as e:
            self._error(e)

    def on_load_model(self):
        path = filedialog.askopenfilename(initialfile=self.cfg.model_path,
                                          filetypes=[("PyTorch", "*.pt"), ("*", "*")])
        if not path:
            return
        try:
            self.engine.load_model(path)
            self._rebuild_bars()
        except Exception as e:
            self._error(e)

    def on_save_settings(self):
        try:
            for key, var in self.settings.items():
                setattr(self.cfg, key, var.get())
            self.cfg.source = self.source_var.get()
            self.cfg.language = self.lang_var.get()
            self.engine.save_config()
            self.refresh_recordings()
            messagebox.showinfo(self.t("save"), self.t("saved_restart"), parent=self.root)
        except Exception as e:
            self._error(e)

    def on_close(self):
        if self.engine.model_dirty and not messagebox.askyesno(self.t("title"), self.t("unsaved_quit"),
                                                               parent=self.root):
            return
        self.engine.save_config()
        self.engine.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_gui(engine):
    App(engine).run()
