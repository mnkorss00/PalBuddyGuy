"""Entry point: GUI by default, `--cli` for the original text interface."""

import argparse
import logging
import sys
import threading
import time

from . import __version__
from .config import DEFAULT_CONFIG_PATH, Config

log = logging.getLogger("palbuddy")

HELP = """commands:
  status             connection / fps overview
  swap               swap eye and face streams (saved in config.json)
  record [name]      record a dataset
  train              train on the classes in config.json
  save / load        save / load the model
  infer              start sending tracking to VRCFT (enter 'stop' to end)
  fastcal            calibrate max power by puppeting the avatar
  stop               stop inference / training / recording
  stats              frame rate over 5 seconds
  convertmmap        convert old .pkl recordings to .mmap
  quit"""


def run_cli(engine):
    print("Pal Buddy Guy %s - type 'help' for commands" % __version__)

    def progress_printer(prefix):
        def fn(done, total, phase):
            print("\r%s %s %d / %d      " % (prefix, phase, done, total), end="", flush=True)
        return fn

    def train_progress(**i):
        print("\repoch %d/%d step %d/%d  loss %.6f  avg %.6f     " % (
            min(i["epoch"] + 1, i["epochs"]), i["epochs"], i["step"], i["steps"], i["loss"], i["avg"]),
            end="", flush=True)

    done_event = threading.Event()

    def finished(ok, message, *_):
        print("\n" + message)
        done_event.set()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "quit"
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        try:
            if cmd == "help":
                print(HELP)
            elif cmd == "status":
                s = engine.status()
                for role, st in s["streams"].items():
                    print("  %-5s %-12s %5.1f fps  %s" % (role, st["state"], st["fps"], st.get("peer") or ""))
                print("  vrcft %s  device %s  inferring %s  model %s" % (
                    "connected" if s["vrcft"]["connected"] else "disconnected", s["device"], s["inferring"],
                    "loaded" if s["model_loaded"] else "-"))
            elif cmd == "swap":
                engine.set_swapped(not engine.cfg.swapped)
                print("swapped = %s" % engine.cfg.swapped)
            elif cmd == "record":
                name = arg or input("dataset name: ")
                done_event.clear()
                engine.record(name, on_progress=progress_printer("recording"), on_done=finished)
                done_event.wait()
            elif cmd == "train":
                done_event.clear()
                engine.train_async(on_progress=train_progress, on_done=finished)
                done_event.wait()
            elif cmd == "save":
                engine.save_model()
            elif cmd == "load":
                engine.load_model(arg or None)
            elif cmd == "infer":
                engine.start_inference()
                print("tracking started, 'stop' to end")
            elif cmd == "fastcal":
                engine.request_fastcal()
            elif cmd == "stop":
                engine.stop_inference()
                engine.cancel_training()
                engine.cancel_record()
            elif cmd == "stats":
                time.sleep(5)
                print("%.1f samples/s" % engine.hub.sample_rate.rate())
            elif cmd == "convertmmap":
                print("converted: %s" % engine.convert_pickles())
            elif cmd in ("quit", "exit"):
                if engine.model_dirty and input("model not saved, quit anyway? [y/N] ").lower() != "y":
                    continue
                break
            else:
                print("unknown command, type 'help'")
        except Exception as e:
            print("error: %s" % e)
    engine.stop()


def main(argv=None):
    p = argparse.ArgumentParser(prog="palbuddy", description="Vive Pro Eye / facial tracker expression trainer")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="path to config.json")
    p.add_argument("--cli", action="store_true", help="text interface instead of the GUI")
    p.add_argument("--infer", action="store_true", help="start tracking immediately (headless use)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")
    cfg = Config.load(args.config)
    for problem in cfg.validate():
        log.warning(problem)

    from .engine import Engine  # imports torch; keep --help fast
    engine = Engine(cfg, args.config)
    try:
        engine.start()
    except OSError as e:
        log.error("Could not open a port (%s). Is another copy of the script or tvm_proxy.py running?", e)
        return 1
    if args.infer:
        try:
            engine.start_inference()
        except Exception as e:
            log.error("Can't start tracking: %s", e)

    if args.cli:
        run_cli(engine)
        return 0
    try:
        from .gui import run_gui
    except ImportError as e:
        log.error("GUI unavailable (%s); falling back to --cli", e)
        run_cli(engine)
        return 0
    run_gui(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
