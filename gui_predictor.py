import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading, os, sys
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    sys.exit("pip install Pillow")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    sys.exit("pip install tensorflow")

MODEL_PATH  = "outputs/model_plain.keras"
IMG_SIZE    = (32, 32)
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat",
    "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
]

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
ACCENT   = "#4ade80"
ACCENT2  = "#60a5fa"
TEXT_LIGHT = "#e6edf3"
TEXT_DIM   = "#8b949e"
DANGER   = "#f87171"
WARN     = "#fb923c"

class PredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CIFAR-10 Image Classifier — AI Project")
        self.geometry("960x720")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)
        self.model = None
        self.current_img = None
        self._build_ui()
        self._load_model_thread()

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg="#0d1f12", pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="CIFAR-10 Image Classifier",
                 font=("Segoe UI", 20, "bold"),
                 fg=ACCENT, bg="#0d1f12").pack()
        tk.Label(hdr,
                 text="Advanced Data Augmentation Project  "
                      "Zain Tahir  &  Huzaifa Khan",
                 font=("Segoe UI", 11),
                 fg=TEXT_DIM, bg="#0d1f12").pack(pady=(2,0))

    def _build_body(self):
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=PANEL_BG,
                        highlightbackground="#30363d",
                        highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self._build_left_panel(left)

        right = tk.Frame(body, bg=PANEL_BG,
                         highlightbackground="#30363d",
                         highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        tk.Label(parent, text="IMAGE INPUT",
                 font=("Segoe UI", 9, "bold"),
                 fg=ACCENT, bg=PANEL_BG).pack(
                     anchor="w", padx=16, pady=(14,4))

        self.canvas = tk.Canvas(parent, width=320, height=320,
                                bg="#0d1117", highlightthickness=0)
        self.canvas.pack(padx=16, pady=8)
        self._draw_placeholder()

        btn_frame = tk.Frame(parent, bg=PANEL_BG)
        btn_frame.pack(pady=12, padx=16, fill="x")

        self.upload_btn = tk.Button(
            btn_frame, text="Upload Image",
            command=self._upload_image,
            bg=ACCENT2, fg=DARK_BG,
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=18, pady=8,
            cursor="hand2")
        self.upload_btn.pack(side="left", fill="x",
                             expand=True, padx=(0,6))

        self.predict_btn = tk.Button(
            btn_frame, text="Predict",
            command=self._predict,
            bg=ACCENT, fg=DARK_BG,
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=18, pady=8,
            cursor="hand2", state="disabled")
        self.predict_btn.pack(side="right", fill="x",
                              expand=True)

        self.img_info_var = tk.StringVar(value="No image loaded")
        tk.Label(parent, textvariable=self.img_info_var,
                 font=("Consolas", 10),
                 fg=TEXT_DIM, bg=PANEL_BG).pack(pady=(0,10))

    def _build_right_panel(self, parent):
        tk.Label(parent, text="PREDICTION RESULTS",
                 font=("Segoe UI", 9, "bold"),
                 fg=ACCENT, bg=PANEL_BG).pack(
                     anchor="w", padx=16, pady=(14,4))

        pred_box = tk.Frame(parent, bg="#0d2818",
                            highlightbackground="#4ade80",
                            highlightthickness=1)
        pred_box.pack(fill="x", padx=16, pady=(4,12))

        tk.Label(pred_box, text="TOP PREDICTION",
                 font=("Segoe UI", 8, "bold"),
                 fg=TEXT_DIM, bg="#0d2818").pack(pady=(12,2))

        self.pred_label = tk.Label(pred_box, text="-- --",
                                   font=("Segoe UI", 22, "bold"),
                                   fg=ACCENT, bg="#0d2818")
        self.pred_label.pack()

        self.conf_label = tk.Label(pred_box,
                                   text="Confidence: --",
                                   font=("Segoe UI", 12),
                                   fg=TEXT_LIGHT, bg="#0d2818")
        self.conf_label.pack(pady=(0,14))

        tk.Label(parent, text="CLASS PROBABILITIES",
                 font=("Segoe UI", 9, "bold"),
                 fg=ACCENT, bg=PANEL_BG).pack(
                     anchor="w", padx=16, pady=(0,6))

        bars_frame = tk.Frame(parent, bg=PANEL_BG)
        bars_frame.pack(fill="x", padx=16)

        self.bar_pbars  = []
        self.bar_labels = []

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Horizontal.TProgressbar",
                        troughcolor="#1a1a2e",
                        background=ACCENT,
                        bordercolor=PANEL_BG)

        for name in CLASS_NAMES:
            row = tk.Frame(bars_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=2)

            tk.Label(row, text=name,
                     font=("Segoe UI", 9),
                     fg=TEXT_LIGHT, bg=PANEL_BG,
                     width=12, anchor="w").pack(side="left")

            bar = ttk.Progressbar(row, length=180,
                                  mode="determinate",
                                  maximum=100)
            bar.pack(side="left", padx=(4,8))

            pct = tk.Label(row, text="0.0%",
                           font=("Consolas", 10),
                           fg=TEXT_DIM, bg=PANEL_BG,
                           width=6)
            pct.pack(side="left")

            self.bar_pbars.append(bar)
            self.bar_labels.append(pct)

    def _build_status_bar(self):
        status_bar = tk.Frame(self, bg="#010409", pady=5)
        status_bar.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(
            value="Loading model...")
        tk.Label(status_bar,
                 textvariable=self.status_var,
                 font=("Consolas", 10),
                 fg=TEXT_DIM,
                 bg="#010409").pack(side="left", padx=12)

        self.model_status = tk.Label(
            status_bar, text="Model: Loading",
            font=("Consolas", 10),
            fg=WARN, bg="#010409")
        self.model_status.pack(side="right", padx=12)

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, 320, 320,
            fill="#0d1117", outline="")
        self.canvas.create_text(
            160, 160,
            text="Upload an image to begin",
            font=("Segoe UI", 11),
            fill="#484f58")

    def _load_model_thread(self):
        t = threading.Thread(
            target=self._load_model, daemon=True)
        t.start()

    def _load_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                self.after(0, lambda: self.status_var.set(
                    "Model not found! Run train_once.py first."))
                self.after(0, lambda: self.model_status.config(
                    text="Model: Not Found", fg=DANGER))
                return
            self.model = load_model(MODEL_PATH)
            self.after(0, lambda: self.status_var.set(
                "Model loaded! Upload an image."))
            self.after(0, lambda: self.model_status.config(
                text="Model: Ready", fg=ACCENT))
            self.after(0, lambda: self.upload_btn.config(
                state="normal"))
        except Exception as e:
            self.after(0, lambda: self.status_var.set(
                f"Error: {e}"))
            self.after(0, lambda: self.model_status.config(
                text="Model: Error", fg=DANGER))

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files",
                 "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.current_img = img

            self.img_info_var.set(
                f"Size: {img.size[0]}x{img.size[1]}  "
                f"{os.path.basename(path)}")

            display = img.copy()
            display.thumbnail((300, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(display)
            self.canvas.delete("all")
            cx = (320 - display.size[0]) // 2
            cy = (320 - display.size[1]) // 2
            self.canvas.create_rectangle(
                0, 0, 320, 320,
                fill="#0d1117", outline="")
            self.canvas.create_image(
                cx, cy, anchor="nw", image=photo)
            self.canvas._photo = photo

            self.pred_label.config(
                text="-- --", fg=TEXT_DIM)
            self.conf_label.config(
                text="Confidence: --")
            for bar, lbl in zip(
                    self.bar_pbars, self.bar_labels):
                bar["value"] = 0
                lbl.config(text="0.0%")

            if self.model:
                self.predict_btn.config(state="normal")

            self.status_var.set(
                f"Image loaded: {os.path.basename(path)}")

        except Exception as e:
            messagebox.showerror("Image Error", str(e))

    def _predict(self):
        if not self.model or not self.current_img:
            return
        self.predict_btn.config(
            state="disabled", text="Predicting...")
        self.status_var.set("Running prediction...")
        t = threading.Thread(
            target=self._run_inference, daemon=True)
        t.start()

    def _run_inference(self):
        try:
            img_resized = self.current_img.resize(
                IMG_SIZE, Image.LANCZOS)
            arr = np.array(img_resized,
                           dtype="float32") / 255.0
            arr = arr[np.newaxis, ...]

            probs    = self.model.predict(
                arr, verbose=0)[0]
            top_idx  = int(np.argmax(probs))
            top_conf = float(probs[top_idx]) * 100

            self.after(0, self._display_results,
                       probs, top_idx, top_conf)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror(
                "Error", str(e)))
            self.after(0, lambda: self.predict_btn.config(
                state="normal", text="Predict"))

    def _display_results(self, probs, top_idx, top_conf):
        self.pred_label.config(
            text=CLASS_NAMES[top_idx], fg=ACCENT)
        self.conf_label.config(
            text=f"Confidence: {top_conf:.2f}%",
            fg=ACCENT if top_conf >= 60
            else WARN if top_conf >= 40
            else DANGER)

        for i, (bar, lbl) in enumerate(
                zip(self.bar_pbars, self.bar_labels)):
            pct = float(probs[i]) * 100
            bar["value"] = pct
            lbl.config(
                text=f"{pct:.1f}%",
                fg=ACCENT if i == top_idx else TEXT_DIM)

        self.predict_btn.config(
            state="normal", text="Predict")
        self.status_var.set(
            f"Done! → {CLASS_NAMES[top_idx]} "
            f"({top_conf:.2f}% confidence)")

if __name__ == "__main__":
    app = PredictorApp()
    app.mainloop()