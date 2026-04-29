import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import os
from src.analyzer import FaceAnalyzer

# Modern UI Colors
COLOR_BG      = "#0a0b1e"
COLOR_PANEL   = "#11122a"
COLOR_ACCENT  = "#6366f1"
COLOR_ACCENT_HOVER = "#4f46e5"
COLOR_TEXT    = "#f8fafc"
COLOR_SUBTEXT = "#94a3b8"

class AgeScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Intelligence - AI Анализатор")
        self.root.geometry("1100x750")
        self.root.configure(bg=COLOR_BG)
        
        self.analyzer = None
        self._last_frame = None
        self._photo = None
        self._camera_active = False
        self._camera_capture = None
        self._camera_thread = None
        self._camera_frame_index = 0
        self._camera_last_annotated = None
        self._camera_last_results = []
        
        self._setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Load analyzer in background
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _init_engine(self):
        try:
            self.analyzer = FaceAnalyzer()
            self.root.after(0, self._on_engine_ready)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить модели: {e}"))

    def _setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg=COLOR_PANEL, height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="FACE INTELLIGENCE", font=("Inter", 24, "bold"), 
                 fg=COLOR_ACCENT, bg=COLOR_PANEL).pack(pady=(12, 0))
        tk.Label(header, text="Система интеллектуального анализа пола и возраста", 
                 font=("Inter", 10), fg=COLOR_SUBTEXT, bg=COLOR_PANEL).pack()

        # Main Content
        main_frame = tk.Frame(self.root, bg=COLOR_BG, padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control Panel
        ctrl_frame = tk.Frame(main_frame, bg=COLOR_BG)
        ctrl_frame.pack(fill=tk.X, pady=(0, 20))

        self.btn_open = tk.Button(
            ctrl_frame, text="ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ", 
            font=("Inter", 12, "bold"),
            bg=COLOR_ACCENT, fg="white",
            activebackground=COLOR_ACCENT_HOVER, activeforeground="white",
            relief=tk.FLAT, padx=30, pady=12,
            cursor="hand2", command=self._open_file,
            state=tk.DISABLED
        )
        self.btn_open.pack(side=tk.LEFT)

        self.btn_camera = tk.Button(
            ctrl_frame, text="СТАРТ КАМЕРЫ",
            font=("Inter", 12, "bold"),
            bg="#0ea5a4", fg="white",
            activebackground="#0d9488", activeforeground="white",
            relief=tk.FLAT, padx=20, pady=12,
            cursor="hand2", command=self._toggle_camera,
            state=tk.DISABLED
        )
        self.btn_camera.pack(side=tk.LEFT, padx=(12, 0))
        
        self.status_label = tk.Label(ctrl_frame, text="Инициализация моделей...", 
                                     font=("Inter", 11), fg=COLOR_SUBTEXT, bg=COLOR_BG)
        self.status_label.pack(side=tk.LEFT, padx=30)

        # Viewer
        viewer_wrap = tk.Frame(main_frame, bg=COLOR_PANEL, bd=1, highlightthickness=0)
        viewer_wrap.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(viewer_wrap, bg="#050614", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

        # Info Footer
        self.footer = tk.Label(self.root, text="Готов к работе", 
                               font=("Inter", 10), fg="#475569", bg=COLOR_PANEL, 
                               pady=8, padx=20, anchor="w")
        self.footer.pack(fill=tk.X, side=tk.BOTTOM)

    def _on_engine_ready(self):
        self.btn_open.config(state=tk.NORMAL)
        self.btn_camera.config(state=tk.NORMAL)
        self.status_label.config(text="Система готова")
        self._show_placeholder("Выберите изображение для анализа")

    def _show_placeholder(self, text):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 1000
        ch = self.canvas.winfo_height() or 500
        self.canvas.create_text(cw//2, ch//2, text=text, fill="#1e293b", font=("Inter", 16, "italic"))

    def _open_file(self):
        if self._camera_active:
            self._stop_camera()

        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path: return
        
        self.btn_open.config(state=tk.DISABLED, text="ОБРАБОТКА...")
        self.status_label.config(text="Анализируем лица...")
        
        threading.Thread(target=self._process_image, args=(path,), daemon=True).start()

    def _toggle_camera(self):
        if self._camera_active:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self.analyzer is None:
            messagebox.showwarning("Подождите", "Модели еще не инициализированы.")
            return

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Ошибка камеры", "Не удалось открыть веб-камеру.")
            return

        self._camera_capture = cap
        self._camera_active = True
        self._camera_frame_index = 0
        self._camera_last_annotated = None
        self._camera_last_results = []

        self.btn_open.config(state=tk.DISABLED)
        self.btn_camera.config(text="СТОП КАМЕРЫ", bg="#dc2626", activebackground="#b91c1c")
        self.status_label.config(text="Камера запущена: анализ в реальном времени")
        self.footer.config(text="Наведите лицо на камеру")

        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()

    def _stop_camera(self):
        self._camera_active = False
        cap = self._camera_capture
        self._camera_capture = None
        if cap is not None:
            cap.release()

        self.btn_open.config(state=tk.NORMAL if self.analyzer else tk.DISABLED, text="ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ")
        self.btn_camera.config(
            state=tk.NORMAL if self.analyzer else tk.DISABLED,
            text="СТАРТ КАМЕРЫ",
            bg="#0ea5a4",
            activebackground="#0d9488"
        )
        self.status_label.config(text="Система готова")
        self.footer.config(text="Готов к работе")

    def _camera_loop(self):
        try:
            while self._camera_active and self._camera_capture is not None:
                ok, frame = self._camera_capture.read()
                if not ok:
                    break

                self._camera_frame_index += 1
                # Analyze every 3rd frame to keep webcam mode responsive.
                if self._camera_frame_index % 3 == 0:
                    annotated, results = self.analyzer.detect_and_analyze(frame)
                    self._camera_last_annotated = annotated
                    self._camera_last_results = results
                elif self._camera_last_annotated is None:
                    self._camera_last_annotated = frame
                    self._camera_last_results = []

                annotated_to_show = self._camera_last_annotated
                results_to_show = self._camera_last_results
                self.root.after(0, lambda f=annotated_to_show, r=results_to_show: self._display_camera_frame(f, r))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка камеры", str(e)))
        finally:
            self.root.after(0, self._stop_camera)

    def _display_camera_frame(self, frame, results):
        if not self._camera_active:
            return
        self._last_frame = frame
        self._render_frame(frame)
        if results:
            summary = " | ".join([f"Лицо {i+1}: возраст {r['age_category']}" for i, r in enumerate(results)])
            self.status_label.config(text=f"Камера: найдено лиц {len(results)}")
            self.footer.config(text=summary)
        else:
            self.status_label.config(text="Камера: лицо не найдено")
            self.footer.config(text="Наведите лицо в центр кадра")

    def _process_image(self, path):
        try:
            frame = self._read_image_safe(path)
            if frame is None: raise Exception("Не удалось прочитать файл")
            
            # Use analyzer
            annotated, results = self.analyzer.detect_and_analyze(frame)
            
            self.root.after(0, lambda: self._display_result(annotated, results))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _read_image_safe(self, path):
        """Read image robustly on Windows paths with non-ASCII characters."""
        frame = cv2.imread(path)
        if frame is not None:
            return frame
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _display_result(self, frame, results):
        self._last_frame = frame
        self._render_frame(frame)
        
        if results:
            summary = " | ".join([
                f"Лицо {i+1}: возраст {r['age_category']}"
                for i, r in enumerate(results)
            ])
            self.status_label.config(text=f"Найдено: {len(results)}")
            self.footer.config(text=summary)
        else:
            self.status_label.config(text="Лица не найдены")
            self.footer.config(text="Попробуйте другое изображение")
            
        self.btn_open.config(state=tk.NORMAL, text="ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ")

    def _on_error(self, msg):
        messagebox.showerror("Ошибка", msg)
        self.btn_open.config(state=tk.NORMAL, text="ЗАГРУЗИТЬ ИЗОБРАЖЕНИЕ")
        self.status_label.config(text="Ошибка обработки")

    def _render_frame(self, frame):
        if frame is None: return
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10: return
        
        ratio = min(cw/pil_img.width, ch/pil_img.height)
        new_w, new_h = int(pil_img.width * ratio), int(pil_img.height * ratio)
        
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self._photo, anchor=tk.CENTER)

    def _on_resize(self, _):
        if self._last_frame is not None:
            self._render_frame(self._last_frame)
        elif self.analyzer:
            self._show_placeholder("Выберите изображение для анализа")

    def _on_close(self):
        self._stop_camera()
        self.root.destroy()
