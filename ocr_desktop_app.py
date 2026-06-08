"""
ocr_desktop_app.py
Scanned PDF → Searchable OCR PDF Converter
- Single-pass OCR (image_to_data only; text reconstructed from word data)
- Optional PDF splitting by 50 MB size limit after OCR
- Outputs: searchable PDF, split parts (optional), JSON, TXT
- Dead code removed; output folder wired up properly
"""

import sys
import os
import shutil
from pathlib import Path
import json
import re
import threading
import queue
from io import BytesIO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import fitz  # PyMuPDF
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output
from pypdf import PdfReader, PdfWriter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TESSERACT_PATH = ""
SPLIT_SIZE_LIMIT_MB = 50


# ---------------------------------------------------------------------------
# Project root — works for plain Python and PyInstaller EXE
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the folder beside the EXE (frozen) or the script (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = get_project_root()
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OCR_TEXT_DIR = DATA_DIR / "ocr_text"

for _folder in (UPLOADS_DIR, OCR_TEXT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def make_safe_filename(name: str) -> str:
    """Strip extension and replace non-alphanumeric chars with underscores."""
    stem = Path(name).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem)
    return safe.strip("_")


def save_ocr_json(path: Path, pages_data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)


def save_ocr_text(path: Path, pages_data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for page in pages_data:
            f.write(f"\n\n--- Page {page['page']} ---\n")
            f.write(page.get("text", ""))


def words_to_lines(words: list, y_tolerance: int = 12) -> list:
    """
    Group Tesseract word-level bounding boxes into logical lines.
    Uses a running centroid so multi-word lines stay together even
    when individual word baselines differ slightly.
    """
    if not words:
        return []

    sorted_words = sorted(
        words,
        key=lambda w: (w["top"] + w["height"] / 2, w["left"])
    )

    line_groups = []

    for word in sorted_words:
        center_y = word["top"] + word["height"] / 2
        placed = False

        for group in line_groups:
            if abs(center_y - group["center_y"]) <= y_tolerance:
                n = len(group["words"])
                group["center_y"] = (group["center_y"] * n + center_y) / (n + 1)
                group["words"].append(word)
                placed = True
                break

        if not placed:
            line_groups.append({"center_y": center_y, "words": [word]})

    lines = []
    for idx, group in enumerate(line_groups, start=1):
        line_words = sorted(group["words"], key=lambda w: w["left"])
        lines.append({
            "line_no": idx,
            "text": " ".join(w["text"] for w in line_words),
            "top": min(w["top"] for w in line_words),
            "left": min(w["left"] for w in line_words),
            "words": line_words,
        })

    return lines


def split_pdf_by_size(
    source_pdf: Path,
    output_dir: Path,
    base_name: str,
    size_limit_mb: float = SPLIT_SIZE_LIMIT_MB,
    log_fn=None,
) -> list:
    """
    Split *source_pdf* into parts so that each part stays under
    *size_limit_mb* MB.  Parts are written as:
        <base_name>_part_01.pdf, _part_02.pdf, …

    Strategy
    --------
    Pages are added to the current part one at a time.  After each
    addition the in-memory size is estimated by writing to a BytesIO
    buffer.  When the next page would push it over the limit the
    current part is flushed to disk and a new writer is started.
    This never produces a part that exceeds the limit (unless a
    single page itself exceeds it, in which case it gets its own file
    with a warning).

    Returns a list of Path objects for every part written.
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)
        print(msg)

    limit_bytes = int(size_limit_mb * 1024 * 1024)
    reader = PdfReader(str(source_pdf))
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise RuntimeError("Source PDF has no pages to split.")

    output_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    part_num = 1
    writer = PdfWriter()
    pages_in_part = []

    def _flush_part():
        nonlocal writer, part_num, pages_in_part
        part_path = output_dir / f"{base_name}_part_{part_num:02d}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)
        size_mb = part_path.stat().st_size / (1024 * 1024)
        _log(
            f"  Part {part_num:02d}: pages {pages_in_part[0]}–{pages_in_part[-1]} "
            f"({len(pages_in_part)} pages, {size_mb:.2f} MB) → {part_path.name}"
        )
        parts.append(part_path)
        part_num += 1
        writer = PdfWriter()
        pages_in_part = []

    def _estimate_size(w: PdfWriter) -> int:
        """Estimate current writer size without touching disk."""
        buf = BytesIO()
        w.write(buf)
        return buf.tell()

    for page_index in range(total_pages):
        page = reader.pages[page_index]
        page_no = page_index + 1

        # Try adding this page to the current part
        writer.add_page(page)
        pages_in_part.append(page_no)

        current_size = _estimate_size(writer)

        if current_size >= limit_bytes:
            if len(pages_in_part) == 1:
                # Single page already exceeds the limit — warn and flush anyway
                _log(
                    f"  Warning: page {page_no} alone is "
                    f"{current_size / (1024*1024):.2f} MB (> {size_limit_mb} MB limit). "
                    "Writing as its own part."
                )
                _flush_part()
            else:
                # Remove the page that pushed us over and flush without it,
                # then start a new part with that page.
                writer2 = PdfWriter()
                for p in writer.pages[:-1]:
                    writer2.add_page(p)
                last_page = writer.pages[-1]
                last_page_no = pages_in_part[-1]

                writer = writer2
                pages_in_part = pages_in_part[:-1]
                _flush_part()

                # Start new part with the page that didn't fit
                writer.add_page(last_page)
                pages_in_part = [last_page_no]

    # Flush whatever remains
    if pages_in_part:
        _flush_part()

    return parts


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class OCRDesktopApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Scanned PDF → Searchable OCR PDF")
        self.root.geometry("800x620")
        self.root.resizable(False, False)

        # --- tk variables ---
        self.input_pdf      = tk.StringVar()
        self.output_folder  = tk.StringVar(value=str(UPLOADS_DIR))
        self.language       = tk.StringVar(value="eng")
        self.dpi            = tk.IntVar(value=300)
        self.tesseract_path = tk.StringVar(value=DEFAULT_TESSERACT_PATH)
        self.do_split       = tk.BooleanVar(value=False)
        self.split_size_mb  = tk.DoubleVar(value=SPLIT_SIZE_LIMIT_MB)

        self.log_queue = queue.Queue()
        self._cancel_flag = threading.Event()
        self._last_json_path = None

        self._build_ui()
        self.root.after(200, self._process_log_queue)

    # ------------------------------------------------------------------
    # Tesseract path resolution
    # ------------------------------------------------------------------

    def _app_base_path(self) -> Path:
        """Return PyInstaller's _MEIPASS or the script's own folder."""
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(__file__).parent

    def _resolve_tesseract(self, manual: str = "") -> tuple:
        """
        Return (tesseract_exe, tessdata_dir) by checking four locations
        in priority order:
          1. Manually typed / browsed path from the GUI
          2. vendor/Tesseract-OCR  (dev checkout)
          3. _MEIPASS/Tesseract-OCR  (PyInstaller bundle)
          4. C:\\Program Files\\Tesseract-OCR  (system install)
        """
        def _check(root: Path):
            exe = root / "tesseract.exe"
            td  = root / "tessdata"
            if exe.exists() and td.exists():
                return str(exe), str(td)
            return None

        if manual:
            exe = Path(manual)
            if not exe.exists():
                raise FileNotFoundError(f"tesseract.exe not found at:\n{exe}")
            td = exe.parent / "tessdata"
            if not td.exists():
                raise FileNotFoundError(
                    f"tessdata folder not found beside tesseract.exe:\n{td}"
                )
            return str(exe), str(td)

        base = self._app_base_path()

        exe_dir = Path(sys.executable).parent

        for candidate_root in [
            base / "vendor" / "Tesseract-OCR",
            base / "Tesseract-OCR",
            exe_dir / "Tesseract-OCR",           # side-by-side with EXE
            Path(r"C:\Program Files\Tesseract-OCR"),
        ]:
            result = _check(candidate_root)
            if result:
                return result

        raise FileNotFoundError(
            "Tesseract was not found.\n\n"
            "Checked:\n"
            f"  • {base / 'vendor' / 'Tesseract-OCR'}\n"
            f"  • {base / 'Tesseract-OCR'}\n"
            f"  • {exe_dir / 'Tesseract-OCR'}\n"
            "  • C:\\Program Files\\Tesseract-OCR\n\n"
            "Fix: place Tesseract-OCR in vendor\\Tesseract-OCR or "
            "select tesseract.exe manually in the GUI."
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        tk.Label(
            self.root,
            text="Scanned PDF → Searchable OCR PDF",
            font=("Segoe UI", 17, "bold"),
        ).pack(pady=(14, 8))

        # ── File paths ──────────────────────────────────────────────
        paths_frame = tk.Frame(self.root)
        paths_frame.pack(fill="x", padx=25)

        def _row(parent, label, var, browse_cmd, row):
            tk.Label(parent, text=label, font=("Segoe UI", 10, "bold")).grid(
                row=row * 2, column=0, sticky="w", pady=(10, 0)
            )
            tk.Entry(parent, textvariable=var, width=76).grid(
                row=row * 2 + 1, column=0, padx=(0, 8), pady=3
            )
            tk.Button(parent, text="Browse", command=browse_cmd, width=8).grid(
                row=row * 2 + 1, column=1
            )

        _row(paths_frame, "Input scanned PDF:",     self.input_pdf,      self._browse_pdf,       0)
        _row(paths_frame, "Output folder:",         self.output_folder,   self._browse_out_folder, 1)
        _row(paths_frame, "Tesseract EXE path (optional):", self.tesseract_path, self._browse_tesseract, 2)

        # ── OCR settings row ────────────────────────────────────────
        settings = tk.Frame(self.root)
        settings.pack(fill="x", padx=25, pady=(12, 4))

        tk.Label(settings, text="OCR Language:", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Combobox(
            settings,
            textvariable=self.language,
            width=16,
            values=["eng", "ara", "eng+ara"],
            state="readonly",
        ).grid(row=0, column=1, padx=(6, 30))

        tk.Label(settings, text="DPI:", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=2, sticky="w"
        )
        ttk.Combobox(
            settings,
            textvariable=self.dpi,
            width=8,
            values=[200, 250, 300, 400],
            state="readonly",
        ).grid(row=0, column=3, padx=6)

        # ── Split settings row ──────────────────────────────────────
        split_frame = tk.Frame(self.root)
        split_frame.pack(fill="x", padx=25, pady=(4, 0))

        self._split_check = tk.Checkbutton(
            split_frame,
            text="Split output PDF into parts (max size per part):",
            variable=self.do_split,
            font=("Segoe UI", 10),
            command=self._on_split_toggle,
        )
        self._split_check.grid(row=0, column=0, sticky="w")

        self._split_size_entry = tk.Spinbox(
            split_frame,
            textvariable=self.split_size_mb,
            from_=1,
            to=500,
            increment=5,
            width=7,
            format="%.0f",
            font=("Segoe UI", 10),
            state="disabled",
        )
        self._split_size_entry.grid(row=0, column=1, padx=(8, 4))

        tk.Label(split_frame, text="MB", font=("Segoe UI", 10)).grid(
            row=0, column=2, sticky="w"
        )

        # ── Start / Cancel buttons ──────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=25, pady=10)

        self.start_button = tk.Button(
            btn_frame,
            text="Start OCR Conversion",
            command=self._start_thread,
            bg="#1f6feb",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            height=2,
        )
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.cancel_button = tk.Button(
            btn_frame,
            text="Cancel",
            command=self._cancel,
            bg="#d9534f",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            height=2,
            state="disabled",
        )
        self.cancel_button.pack(side="left", fill="x", expand=True)

        self.ingest_button = tk.Button(
            btn_frame,
            text="Ingest to ChromaDB",
            command=self._ingest_to_chroma,
            bg="#238636",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            height=2,
            state="disabled",
        )
        self.ingest_button.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # ── Progress bar ────────────────────────────────────────────
        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", length=750, mode="determinate"
        )
        self.progress.pack(padx=25, pady=(4, 2))

        self._progress_label = tk.Label(
            self.root, text="", font=("Segoe UI", 9), fg="#555"
        )
        self._progress_label.pack(anchor="w", padx=25)

        # ── Log box ─────────────────────────────────────────────────
        tk.Label(
            self.root, text="Process Log:", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=25, pady=(6, 0))

        log_frame = tk.Frame(self.root)
        log_frame.pack(fill="both", expand=True, padx=25, pady=(2, 14))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")

        self.log_box = tk.Text(
            log_frame,
            height=12,
            yscrollcommand=scrollbar.set,
            state="disabled",
            font=("Consolas", 9),
        )
        self.log_box.pack(fill="both", expand=True)
        scrollbar.config(command=self.log_box.yview)

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def _on_split_toggle(self):
        state = "normal" if self.do_split.get() else "disabled"
        self._split_size_entry.config(state=state)

    def _browse_pdf(self):
        path = filedialog.askopenfilename(
            title="Select scanned PDF", filetypes=[("PDF Files", "*.pdf")]
        )
        if path:
            self.input_pdf.set(path)

    def _browse_out_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_folder.set(folder)

    def _browse_tesseract(self):
        path = filedialog.askopenfilename(
            title="Select tesseract.exe",
            filetypes=[("Tesseract EXE", "tesseract.exe"), ("EXE Files", "*.exe")],
        )
        if path:
            self.tesseract_path.set(path)

    def _cancel(self):
        self._cancel_flag.set()
        self.log("Cancellation requested — stopping after current page…")
        self.cancel_button.config(state="disabled")

    # ------------------------------------------------------------------
    # Logging helpers (thread-safe)
    # ------------------------------------------------------------------

    def log(self, message: str):
        self.log_queue.put(message)

    def _process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.root.after(200, self._process_log_queue)

    def _set_progress(self, value: int, maximum: int = None, label: str = ""):
        def _update():
            if maximum is not None:
                self.progress.config(maximum=maximum)
            self.progress.config(value=value)
            if label:
                self._progress_label.config(text=label)
        self.root.after(0, _update)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate(self) -> bool:
        pdf = self.input_pdf.get().strip()
        if not pdf:
            messagebox.showerror("Missing input", "Please select a scanned PDF.")
            return False
        if not Path(pdf).exists():
            messagebox.showerror("File not found", f"PDF not found:\n{pdf}")
            return False

        out = self.output_folder.get().strip()
        if not out:
            self.output_folder.set(str(UPLOADS_DIR))

        tess = self.tesseract_path.get().strip()
        if tess and not Path(tess).exists():
            messagebox.showerror(
                "Invalid Tesseract path",
                f"tesseract.exe not found at:\n{tess}",
            )
            return False

        if self.do_split.get():
            try:
                mb = float(self.split_size_mb.get())
                if mb < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid split size",
                    "Split size must be a number ≥ 1 MB.",
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def _start_thread(self):
        if not self._validate():
            return

        self._cancel_flag.clear()
        self.start_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.progress.config(value=0)
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

        threading.Thread(target=self._run_ocr, daemon=True).start()

    # ------------------------------------------------------------------
    # Core OCR pipeline
    # ------------------------------------------------------------------

    def _render_page(self, page, dpi: int) -> Image.Image:
        """Render a fitz page to a PIL RGB image at the requested DPI."""
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.info["dpi"] = (dpi, dpi)
        return img

    def _preprocess(self, img: Image.Image) -> Image.Image:
        """
        Grayscale + autocontrast improves Tesseract accuracy.
        Convert back to RGB only for image_to_pdf_or_hocr compatibility.
        """
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        return img.convert("RGB")

    def _ocr_page(self, img: Image.Image, lang: str, dpi: int, tessdata: str) -> tuple:
        """
        Single-pass OCR using image_to_data.
        Returns (words: list, pdf_bytes: bytes).

        Text is reconstructed from words by the caller via words_to_lines()
        so that line/table structure is preserved.
        """
        # TESSDATA_PREFIX env var (set above) tells Tesseract where tessdata is.
        # --tessdata-dir is intentionally omitted: pytesseract splits config on
        # spaces, which breaks paths that contain spaces (e.g. "OneDrive - Agility").
        ocr_config = f"--dpi {dpi}"

        data = pytesseract.image_to_data(
            img,
            lang=lang,
            config=ocr_config,
            output_type=Output.DICT,
        )

        words = []

        for i in range(len(data["text"])):
            raw = data["text"][i].strip()
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0

            if raw and conf >= 0:
                words.append({
                    "text":       raw,
                    "confidence": conf,
                    "left":       data["left"][i],
                    "top":        data["top"][i],
                    "width":      data["width"][i],
                    "height":     data["height"][i],
                    "block_num":  data["block_num"][i],
                    "line_num":   data["line_num"][i],
                    "word_num":   data["word_num"][i],
                })

        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            img,
            extension="pdf",
            lang=lang,
            config=ocr_config,
        )

        return words, pdf_bytes

    def _run_ocr(self):
        """Background thread: OCR all pages, save outputs, optionally split."""
        doc = None
        try:
            # ── Resolve Tesseract ────────────────────────────────────
            manual = self.tesseract_path.get().strip()
            tess_exe, tessdata = self._resolve_tesseract(manual)
            pytesseract.pytesseract.tesseract_cmd = tess_exe
            os.environ["TESSDATA_PREFIX"] = tessdata

            self.log(f"Tesseract : {tess_exe}")
            self.log(f"tessdata  : {tessdata}")

            # ── Paths ────────────────────────────────────────────────
            input_path   = Path(self.input_pdf.get().strip())
            base_out     = Path(self.output_folder.get().strip())

            # Output structure inside the user-selected folder:
            #   <output_folder>/uploads/   → original PDF + searchable OCR PDF
            #   <output_folder>/ocr_text/  → .txt and .json
            uploads_dir  = base_out / "uploads"
            ocr_text_dir = base_out / "ocr_text"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            ocr_text_dir.mkdir(parents=True, exist_ok=True)

            doc_id       = make_safe_filename(input_path.name)
            source_copy  = uploads_dir / input_path.name
            output_pdf   = uploads_dir / f"{doc_id}_searchable_ocr.pdf"
            output_txt   = ocr_text_dir / f"{doc_id}.txt"
            output_json  = ocr_text_dir / f"{doc_id}.json"

            # Copy original PDF into uploads (skip if same file)
            if input_path.resolve() != source_copy.resolve():
                shutil.copy2(input_path, source_copy)

            # ── Language file check ──────────────────────────────────
            lang = self.language.get()
            dpi  = int(self.dpi.get())

            for code in lang.split("+"):
                td_file = Path(tessdata) / f"{code}.traineddata"
                if not td_file.exists():
                    raise FileNotFoundError(
                        f"Missing Tesseract language file:\n{td_file}"
                    )

            self.log(f"Input        : {input_path}")
            self.log(f"Uploads dir  : {uploads_dir}")
            self.log(f"OCR text dir : {ocr_text_dir}")
            self.log(f"Language    : {lang}   DPI: {dpi}")
            self.log("Starting OCR…")

            # ── Open PDF ─────────────────────────────────────────────
            doc = fitz.open(str(input_path))
            try:
                total_pages = len(doc)
                if total_pages == 0:
                    raise RuntimeError("The selected PDF has no pages.")
            except Exception:
                doc.close()
                raise

            writer     = PdfWriter()
            pages_data = []

            # ── Page loop ────────────────────────────────────────────
            for idx in range(total_pages):
                if self._cancel_flag.is_set():
                    self.log("Cancelled by user.")
                    doc.close()
                    return

                page_no = idx + 1
                self._set_progress(
                    idx,
                    maximum=total_pages,
                    label=f"OCR: page {page_no} / {total_pages}",
                )
                self.log(f"  Page {page_no}/{total_pages}…")

                img = self._render_page(doc[idx], dpi)
                img = self._preprocess(img)

                words, pdf_bytes = self._ocr_page(img, lang, dpi, tessdata)
                lines = words_to_lines(words)
                # Reconstruct text preserving line breaks (better for tables)
                text = "\n".join(line["text"] for line in lines)

                pages_data.append({
                    "page":   page_no,
                    "text":   text,
                    "lines":  lines,
                    "words":  words,
                })

                page_reader = PdfReader(BytesIO(pdf_bytes))
                writer.add_page(page_reader.pages[0])

            # Close fitz before writing to prevent Windows file lock
            doc.close()
            doc = None

            self._set_progress(total_pages, label="Saving OCR PDF…")

            # ── Save full searchable PDF ─────────────────────────────
            with open(output_pdf, "wb") as f:
                writer.write(f)
            self.log(f"Saved OCR PDF  : {output_pdf}")

            # ── Save TXT ─────────────────────────────────────────────
            save_ocr_text(output_txt, pages_data)
            self.log(f"Saved TXT      : {output_txt}")

            # ── Save document-level JSON ──────────────────────────────
            full_text = "\n\n".join(p["text"] for p in pages_data)
            doc_json = {
                "doc_id":        doc_id,
                "file_name":     input_path.name,
                "document_type": "scanned_pdf",
                "source_pdf":    str(source_copy),
                "searchable_pdf": str(output_pdf),
                "text_file":     str(output_txt),
                "full_text":     full_text,
                "pages":         pages_data,
            }
            save_ocr_json(output_json, doc_json)
            self.log(f"Saved JSON     : {output_json}")

            self._last_json_path = output_json
            self.root.after(0, lambda: self.ingest_button.config(state="normal"))

            # ── Optional split ───────────────────────────────────────
            split_paths = []
            if self.do_split.get():
                limit_mb = float(self.split_size_mb.get())
                full_size_mb = output_pdf.stat().st_size / (1024 * 1024)

                if full_size_mb <= limit_mb:
                    self.log(
                        f"Split skipped: full PDF is {full_size_mb:.2f} MB "
                        f"(≤ {limit_mb:.0f} MB limit)."
                    )
                else:
                    self.log(
                        f"Splitting {full_size_mb:.2f} MB PDF "
                        f"into ≤ {limit_mb:.0f} MB parts…"
                    )
                    split_dir = uploads_dir / f"{doc_id}_parts"
                    self._set_progress(0, label="Splitting PDF…")
                    split_paths = split_pdf_by_size(
                        source_pdf=output_pdf,
                        output_dir=split_dir,
                        base_name=doc_id,
                        size_limit_mb=limit_mb,
                        log_fn=self.log,
                    )
                    self.log(
                        f"Split complete: {len(split_paths)} parts → {split_dir}"
                    )

            self._set_progress(total_pages, label="Done.")
            self.log("✓ All done.")

            # ── Completion dialog ────────────────────────────────────
            summary = (
                f"OCR conversion completed.\n\n"
                f"Pages processed : {total_pages}\n"
                f"Searchable PDF  : {output_pdf}\n"
                f"Text file       : {output_txt}\n"
                f"JSON file       : {output_json}\n\n"
                f"Click 'Ingest to ChromaDB' to index this document."
            )
            if split_paths:
                summary += f"\nSplit parts ({len(split_paths)}):\n"
                for p in split_paths:
                    size_mb = p.stat().st_size / (1024 * 1024)
                    summary += f"  {p.name}  ({size_mb:.2f} MB)\n"

            self.root.after(
                0,
                lambda: messagebox.showinfo("Completed", summary),
            )

        except Exception as exc:
            self.log(f"ERROR: {exc}")
            self.root.after(
                0,
                lambda e=str(exc): messagebox.showerror("OCR Failed", e),
            )

        finally:
            # Always close the fitz document to release Windows file lock
            try:
                if doc is not None:
                    doc.close()
            except Exception:
                pass
            self.root.after(0, self._reset_buttons)

    def _ingest_to_chroma(self):
        if not self._last_json_path or not Path(self._last_json_path).exists():
            messagebox.showerror(
                "Ingest Error",
                "No OCR JSON available. Run OCR conversion first."
            )
            return
        self.ingest_button.config(state="disabled")
        self.log("Ingesting into ChromaDB…")
        threading.Thread(target=self._run_ingest, daemon=True).start()

    def _run_ingest(self):
        try:
            from ingest_document import ingest_ocr_json
            ingest_ocr_json(self._last_json_path)
            self.log(f"Ingestion complete: {self._last_json_path}")
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Ingestion Complete",
                    f"Document ingested into ChromaDB.\n\n{self._last_json_path}"
                ),
            )
        except Exception as exc:
            self.log(f"Ingest ERROR: {exc}")
            self.root.after(
                0,
                lambda e=str(exc): messagebox.showerror("Ingest Failed", e),
            )
        finally:
            self.root.after(0, lambda: self.ingest_button.config(state="normal"))

    def _reset_buttons(self):
        self.start_button.config(state="normal")
        self.cancel_button.config(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRDesktopApp(root)
    root.mainloop()
