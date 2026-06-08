"""
app.py — Scanned PDF → Searchable OCR PDF  (Streamlit Cloud)
Outputs: searchable PDF + optional split parts (by MB or by page count).
Split parts are bundled into a single ZIP download.
No JSON or TXT files are generated.
"""

import zipfile
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageOps
import pytesseract
from pypdf import PdfReader, PdfWriter


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PDF OCR Converter",
    page_icon="📄",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Core: render + OCR one page → PDF bytes
# ---------------------------------------------------------------------------

def ocr_page_to_pdf(fitz_page, dpi: int, lang: str) -> bytes:
    zoom = dpi / 72
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.info["dpi"] = (dpi, dpi)

    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")

    config = f"--dpi {dpi}"
    pdf_bytes: bytes = pytesseract.image_to_pdf_or_hocr(
        img, extension="pdf", lang=lang, config=config
    )
    return pdf_bytes


# ---------------------------------------------------------------------------
# Core: full OCR pipeline → in-memory PDF bytes
# ---------------------------------------------------------------------------

def run_ocr(pdf_bytes_in: bytes, lang: str, dpi: int, progress_bar, status_text) -> bytes:
    doc = fitz.open(stream=pdf_bytes_in, filetype="pdf")
    total = len(doc)
    writer = PdfWriter()

    for idx in range(total):
        status_text.text(f"OCR: page {idx + 1} / {total}")
        progress_bar.progress((idx) / total)

        page_pdf = ocr_page_to_pdf(doc[idx], dpi, lang)
        reader = PdfReader(BytesIO(page_pdf))
        writer.add_page(reader.pages[0])

    doc.close()
    progress_bar.progress(1.0)
    status_text.text("Saving OCR PDF…")

    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_by_pages(pdf_bytes: bytes, pages_per_part: int) -> list[tuple[str, bytes]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    total = len(reader.pages)
    parts = []
    part_num = 1

    for start in range(0, total, pages_per_part):
        end = min(start + pages_per_part, total)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        buf = BytesIO()
        writer.write(buf)
        label = f"part_{part_num:02d}_pages_{start + 1}-{end}.pdf"
        parts.append((label, buf.getvalue()))
        part_num += 1

    return parts


def split_by_mb(pdf_bytes: bytes, limit_mb: float) -> list[tuple[str, bytes]]:
    limit_bytes = int(limit_mb * 1024 * 1024)
    reader = PdfReader(BytesIO(pdf_bytes))
    total = len(reader.pages)

    parts = []
    part_num = 1
    writer = PdfWriter()
    pages_in_part: list[int] = []

    def _flush():
        nonlocal writer, part_num, pages_in_part
        buf = BytesIO()
        writer.write(buf)
        size_mb = len(buf.getvalue()) / (1024 * 1024)
        label = (
            f"part_{part_num:02d}_pages_{pages_in_part[0]}-{pages_in_part[-1]}"
            f"_{size_mb:.1f}MB.pdf"
        )
        parts.append((label, buf.getvalue()))
        part_num += 1
        writer = PdfWriter()
        pages_in_part = []

    def _size(w: PdfWriter) -> int:
        b = BytesIO()
        w.write(b)
        return b.tell()

    for page_idx in range(total):
        page = reader.pages[page_idx]
        page_no = page_idx + 1

        writer.add_page(page)
        pages_in_part.append(page_no)

        if _size(writer) >= limit_bytes:
            if len(pages_in_part) == 1:
                # Single page exceeds limit — flush it alone
                _flush()
            else:
                # Remove last page, flush without it, start new part with it
                tmp = PdfWriter()
                for p in list(writer.pages)[:-1]:
                    tmp.add_page(p)
                last_page = list(writer.pages)[-1]
                last_no = pages_in_part[-1]
                writer = tmp
                pages_in_part = pages_in_part[:-1]
                _flush()
                writer.add_page(last_page)
                pages_in_part = [last_no]

    if pages_in_part:
        _flush()

    return parts


# ---------------------------------------------------------------------------
# ZIP helper
# ---------------------------------------------------------------------------

def make_zip(stem: str, parts: list[tuple[str, bytes]]) -> bytes:
    """Pack all (filename, pdf_bytes) parts into an in-memory ZIP archive."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts:
            zf.writestr(f"{stem}_{name}", data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Scanned PDF → Searchable OCR PDF")
st.caption("Upload a scanned PDF, run OCR, and download the searchable result.")

# ── Upload ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload scanned PDF", type=["pdf"])

if uploaded:
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        lang = st.selectbox(
            "OCR Language",
            options=["eng", "ara", "eng+ara"],
            index=0,
            help="Language(s) for Tesseract. Use 'eng+ara' for mixed documents.",
        )

    with col2:
        dpi = st.selectbox(
            "Render DPI",
            options=[200, 250, 300, 400],
            index=2,
            help="Higher DPI = better accuracy but slower processing.",
        )

    st.divider()

    # ── Split settings ───────────────────────────────────────────────────────
    split_mode = st.radio(
        "Split output PDF?",
        options=["No split", "Split by file size (MB)", "Split by page count"],
        horizontal=True,
    )

    split_mb = None
    split_pages = None

    if split_mode == "Split by file size (MB)":
        split_mb = st.number_input(
            "Max size per part (MB)",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
        )

    elif split_mode == "Split by page count":
        split_pages = st.number_input(
            "Pages per part",
            min_value=1,
            max_value=10000,
            value=50,
            step=1,
        )

    st.divider()

    # ── Run ──────────────────────────────────────────────────────────────────
    if st.button("Start OCR Conversion", type="primary", use_container_width=True):
        input_bytes = uploaded.read()
        stem = Path(uploaded.name).stem

        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Running OCR…"):
            try:
                ocr_bytes = run_ocr(input_bytes, lang, dpi, progress_bar, status_text)
            except Exception as exc:
                st.error(f"OCR failed: {exc}")
                st.stop()

        progress_bar.empty()
        status_text.empty()

        ocr_size_mb = len(ocr_bytes) / (1024 * 1024)
        st.success(
            f"OCR complete — {ocr_size_mb:.2f} MB searchable PDF generated."
        )

        # ── Download: full PDF ────────────────────────────────────────────
        st.download_button(
            label="Download Searchable PDF",
            data=ocr_bytes,
            file_name=f"{stem}_searchable_ocr.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # ── Split + ZIP download ──────────────────────────────────────────
        if split_mode != "No split":
            st.divider()

            with st.spinner("Splitting PDF…"):
                if split_mode == "Split by file size (MB)":
                    full_mb = len(ocr_bytes) / (1024 * 1024)
                    if full_mb <= split_mb:
                        st.info(
                            f"Split skipped — the OCR PDF is only {full_mb:.2f} MB "
                            f"(≤ {split_mb:.0f} MB limit)."
                        )
                    else:
                        parts = split_by_mb(ocr_bytes, split_mb)
                        zip_bytes = make_zip(stem, parts)

                elif split_mode == "Split by page count":
                    parts = split_by_pages(ocr_bytes, split_pages)
                    zip_bytes = make_zip(stem, parts)

            # Only show the ZIP block when splitting actually happened
            if split_mode != "No split" and (
                split_mode == "Split by page count"
                or (split_mode == "Split by file size (MB)" and len(ocr_bytes) / (1024 * 1024) > split_mb)
            ):
                total_zip_mb = len(zip_bytes) / (1024 * 1024)

                # Summary table
                rows = [
                    f"| {i+1} | {name} | {len(data)/(1024*1024):.2f} MB |"
                    for i, (name, data) in enumerate(parts)
                ]
                st.markdown(
                    f"**{len(parts)} part(s) — {total_zip_mb:.2f} MB total:**\n\n"
                    "| # | File | Size |\n|---|---|---|\n" + "\n".join(rows)
                )

                st.download_button(
                    label=f"⬇ Download All {len(parts)} Parts as ZIP",
                    data=zip_bytes,
                    file_name=f"{stem}_split_parts.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                )

else:
    st.info("Upload a scanned PDF above to get started.")
