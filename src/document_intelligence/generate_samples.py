"""
generate_samples.py — creates synthetic test documents for the document
intelligence pipeline: one native-text PDF (trade confirmation), one
native-text PDF (ISDA excerpt), and one image-only 'scanned' options chain
that has NO extractable text layer, forcing real OCR to be exercised.

Run once: python3 -m src.document_intelligence.generate_samples
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "documents")


def make_trade_confirmation_pdf():
    path = os.path.join(OUT_DIR, "trade_confirmation_sample.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    lines = [
        "TRADE CONFIRMATION",
        "[Synthetic document for demo/testing purposes only]",
        "",
        "Trade Date: 2026-03-14",
        "Counterparty: Harbor Financial Group",
        "Instrument: Interest Rate Swap",
        "Notional Amount: USD 5,000,000",
        "Settlement Date: 2026-03-16",
        "Fixed Rate: 4.25%",
        "Floating Rate Index: SOFR + 0.35%",
        "Trade Reference: TRD-2026-08841",
    ]
    y = 740
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    return path


def make_isda_pdf():
    path = os.path.join(OUT_DIR, "isda_agreement_excerpt.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    lines = [
        "ISDA MASTER AGREEMENT — SCHEDULE EXCERPT",
        "[Synthetic document for demo/testing purposes only]",
        "",
        "Party A: QuantEdge Capital Partners LLC",
        "Party B: Harbor Financial Group",
        "Agreement Date: 2025-11-01",
        "Governing Law: State of New York",
        "Threshold Amount: USD 10,000,000",
        "Termination Currency: USD",
        "Credit Support Annex: Attached as Exhibit A",
        "Automatic Early Termination: Not Applicable",
    ]
    y = 740
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    return path


def make_options_chain_scanned_image():
    """Image-only document (no text layer) — forces real OCR, simulating
    a scanned/faxed options chain sheet."""
    path = os.path.join(OUT_DIR, "options_chain_scanned.png")
    img = Image.new("RGB", (700, 320), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    lines = [
        "ACME ROBOTICS (ACME) OPTIONS CHAIN — EXP 2026-09-18",
        "[Synthetic scanned-style document for OCR testing]",
        "",
        "STRIKE   TYPE   BID    ASK    VOLUME",
        "140.00   CALL   8.20   8.45   1240",
        "150.00   CALL   3.10   3.35   3560",
        "160.00   CALL   1.05   1.20   980",
        "140.00   PUT    2.15   2.35   740",
        "150.00   PUT    6.40   6.65   1890",
    ]
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += 32
    img.save(path)
    return path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    p1 = make_trade_confirmation_pdf()
    p2 = make_isda_pdf()
    p3 = make_options_chain_scanned_image()
    print(f"Created:\n  {p1}\n  {p2}\n  {p3}")
