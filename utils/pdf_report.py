from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

def generate_pdf(record_id, cls, conf, risk, plot_path, save_dir):
    pdf_name = f"report_{record_id}.pdf"
    pdf_path = os.path.join(save_dir, pdf_name)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.drawString(50, 800, "Arrhythmia Detection Report")
    c.drawString(50, 760, f"Class: {cls}")
    c.drawString(50, 740, f"Confidence: {conf:.2f}")
    c.drawString(50, 720, f"Risk Level: {risk}")

    if os.path.exists(plot_path):
        c.drawImage(plot_path, 50, 450, width=500, height=200)

    c.save()
    return pdf_name
