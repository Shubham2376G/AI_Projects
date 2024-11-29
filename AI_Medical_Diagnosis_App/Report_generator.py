
import datetime

from fillpdf import fillpdfs
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import pdfrw
from pdf2image import convert_from_path
from PIL import Image



def pdf_to_image(input_pdf_path):
    """Convert PDF to a list of images, one per page."""
    # Convert PDF to a list of image objects (one per page)
    images = convert_from_path(input_pdf_path, poppler_path=r'D:\Seed_Hackathon\poppler-24.08.0\Library\bin')
    return images

def images_to_pdf(images, output_pdf_path):
    """Convert a list of images back to a PDF."""
    # Convert images back to PDF (using the first image as the base)
    images[0].save(output_pdf_path, save_all=True, append_images=images[1:], resolution=100.0, quality=95)

def flatten_pdf(input_pdf_path, output_pdf_path):
    """Flatten a PDF by converting it to images and back to PDF."""
    try:
        # Convert the PDF to images
        images = pdf_to_image(input_pdf_path)

        # Convert the images back to a flattened PDF
        images_to_pdf(images, output_pdf_path)

        print(f"Flattened PDF saved to {output_pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def add_image_to_existing_pdf(input_pdf, image_path, output_pdf):
  # Read the existing PDF
  reader = PdfReader(input_pdf)
  writer = PdfWriter()

  # Create a temporary PDF with the image on the second page
  packet = BytesIO()
  # page_width, page_height = letter  # Assuming letter-sized PDF
  # img_width, img_height = 4 * 72, 4 * 72  # Image size (3x3 inches, 1 inch = 72 points)
  # x = (page_width - img_width) / 2
  # y = (page_height - img_height) / 2

  page_width, page_height = 1191.6, 1684.8  # Page size in points

  # Original image dimensions in pixels
  img_width_px, img_height_px = 224, 224

  # Define DPI (dots per inch)
  dpi = 72  # Standard DPI for PDFs

  # Convert image dimensions from pixels to points
  img_width = img_width_px * (172 / dpi)  # Convert width to points
  img_height = img_height_px * (172 / dpi)  # Convert height to points

  # Calculate centered position for the image
  x = (page_width - img_width) / 2  # X-coordinate (horizontal center)
  y = (page_height - img_height) / 2  # Y-coordinate (vertical center)



  # Use ReportLab to create the overlay
  c = canvas.Canvas(packet, pagesize=(page_width, page_height))
  c.drawImage(image_path, x, y, width=img_width, height=img_height)
  c.save()
  packet.seek(0)

  # Merge the overlay into the second page
  overlay_reader = PdfReader(packet)
  for i, page in enumerate(reader.pages):
      if i == 1:  # Add the image only to the second page
          page.merge_page(overlay_reader.pages[0])
      writer.add_page(page)

  # Write the output PDF
  with open(output_pdf, "wb") as output:
      writer.write(output)



def report_gen(name, age, gender, symptoms, ai_report, date, logs):

  form_fields=list(fillpdfs.get_form_fields("Report_template/template_1.pdf").keys())

  data_dict={
      form_fields[0]:name,
      form_fields[1]:age,
      form_fields[2]:gender,
      form_fields[3]:symptoms,
      form_fields[4]:ai_report,
      form_fields[5]:str(datetime.datetime.now().date()),
      form_fields[6]:f"Xray results are {logs}",
  }



  # fillpdfs.write_fillable_pdf("Report_template/template_1.pdf",f"{name}.pdf",data_dict)
  fillpdfs.write_fillable_pdf("Report_template/template_1.pdf",f"Records/{name}.pdf",data_dict)

  input_pdf_path = f"Records/{name}.pdf"

  # Path to save the flattened PDF
  output_pdf_path = f"Records/{name}.pdf"

  # Read the fillable PDF with form data
  pdf = pdfrw.PdfReader(input_pdf_path)

  # Iterate over all the pages
  for page in pdf.pages:
      annotations = page.get('/Annots')
      if annotations:
          for annotation in annotations:
              if annotation.get('/Subtype') == '/Widget':
                  # Flatten the form field by removing its appearance dictionary
                  annotation.update(pdfrw.PdfDict(Subtype=pdfrw.PdfName('Widget'), T=annotation['/T']))

  # Write the flattened PDF
  pdfrw.PdfWriter(output_pdf_path, trailer=pdf).write()


  flatten_pdf(f"Records/{name}.pdf", f"Records/{name}1.pdf")

  add_image_to_existing_pdf(f"Records/{name}1.pdf", "outputs/output.jpg", f"Records/{name}_full.pdf")





