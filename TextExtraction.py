import fitz  

def extract_text_from_pdf(pdf_file):
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            text += page_text
        return text
    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting text: {e}")
