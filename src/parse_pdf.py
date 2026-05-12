#!/data2/leyizhao/CommTool/.venv/bin/python
import sys
import fitz  # PyMuPDF

def parse_pdf(pdf_path):
    """
    Parses a PDF file and prints its text content using PyMuPDF.
    """
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        # print(f"Successfully opened: {pdf_path}")
        # print(f"Total pages: {len(doc)}")
        # print("-" * 30)
        
        full_text = ""
        # Iterate over each page
        for page_num, page in enumerate(doc):
            # print(f"\n--- Page {page_num + 1} ---")
            
            # Extract text from the page
            text = page.get_text()
            full_text += text + "\n"
            
            # Print the extracted text
            # print(text.strip())
            
        doc.close()
        return full_text
        
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./parse_pdf.py <path_to_pdf>")
        print("Or run with specific python: /data2/leyizhao/CommTool/.venv/bin/python parse_pdf.py <path_to_pdf>")
    else:
        text = parse_pdf(sys.argv[1])
        if text:
            print(text)
