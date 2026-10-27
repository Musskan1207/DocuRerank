from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(pdf_file) -> list[str]:
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            extracted_text += f"\n[Page {page_idx + 1}]\n" + text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(extracted_text)