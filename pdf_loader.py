# pdf_loader.py
import PyPDF2

def load_pdfs(file_paths):
    """
    Load and read multiple PDF files.
    Args:
        file_paths (list): List of PDF file paths.
    Returns:
        str: The extracted text from all PDFs combined.
    """
    all_text = ""
    
    for file_path in file_paths:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    all_text += page.extract_text() or ""
        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {e}")
    
    return all_text

def split_text(text, chunk_size=1000, overlap=100):
    """
    Split text into chunks of a specified size with optional overlap.
    Args:
        text (str): The text to split.
        chunk_size (int): The size of each chunk.
        overlap (int): The overlap between consecutive chunks.
    Returns:
        list: A list of text chunks.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    
    return chunks