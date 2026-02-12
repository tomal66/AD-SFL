
import sys
from pypdf import PdfReader

try:
    reader = PdfReader("2025-1698-paper.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Write to a file so I can read it
    with open("paper_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Paper content extracted to paper_content.txt")
except Exception as e:
    print(f"Error extracting text: {e}")
