import zipfile
import xml.etree.ElementTree as ET
import sys
import os

try:
    docx_path = r"C:\Users\bhanu\Downloads\FLUXION_Project_Document.docx"
    out_path = r"C:\Users\bhanu\Downloads\Neon-Physics-FLUXION\out.txt"
    if not os.path.exists(docx_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("File does not exist!")
        sys.exit(1)
        
    with zipfile.ZipFile(docx_path) as docx:
        xml_content = docx.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        paragraphs = []
        for p in tree.iterfind('.//w:p', namespaces):
            texts = [node.text for node in p.findall('.//w:t', namespaces) if node.text]
            if texts:
                paragraphs.append(''.join(texts))
                
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(paragraphs))
except Exception as e:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Error: {e}")
