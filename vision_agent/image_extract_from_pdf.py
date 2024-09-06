import pymupdf # imports the pymupdf library
import io
import fitz
from PIL import Image
import PIL.Image
import os

def extract_images(pdf: fitz.Document, page: int, imgDir: str):#= '/teamspace/studios/this_studio/table_data1/'):
    imageList = pdf[page].get_images()
    os.makedirs(imgDir, exist_ok=True)
    if imageList:
        print(page)
        for idx, img in enumerate(imageList, start=1):
            data = pdf.extract_image(img[0])
            with PIL.Image.open(io.BytesIO(data.get('image'))) as image:
                image.save(f'{imgDir}/{page}-{idx}.{data.get("ext")}', mode='wb')

# def main(url: str = '/teamspace/studios/this_studio/NeurIPS_Materials/data1/llmware_magicmind/test3/uploads/1-s2.0-S1359645419308444-am (1).pdf'):
#     filename ="/teamspace/studios/this_studio/NeurIPS_Materials/data1/llmware_magicmind/test3/uploads/1-s2.0-S1359645419308444-am (1).pdf"
#     pdf = fitz.open(filename)
#     for page in range(pdf.page_count):
#         extract_images(pdf, page)

# if __name__ == '__main__':
#     main()