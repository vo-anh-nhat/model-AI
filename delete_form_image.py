from rembg import remove
from PIL import Image
input_path = r"C:\MEARCHING LEARNING\test\nhom.jpg"
output_path = r"C:\MEARCHING LEARNING\test\nhom.png"
inp = Image.open(input_path)
output_path = remove(inp)
Image.open("nhom.path")
