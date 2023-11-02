
from PIL import Image
# Open an image file
image = Image.open("dataset\TrainDataset\Fix\camourflage_00001.png")  # Replace "example.jpg" with the path to your image

img = image.convert("L")
# img = img.convert("I")
img = image.convert("1")

img.show()
