# ELA

from PIL import Image, ImageChops

def ELA_analysis(image_path, quality=90):
    original = Image.open(image_path)
    compressed_path = 'compressed.jpg'
    original.save(compressed_path, 'JPEG', quality=quality)

    compressed = Image.open(compressed_path)
    diff = ImageChops.difference(original, compressed)

    return diff

diff_image = ELA_analysis('input_image.jpg')
diff_image.show()