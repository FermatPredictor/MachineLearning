from PIL import Image
def getPixel(image):
    """ 取得影像中心點的color """
    width, height = image.size
    print(image.size)
    return image.getpixel((width//2, height//2))

if __name__=='__main__':
    img = Image.open('button2.png')
    color = getPixel(img)