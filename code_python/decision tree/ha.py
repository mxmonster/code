# coding=utf-8
from PIL import Image
im = Image.open('tet.jpg')
print im.format, im.size, im.mode

