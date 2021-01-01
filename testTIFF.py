
import numpy
import pathlib
import tifffile

volume = numpy.random.randn(57, 256, 256).astype('uint16')
fileName = 'temp2.tif'
tifffile.imwrite(fileName, volume, imagej=False)
print(fileName)

img =  tifffile.imread(fileName)
print(img.shape)
