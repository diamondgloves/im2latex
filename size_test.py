from scipy.misc import imread
from model.utils.image import greyscale
from PIL import Image
import numpy as np


crop = 'sample_cropped.png'
proc = 'sample_processed.png'
small = 'small.png'
medium = 'medium.png'
big = 'big.png'
mac = 'mac_crop.png'

tmp = 'sample.png'
# img = imread(tmp)
# print(img.size//img[:,:,0].size)
for img_path in [ mac, tmp,crop, small, medium, big]:
	img = imread(img_path)
	# img = np.asarray(Image.open(img_path).convert('L'),dtype=np.uint8)
	# img = img[:,:,np.newaxis]
	oldimg = img
	n_chn = oldimg.size//oldimg[:,:,0].size
	for i in range(n_chn):
		img = oldimg[:,:,i:i+1]
	
		print('name: {}, min:{}, max:{}, size:{}, channels:{}, 255s:{}, 0s:{}'.format(
			img_path, img.min(), img.max(), img.size, n_chn, img[img==255].size, img[img==0].size))
	# input()
	# img = greyscale(img[:,:,:3])
	# print('name: {}, min:{}, max:{}, size:{}, channels:{}, 255s:{}, 0s:{}'.format(
	# 	img_path,img.min(),img.max(),img.size, n_chn,img[img==255].size, img[img==0].size))



# img_path = proc
# oldimg = imread(img_path)
# n_chn = oldimg.size//oldimg[:,:,0].size
# for i in range(n_chn):
# 	img = oldimg[:,:,i:i+1]
# 	print('name: {}, min:{}, max:{}, size:{}, channels:{}, 255s:{}, 0s:{}'.format(
# 		img_path,img.min(),img.max(),img.size, n_chn,img[img==255].size, img[img==0].size))
	# img = greyscale(img)
	# print('name: {}, min:{}, max:{}, size:{}, channels:{}, 255s:{}, 0s:{}'.format(
		# img_path,img.min(),img.max(),img.size, n_chn,img[img==255].size, img[img==0].size))

