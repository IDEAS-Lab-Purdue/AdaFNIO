import cv2
import os
import glob
import time
import numpy as np
#import matplotlib.pyplot as plt
import torch
import gzip
primary_path = '/scratch/gilbreth/hviswan/vimeo_triplet/sequences/'
dir_list = os.listdir(primary_path)
x_data = []
y_data = []
counter = 0
def generateStuff():
	for video_ids in sorted(dir_list):
		path = primary_path+video_ids
		print(path)
		for triplet_no in sorted(os.listdir(path)):
			path_inner = path + '/'+triplet_no + '/*.png'
			#print(path_inner)
			images = glob.glob(path_inner)
			images = sorted(images)
			#print(images)
			length = len(images)
			#print(length)
			for i in range(length-2):
				left = cv2.imread(images[i])
				right = cv2.imread(images[i+2])
				center = cv2.imread(images[i+1])
				#left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
				left = cv2.resize(left, (85,85), interpolation=cv2.INTER_AREA)
				#right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
				right = cv2.resize(right, (85,85), interpolation=cv2.INTER_AREA)
				#center = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
				center = cv2.resize(center, (85,85), interpolation=cv2.INTER_AREA)
				input_im = np.asarray([left, right])
				target_im = np.asarray(center)
				target_im = target_im.reshape(target_im.shape[2], target_im.shape[0], target_im.shape[1])
				x_data.append(input_im)
				y_data.append(target_im)
	"""
	data_size = len(x_data)
	for i in range(0, data_size, 1000):
    	lower_limit = i
    	if(i+1000 >= data_size):
    	    upper_limit = data_size
    	else:
    	    upper_limit = i+1000
    	slice_x = np.asarray(x_data[lower_limit:upper_limit]).astype(np.float32)
    	slice_y = np.asarray(y_data[lower_limit:upper_limit]).astype(np.float32)
    	with open('data_x_slice_'+str(i//1000)+'.npy', 'wb') as f:
    	    np.save(f, slice_x)
    	with open('data_y_slice_'+str(i//1000)+'.npy', 'wb') as f:
    	    np.save(f, slice_y)
    	print(lower_limit, upper_limit)
	
	"""

	"""
	train_size = int(0.8 * data_size)
	train_x = np.asarray(x_data[:train_size]).astype(np.float32)
	f = gzip.GzipFile("train_x.npy.gz", "w")
	np.save(file=f, arr=train_x)
	f.close()
	with open('x_train.npy', 'wb') as f:
    	np.save(f, train_x)
	del train_x	
	"""
	data_size = len(x_data)
	print(data_size)
	train_size = int(0.8 * data_size)
	train_x = torch.from_numpy(np.asarray(x_data[:train_size]).astype(np.float32))
	torch.save(train_x, '/scratch/gilbreth/hviswan/x_train_vimeo_small.pt')
	#del train_x
	print("saved x_train")
	train_y = torch.from_numpy(np.asarray(y_data[:train_size]).astype(np.float32))
	torch.save(train_y, '/scratch/gilbreth/hviswan/y_train_vimeo_small.pt')
	print("saved y_train")
	#with open('y_train.npy', 'wb') as f:
	#    np.save(f, train_y)
	#del train_y
	test_x = torch.from_numpy(np.asarray(x_data[train_size:]).astype(np.float32))
	torch.save(test_x, '/scratch/gilbreth/hviswan/x_test_vimeo_small.pt')
	#del test_x
	
	#with open('x_test.npy', 'wb') as f:
	#    np.save(f, test_x)
	#del test_x
	test_y = torch.from_numpy(np.asarray(y_data[train_size:]).astype(np.float32))
	torch.save(test_y, '/scratch/gilbreth/hviswan/y_test_vimeo_small.pt')
	#with open('y_test.npy', 'wb') as f:
	#    np.save(f, test_y)
	#del test_y
	#print(train_x.shape)
	#print(train_y.shape)
	#print(test_x.shape)
	#print(test_y.shape)
	#torch.save(train_x, 'x_train.pt')
	#torch.save(train_y, 'y_train.pt')
	#torch.save(test_x, 'x_test.pt')
	#torch.save(test_y, 'y_test.pt')
