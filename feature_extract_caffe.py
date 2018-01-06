#! /usr/bin/env python
#coding=utf-8

# Authors: ikinest(Shijp)
# use caffe and pre-trained model to compute image feature for image retrieval

# 代码功能：用caffe和预先训练好的卷积神经网络模型，从图片中抽取出图像检索所需的特征，并存储在txt文件中
# 作者：ikinest(Shijp)

import sys
# change the following path to your compiled caffe python path
# 请先编译caffe代码以及python接口部分，将如下的路径改为编译后的python路径
sys.path.append("/home/ikinest/Downloads/image_retrieval/caffe/python")
import caffe
import numpy as np
import os
#from scipy.sparse import csr_matrix
#import cPickle
#import logging
import datetime
from PIL import ImageFile

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print "usage: python compute_fea_for_image_retrieval.py [dataset_path] [net_def_prototxt] [trained_net_caffemodel] [image_mean_file] [out_put file]"
		exit(1)
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	#每300张一个batch进行前向计算，拿到feature
	batchsize = 1
	path  = sys.argv[1]

	logfile = "log_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
	logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	
	# prototxt神经网络配置文件
	net_def_prototxt = sys.argv[2]
	# 训练好的模型
	trained_net_caffemodel = sys.argv[3]
	# 设定gpu模型，若只有cpu，注释掉这一行
	#caffe.set_mode_gpu()
	# 设定cpu模式，若有gpu，请注释掉这一行
	caffe.set_mode_cpu()
	# 通过网络定义文件prototxt和预训练好的模型设定神经网路
	net = caffe.Net(net_def_prototxt, trained_net_caffemodel, caffe.TEST)
	# 图像预处理部分需要transform
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
	# 载入均值文件
	mean_file = sys.argv[4]
	mean_file = np.load(mean_file).mean(1).mean(1)
	transformer.set_mean('data', mean_file) #### subtract mean ####
	transformer.set_raw_scale('data', 255) # pixel value range
	transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR 255->
	# 设定batchsize
	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])
	
	
	img_list = [os.path.join(path,f) for f in os.listdir(path)]
	output = open("/home/ikinest/Downloads/image2012features.txt","w")
	#开始提取特征，将图像和特征保存到TXT文件中
	for i, img_path in enumerate(img_list):        	
        	img_name = os.path.split(img_path)[1]
		print "extracting feature from image No. %d , %d images in total" %((i+1), len(img_list))
		try:		
			x = caffe.io.load_image(img_path)
			net.blobs['data'].data[...] = transformer.preprocess('data',x)
		except ValueError:#only-2D image be surport
			os.remove(img_path)
			continue
		# 卷积神经网络前向运算			
		net.forward()
		# 获得cf7层4096维的feature
		fc7_fea = net.blobs["fc7"].data[:][0]
		features = [str(f) for f in fc7_fea]
		output.write("%s,%s\n" % (img_name, ",".join(features)))	
		#print fc7_fea
		#print "*********************"
		#print len(fc7_fea)
	output.close()
	

