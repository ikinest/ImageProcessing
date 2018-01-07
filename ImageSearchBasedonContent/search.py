#! /usr/bin/env python
#coding=utf-8

# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset
# author:ikinest(shijp)

# import the necessary packages

from searcher import Searcher
import argparse
import numpy as np
import sys
import caffe
from PIL import ImageFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# change the following path to your compiled caffe python path
# 请先编译caffe代码以及python接口部分，将如下的路径改为编译后的python路径
sys.path.append("/home/ikinest/Downloads/image_retrieval/caffe/python")


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())

#提取特征
ImageFile.LOAD_TRUNCATED_IMAGES = True
batchsize = 1
# prototxt神经网络配置文件
net_def_prototxt = "/home/ikinest/Downloads/image_retrieval/image_retrieval_batch.prototxt"
# 训练好的模型
trained_net_caffemodel = "/home/ikinest/Downloads/image_retrieval/Image_Retrieval_128_hash_code.caffemodel"
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
mean_file = "/home/ikinest/Downloads/image_retrieval/image_retrieval_mean.npy"
mean_file = np.load(mean_file).mean(1).mean(1)
transformer.set_mean('data', mean_file) #### subtract mean ####
transformer.set_raw_scale('data', 255) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR 255->
# 设定batchsize
data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

#开始提取特征，将图像和特征保存到TXT文件中
#for i, img_path in enumerate(img_list):        	

#print "extracting feature from image No. %d , %d images in total" %((i+1), len(img_list))
try:		
	x = caffe.io.load_image(args["query"])
	net.blobs['data'].data[...] = transformer.preprocess('data',x)
except ValueError:
	#os.remove(img_path)
	print "提取特征出错"
	#continue
# 卷积神经网络前向运算			
net.forward()
# 获得cf7层4096维的feature
fc7_fea = net.blobs["fc7"].data[:][0]
feature = [float(f) for f in fc7_fea]

# initialize the image descriptor
#feature = Features('vgg16','imagenet')

# load the query image and describe it
#image_path = "/home/april/flower_recognition/dataset/flowers_17/image_0001.jpg"
query = mpimg.imread(args["query"])
plt.title("Query Image")
plt.imshow(query)
plt.show()

#features = [float(f) for f in feature.extractFeatures(args["query"])]

# perform the search
#index = cPickle.loads(open(args["index"]).read())
#searcher = Searcher(index)

searcher = Searcher(args["index"])
results = searcher.search(feature)
print results
print "\n"
print len(results)

# show top #maxres retrieved result one by one
for i in range(len(results)):
    image = mpimg.imread(args["result_path"]+"/"+results[i][1])
    plt.title("search output %d" %(i+1))
    plt.imshow(image)
    plt.show()

# display the query
#cv2.imshow("Query", query)
#cv2.waitKey(0)

# initialize the two montages to display our results --
# we have a total of 25 images in the index, but let's only
# display the top 10 results; 5 images per montage, with
# images that are 400x166 pixels
#montageA = np.zeros((60 * 10, 60, 3), dtype = "uint8")
#montageB = np.zeros((100 * 5, 100, 3), dtype = "uint8")

# single display 
# loop over the results
#for (score, resultID) in results:
#	# load the result image and display it
#	result = cv2.imread(args["result_path"] + "/" + resultID)
#	cv2.imshow("Result", result)
#	cv2.waitKey(0)

# list display
# loop over the top ten results
'''
for j in xrange(0, 10):
	# grab the result (we are using row-major order) and
	# load the result image
	(score, imageName) = results[j]
	path = args["result_path"] + "/%s" % (imageName)
	
	result = cv2.imread(path)
	resize = cv2.resize(result,(60,60),interpolation=cv2.INTER_CUBIC)
	print len(result[0])
	print "\t%d. %s : %.3f" % (j + 1, imageName, score)
 
	# check to see if the first montage should be used
	if j < 10:
		montageA[j * 60:(j + 1) * 60, :] = resize
 
	# otherwise, the second montage should be used
	#else:
	#	montageB[(j - 5) * 100:((j - 5) + 1) * 100, :] = resize
 
	# show the results
cv2.imshow("Results 1-5", montageA)
#cv2.imshow("Results 6-10", montageB)
cv2.waitKey(0)
'''
