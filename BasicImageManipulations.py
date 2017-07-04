'''
Image loading and displaying
'''

import cv2

image = cv2.imread("/media/april/Files/ubuntuProject/.vscode/ImageProcessing/images/jurassic-park-tour-jeep.jpg")#it works
#image = cv2.imread("./image/jurassic-park-tour-jeep.jpg")#The relative path will encounter error(-215)
cv2.imshow("original",image)
cv2.waitKey(0)#keyboard events 0 indicates that any key will be un-pause the excution
print (image.shape)#show  width ,height and RGB of the image

print(image.shape[0])

'''
resizing image
'''
aspectRatio = 100.0 / image.shape[1] # aspect ratio show always keep in mind
dim = (100,int(aspectRatio * image.shape[0])) # fix width 

resized = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)#resize image
cv2.imshow("resized",resized)
cv2.waitKey(0)
print(resized.shape)

'''
rotating image
get the dimentions of image and caulate center of image
'''

(h,w)= image.shape[:2]
center = (w/2,h/2)

M = cv2.getRotationMatrix2D(center,90,1.0)#wrap matrix
rotated = cv2.warpAffine(image,M,(w,h))
cv2.imshow("rotated",rotated)
cv2.waitKey(0)
print(rotated.shape)


'''
Cropping image
'''

cropped = image[100:300,250:600]
cv2.imshow("cropped",cropped)
cv2.waitKey(0)
print(cropped.shape)


'''
Writting image
'''
cv2.imwrite("thumbnail.png",cropped)
