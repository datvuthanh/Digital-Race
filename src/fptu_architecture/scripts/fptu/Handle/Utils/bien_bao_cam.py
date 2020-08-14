import cv2
import numpy as np
img=cv2.imread('01.png')
img_=cv2.imread('03_progress.png')
check=int(input())

src_pts=np.float32([[35,0],[93,0],[128,64],[0,64]])
dst_pts=np.float32([[30,0],[98,0],[98,128],[30,128]])
def perspective_transform(src_pts,dst_pts):
    '''
    perspective transform
    args:source and destiantion points
    return M and Minv
    '''

    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts,src_pts)

    return {'M':M,'Minv':Minv}
def birdView(img,M):
    '''
    Transform image to birdeye view
    img:binary image
    M:transformation matrix
    return a wraped image
    '''
    #print("wewewewewewewwewewew",img.shape)
    img_sz = (img.shape[1],img.shape[0])
    img_warped = cv2.warpPerspective(img,M,img_sz,flags = cv2.INTER_LINEAR)
    return img_warped
img_=cv2.resize(img_,(128,128))
binary=img_[64:,:]
transform_matrix=perspective_transform(src_pts,dst_pts)
warped_image=birdView(binary,transform_matrix['M'])
warped_image=cv2.resize(warped_image,(512,512))
warped_image=warped_image[:,:,0]

if check ==1:
    max1=0
    max2=0
    cv2.imshow('helo',warped_image)
    for i in range(0,512):
        if warped_image[0][i]!=0:
            max1=i
        if warped_image[511][i]!=0:
            max2=i
    for i in range(0,512):
        for j in range(min(max1,max2),512):
            warped_image[i][j]=0
    print(max1, max2)
if check == 0 :
    min1=0
    min2=0
    cv2.imshow('helo',warped_image)
    print(warped_image[511])
    for i in range(0,512):
        if warped_image[0][i]!=0:
            min1=i
            break
    for i in range(0,512):
        if warped_image[511][i]!=0:
            min2=i
            print("hereee",min2)
            break
    for i in range(0,512):
        for j in range(0,max(min1,min2)):
            warped_image[i][j]=0
    print(min1, min2)

cv2.imshow('hello',warped_image)

print(np.shape(warped_image))
cv2.waitKey(0)



