import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
badlst = []
imgNum = 45
stndrdSize = (330, 200, 3)

remote = np.zeros((198000,imgNum))
for i in range(0, 20):
    if i in badlst:
        continue
    im_rm = imageio.imread('C:/Users/Isaac Pang/Downloads/' +
                           'Remotes_ForHPC/Remote_'+ str(i) + '.jpg')

    
    if im_rm.shape[1] >= stndrdSize[1]:
        left_x = int((im_rm.shape[1]-stndrdSize[1])/2)
        if im_rm.shape[1]%2 != 0:
            im_rm = im_rm[:, left_x:im_rm.shape[1] - left_x - 1]
        else:
            im_rm = im_rm[:, left_x:im_rm.shape[1] - left_x]            
    
    if im_rm.shape[0] >= stndrdSize[0]:
        right_x = int((im_rm.shape[0]-stndrdSize[0])/2)
        if im_rm.shape[0]%2 != 0:
            im_rm = im_rm[right_x:im_rm.shape[0] - right_x-1, :]
        else:
            im_rm = im_rm[right_x:im_rm.shape[0] - right_x, :] 
    
    remote[:, i] = im_rm.ravel()
    
[U, S, V] = lg.svd(remote,full_matrices=False)

sigma = np.diag(S)
idx = np.linspace(0, 45, 45)

plt.plot(idx, np.diag(sigma))

img_num = 0
new_image = np.zeros(stndrdSize)
for i in range(3):
    temp_im = U[int(198000*i/3):int(198000*(i+1)/3), img_num]# - np.min(U[int(198000*i/3):int(198000*(i+1)/3), img_num])
    temp_im /= np.max(U[int(198000*i/3):int(198000*(i+1)/3), img_num])
    new_image[:,:,i] =  temp_im.reshape(stndrdSize[0:2])

#plt.imshow(new_image)


