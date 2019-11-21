''' ################# Library ############################## '''
from skimage.io import imread, imshow
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import match_template
import os
import pandas as pd
import math
import seaborn as sns
import scipy.stats as ss
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


''' ######################### Imager Templet ################## '''
immg = imread('Dig 40 nM.tif')
immg = rgb2gray(immg)

imager_temp = immg[600:1550,1542:1639]

data_dir = ('../../Dataset Gold-NP-LFA/IMAGES GOLD-DIGOXIGENIN-LFA/Data Serum/14-11-2018 Dig Serum BioImager/DIG-Serum I/')

directory = data_dir.split('/')
directory = directory[4:]
save_dir = 'E:/02 Study/MasterThesis/Test/Masking/SavedFiles/'
save_dir = save_dir + '/'.join(directory)

try:
    os.makedirs(save_dir)
except:
    pass


for img in os.listdir(data_dir):
#for img in ['Dig 1 nM.TIF']:

    if img.endswith('.TIF') or img.endswith('tif'):
        try:
            image= imread(data_dir+img)
            image = rgb2gray(image)
#            plt.figure()
#            imshow(image)
#            plt.title(img)

            result = match_template(image, imager_temp)

            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[::-1]
            crop_img = image[y:y+imager_temp.shape[0],x:x+imager_temp.shape[1]]
            plt.figure()
            plt.title(img.split('.')[0])
            imshow(crop_img)

            crop = crop_img.copy()
#            crop[crop > (crop.mean()+crop.min())] = 0
            crop[crop > (crop.mean()+(crop.min())/15)] = 0
#            crop[crop > crop.mean()] = 0

#            plt.figure()
#            imshow(crop)
#            plt.yticks(range(0,crop.shape[0],15))
#            plt.title('Applying black MASK: crop '+ img.split('.')[0])

            n=0
            crop1 = crop.copy()
            for k in crop:
                i = 0
                for y in k:
                    if y == 0 or y == 0.0:
                        i = i +1
                if i > 20:
                    crop1[n, :] = 0
                n = n + 1
#            plt.figure()
#            imshow(crop1)
#            plt.title('Crop1 '+ img.split('.')[0])
#            plt.yticks(range(0,crop1.shape[0],15))

            pts = np.argwhere(crop > 0)
            y1,x1 = pts.min(axis=0)
            y2,x2 = pts.max(axis=0)

            crop2 = crop1[y1:y2, x1:x2]

#            plt.figure()
#            imshow(crop2)
#            plt.title('Removing Zero pixel area crop2 '+ img.split('.')[0])
#            plt.yticks(range(0,crop2.shape[0],15))

            if crop2.shape[0] > 600:
                crop2 = crop2[(round(crop2.shape[0]/2)):,:]

#            plt.figure()
#            imshow(crop2)
#            plt.title('Removing Zero pixel area crop2_1 '+ img.split('.')[0])
#            plt.yticks(range(0,crop2.shape[0],15))


            zero_list = list()
            for i, j in enumerate(crop2):
                if sum(j) != 0:
                    zero_list.append(i)

            ## finding rows of ROI
            index_list = list()
            for num in zero_list:
                try:
                    if num + 1 != zero_list[zero_list.index(num) + 1]:
                        differ = zero_list[zero_list.index(num) + 1] - num
                        if differ > 49:
                            index_list.append(zero_list[zero_list.index(num) + 1])
                        else:
                            index_list.append(num)
                            index_list.append(zero_list[zero_list.index(num) + 1])
        #                    print(num)
        #                    print(zero_list[zero_list.index(num) + 1])
                except:
                    pass

            new_row = list()
            for ele in index_list:
                try:
                    diff = index_list[index_list.index(ele)+1] - ele
                    if diff < 45 and diff > 15:
                        new_row.append(ele)
                        new_row.append(index_list[index_list.index(ele)+1])
        #                print(ele)
        #                print(index_list[index_list.index(ele)+1])
                except:
                    None

            new_row = list(dict.fromkeys(new_row))

            final_index=list()
            final_index.append(new_row[0])
            for  index in new_row:
                try:
                    index_diff = new_row[new_row.index(index)+1] - new_row[0]
                    if index_diff < 100:
                        final_index.append(new_row[new_row.index(index)+1])
                except:
                    None

            new_row = final_index.copy()

            n = new_row[0]
            m = new_row[0]

            try:
                while n > 0 :
                    if sum(crop2[n-1,:]) != 0:
                        n = n-1
        #                print(n)
                    else:
                        break
            except:
                None

            try:
                while m < new_row[1]:
                    if sum(crop2[m+1,:]) != 0:
                        m = m+1
        #                print(m)
                    else:
                        break
            except:
                pass

            try:
                if n != new_row[0]:
                    ROI_1 = crop2[n:new_row[0],:]

#                    plt.figure()
#                    imshow(ROI_1)
#
#                    plt.savefig(save_dir+img.split('.')[0] + ' DS_I_DigS_III' + ' ROI_1.tif')
#                    plt.title('ROI 1 '+ img.split('.')[0])
                else:
                    ROI_1 = crop2[new_row[0]:m,:]
#                    plt.figure()
#                    imshow(ROI_1)
#
#                    plt.savefig(save_dir+img.split('.')[0] + ' DS_I_DigS_III' + ' ROI_1.tif')
#                    plt.title('ROI 1 ' + img.split('.')[0])
            except:
                pass

            p = new_row[-1]
            q = new_row[-1]

            try:
                while p > 0 :
                    if sum(crop2[p-1,:]) != 0:
                        p = p-1
        #                print(p)
                    else:
                        break
            except:
                None

            try:
                while q < crop2.shape[0]:
                    if sum(crop2[q+1,:]) != 0:
                        q = q+1
        #                print(q)
                    else:
                        break
            except:
                pass

            try:
                if p != new_row[-1]:
                    ROI_2 = crop2[p:new_row[-1],:]
#                    plt.figure()
#                    imshow(ROI_2)
#
#                    plt.savefig(save_dir+img.split('.')[0] + ' DS_I_DigS_III' + ' ROI_2.tif')
#                    plt.title('ROI 2 ' + img.split('.')[0])
                else:
                    ROI_2 = crop2[new_row[-1]:q,:]
#                    plt.figure()
#                    imshow(ROI_2)
#
#                    plt.savefig(save_dir+img.split('.')[0] + ' DS_I_DigS_III' + ' ROI_2.tif')
#                    plt.title('ROI 2 '+ img.split('.')[0])

            except:
                pass


        except:
            pass
