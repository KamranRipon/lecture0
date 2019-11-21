''' ################# Library ############################## '''
from skimage.io import imread, imshow
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import match_template
import os


''' ########################## Iphone Templet ################# '''
iphone = imread('0-I.jpg')

iphone = rgb2gray(iphone)
iphone_temp = iphone[450:2220,1155:1326]

''' ######################### Imager Templet ################## '''
immg = imread('Dig 40 nM.tif')
immg = rgb2gray(immg)

#plt.xticks(range(0,immg.shape[1],20))
imager_temp = immg[600:1550,1542:1639]
#plt.figure()
#imshow(imager_temp)

''' #################### Imager Templet 2 #################### '''
#imager_temp2 = imager_temp[693:717,:]
#plt.figure()
#imshow(imager_temp2)

data_dir = ('../../Dataset Gold-NP-LFA/IMAGES GOLD-DIGOXIGENIN-LFA/Data Serum/14-11-2018 Dig Serum iPhone/DIG Serum I/')

directory = data_dir.split('/')
directory = directory[4:]
save_dir = 'E:/02 Study/MasterThesis/Test/Masking/SavedFiles/'
save_dir = save_dir + '/'.join(directory)

try:
    os.makedirs(save_dir)
except:
    pass

for img in os.listdir(data_dir):
#for img in ['20 nM.JPG']:

    if img.endswith('.JPG') or img.endswith('jpg'):
        try:
            image= imread(data_dir+img)
            image = rgb2gray(image)
#            plt.figure()
#            imshow(image)
#            plt.title(img)

            result = match_template(image, iphone_temp)

            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[::-1]
            crop_img = image[y:y+iphone_temp.shape[0],x:x+iphone_temp.shape[1]]

            plt.figure()
            imshow(crop_img)
            plt.title(img.split('.'))



            crop_img_BM = crop_img.copy()
            crop_img_BM[crop_img_BM>crop_img_BM.mean()] = 0
#            crop_img_BM[crop_img_BM>(crop_img_BM.mean()+(crop_img_BM.min())/15)] = 0

#            plt.figure()
#            imshow(crop_img_BM)
#            plt.title(img)

            n=0
            crop1 = crop_img_BM.copy()
            for k in crop1:
                i = 0
                for y in k:
                    if y == 0 or y == 0.0:
                        i = i +1
                if i > 8:
                    crop1[n, :] = 0
                n = n + 1

#            plt.figure()
#            imshow(crop1)
#            plt.title('Crop1 '+ img.split('.')[0])
#            plt.yticks(range(0,crop1.shape[0],15))

            pts = np.argwhere(crop_img_BM > 0)
            y1,x1 = pts.min(axis=0)
            y2,x2 = pts.max(axis=0)

            crop2 = crop1[y1:y2, x1:x2]

#            plt.figure()
#            plt.imshow(crop2)
#            plt.title('Removing Zero pixel area crop2 '+ img.split('.')[0])
#            plt.yticks(range(0,crop2.shape[0],15))

            if crop2.shape[0] > 1500:
                crop2 = crop2[(round(crop2.shape[0]/1.5)):,:]
            elif crop2.shape[0] > 600:
                crop2 = crop2[(round(crop2.shape[0]/2)):,:]

#            plt.figure()
#            imshow(crop2)
#            plt.title('Removing Zero pixel area crop2_1 '+ img.split('.')[0])
#            plt.yticks(range(0,crop2.shape[0],15))

            zero_list = list()
            for i, j in enumerate(crop2):
                if sum(j) != 0:
                    zero_list.append(i)

            crop3 = crop2[zero_list[0]:zero_list[-1],:]

#            plt.figure()
#            imshow(crop3)
#            plt.title(img+' Zero List')

            ########################
            pts = np.argwhere(crop3 > 0)
            y1,x1 = pts.min(axis=0)
            y2,x2 = pts.max(axis=0)

#            crop4 = crop2[y1:y2, x1:x2]
            crop4 = crop3[y1:y2, x1:x2]

#            plt.figure()
#            imshow(crop4)
#            plt.title(img+' Crop4')

            crop_index=list()
            for l,k in enumerate(crop4):
                if sum(k) != 0:
                    crop_index.append(l)
#
            crop5 = crop4[crop_index[0]:,:]

#            plt.figure()
#            imshow(crop5)
#            plt.title(img+' crop5')
#            plt.yticks(range(0,crop5.shape[0],5))

            list_list = list()
            for z,q in enumerate(crop5):
                if sum(q)==0:
                    break
                else:
                    list_list.append(z)
            if list_list[-1] - list_list[0] < 10:
                crop5 = crop5[list_list[-1]+1:,:]


#            plt.figure()
#            imshow(crop5)
#            plt.title(img+' crop5_1')
#            plt.yticks(range(0,crop5.shape[0],5))

            Npix_idx = list()
            for idx, Npix in enumerate(crop5):
                if sum(Npix) != 0:
                    Npix_idx.append(idx)
                else:
                    pass
            crop6 = crop5[Npix_idx[1]:,:]

#            plt.figure()
#            imshow(crop6)
#            plt.title(img+' crop6')
#            plt.yticks(range(0,crop6.shape[0],10))

            roi1_indx=list()
            for indx, roi1 in enumerate(crop6):
                if sum(roi1) != 0:
                    roi1_indx.append(indx)
                else:
                    pass
#            roi1 = crop6[roi1_indx[2]:roi1_indx[20],:]
            testtest = list()
            for x in roi1_indx:
                try:
                    if x+1 == roi1_indx[roi1_indx.index(x)+1]:
                        testtest.append(x)
                    else:
                        break
                except:
                    pass
            roi1 = crop6[testtest[1]:testtest[-1],:]


            try:
                roi1_idx=list()
                for i, row in enumerate(roi1):
                    try:
                        if sum(row) != 0:
                            roi1_idx.append(i)
                    except:
                        pass
            except:
                pass

#            roi1 = roi1[roi1_idx[0]:roi1_idx[-1],:]
            roi1 = roi1[roi1_idx[0]:roi1_idx[-1],0:166]

#            plt.figure()
#            imshow(roi1)
#            plt.savefig(save_dir+img.split('.')[0] + ' DS_P_Dig_III'+ ' ROI_1.tif')
#            plt.title(img+' ROI 1')
#            plt.yticks(range(0,roi1.shape[0],15))

            crop7_index=list()
            for index in roi1_indx:
                try:
                    if index+1 != roi1_indx[roi1_indx.index(index)+1]:
                        crop7_index.append(roi1_indx[roi1_indx.index(index)+1])
                except:
                    pass

            crop7 = crop6[crop7_index[0]:,:]

#            plt.figure()
#            imshow(crop7)
#            plt.title(img+' crop7')
#            plt.yticks(range(0,crop7.shape[0],15))

            roi2_indx=list()
            for indx2, ro2 in enumerate(crop7):
                if sum(ro2) != 0:
                    roi2_indx.append(indx2)

                    n = n - 1
                else:
                    pass

#            roi2 = crop7[roi2_indx[8]:roi2_indx[2*len(roi1_indx)-8],:]
            crop8 = crop7[roi2_indx[7]:roi2_indx[48],:]
            roi2_index = list()
            for i, indx in enumerate(crop8):
                if sum(indx) != 0:
                    roi2_index.append(i)

            roi2_final_inx = list()
            for x in roi2_index:
                try:
                    if roi2_index[roi2_index.index(x)+1] - x < 55:
                        roi2_final_inx.append(x)
                    else:
                        break
                except:
                    pass
#            roi2 = crop8[roi2_final_inx[1]:roi2_final_inx[-2],:]
            roi2 = crop8[roi2_final_inx[1]:roi2_final_inx[-7],0:166]

#            plt.figure()
#            imshow(roi2)
#            plt.savefig(save_dir+img.split('.')[0]+ ' DS_P_Dig_III' +' ROI_2.tif')
#            plt.title(img+' roi2')
#            plt.yticks(range(0,roi2.shape[0],10))

        except:
            pass

    else:
        pass
