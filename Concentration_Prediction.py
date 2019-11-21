import pandas as pd
from skimage.io import imread, imshow
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage.feature import match_template
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import math
from sklearn import linear_model
import sklearn.metrics as sklm
import scipy.stats as ss
import re
import sklearn.model_selection as ms
import numpy.random as nr
import scipy.stats as ss

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

nr.seed(7854)

################################################################
pd.set_option('display.max_columns',None)
################################################################

data_dir = 'E:/02 Study/MasterThesis/Test/Masking/ML_Data/'

def getNumbers(str):
    array = re.findall(r'[0-9]+', str)
    return array


img_name = list()

roi1_mean = list()
roi1_median = list()
roi1_std = list()

roi2_mean = list()
roi2_median = list()
roi2_std = list()

concentration = list()


for i, roi in enumerate(os.listdir(data_dir)):


    img = imread(data_dir+roi)

#####################################################################
#    print(i,' ',roi)
#    print()
#    print(np.mean(img))
#    print(np.median(img))
#    print(np.std(img))
#    print()

#####################################################################


########################   Index    ################################
    images = os.listdir(data_dir)
#    for name in images:
    imgname_1 = roi

    try:
        imgname_2 = images[images.index(imgname_1)+1]

        if imgname_1.split()[:-1] == imgname_2.split()[:-1]:
            img_name.append(roi.split('.')[0])
    except:
        pass

##################### ROI 1 #############################
    if roi.split()[-1].split('.')[0] == 'ROI_1':
        roi1_mean.append(np.mean(img))
#        roi1_mean.append(np.log(np.mean(img)))
        roi1_median.append(np.median(img))
        roi1_std.append(np.std(img))
#        roi1_std.append(np.log(np.std(img)))


##################### ROI 2 #############################
    if roi.split()[-1].split('.')[0] == 'ROI_2':
        roi2_mean.append(np.mean(img))
        roi2_median.append(np.median(img))
        roi2_std.append(np.std(img))
##################### Concentra column ##################
    name=roi

    try:
        name2 = images[images.index(name)+1]

        if name.split()[:-1] == name2.split()[:-1]:
            arr = getNumbers(name)
            try:
                concentration.append(np.int(arr[0]))
            except:
                pass

    except:
        pass

zippedList =  list(zip(img_name, roi1_mean, roi1_median, roi1_std,
                       roi2_mean, roi2_median,roi2_std,concentration))

df = pd.DataFrame(zippedList, columns = ['img_name', 'roi1_mean', 'roi1_median','roi1_std',
                                         'roi2_mean', 'roi2_median','roi2_std', 'concentration'])

#zippedList =  list(zip(roi1_mean, roi1_median, roi1_std,
#                       roi2_mean, roi2_median,roi2_std,concentration))
#
#df = pd.DataFrame(zippedList, columns = ['roi1_mean', 'roi1_median','roi1_std',
#                                      'roi2_mean', 'roi2_median','roi2_std', 'concentration'])
#


#scaler = StandardScaler()
#
#scaler.fit(df.drop('concentration',axis=1))
#scaled_features = scaler.transform(df.drop('concentration',axis=1))
#df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])


#X = df[['roi1_mean', 'roi1_median','roi1_std', 'roi2_mean', 'roi2_median','roi2_std']]
X = df[['roi1_mean','roi1_std','roi1_median','roi2_mean','roi2_std']]

y = df['concentration']

#X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['concentration'], test_size=0.3, random_state=101)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#lm = LinearRegression()
lm = LinearRegression(fit_intercept=False)

lm.fit(X_train,y_train)

# print the intercept
print(lm.intercept_)
print(lm.coef_)

#coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
#print(coeff_df)

predictions = lm.predict(X_test)

#print('MAE:', metrics.mean_absolute_error(y_test, predictions))
#print('MSE:', metrics.mean_squared_error(y_test, predictions))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


def print_metrics(y_true, y_predicted, n_parameters):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)

    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))

y_score = lm.predict(X_test)
print_metrics(y_test, y_score, 28)


predictions = predictions.round()

def hist_resids(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test, y_score)
    ## now make the residual plots
    plt.figure()
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')

hist_resids(y_test, y_score)

def resid_qq(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test, y_score)
    ## now make the residual plots
#    ss.probplot(resids.flatten(), plot = plt)
    plt.figure()
    ss.probplot(resids, plot = plt)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')

resid_qq(y_test, y_score)

def resid_plot(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test, y_score)
    ## now make the residual plots
    plt.figure()
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')

resid_plot(y_test, y_score)
#df.to_csv('CropImage_Data.csv')

#def distribution(num_col):
#    plt.figure()
#    sns.distplot(X[num_col], color='g')
#    plt.title(num_col)
#
#for num_col in X.columns:
#    distribution(num_col)



########################## K Fold Cross Validation ##################
print('KFold Cross Validation')
print()

lin_model_cv = cross_val_score(lm,X,y,cv=5)
print(lin_model_cv)

print("RSquare: %0.2f (+/- %0.2f)" % (lin_model_cv.mean(), lin_model_cv.std() * 2))

################ Shuffle Split K-Fold Cross-Validation ####################

lm_shuffle_split = ShuffleSplit(n_splits=5,test_size=0.3)
lm_shuffle_split_cv = cross_val_score(lm, X, y, cv=lm_shuffle_split)
print(lm_shuffle_split_cv)

print("RSquare: %0.2f (+/- %0.2f)" % (lm_shuffle_split_cv.mean(), lm_shuffle_split_cv.std() * 2))

################### finding Best Model and HyperParameters ###########

params_RF = {"max_depth": [3,5,6,7,8,9],
              "max_features":['auto', 'sqrt', 'log2'],
              "min_samples_split": [2, 3,5,7],
              "min_samples_leaf": [1, 3,5,6]}

model_RF_GS = GridSearchCV(RandomForestRegressor(), param_grid=params_RF,cv=5)
model_RF_GS.fit(X,y)

print(model_RF_GS.best_params_)

pred_rf_grid = model_RF_GS.predict(X)

resid_qq(y,pred_rf_grid)
hist_resids(y,pred_rf_grid)
print_metrics(y, pred_rf_grid, 6)

###############  Computing R-Square #########################

print(sklm.r2_score(y,pred_rf_grid))

###################### K-Fold and Holdout Cross-Validation ###############

params_RF = {"max_depth": [3,5,6,7,8,9],
              "max_features":['auto', 'sqrt', 'log2'],
              "min_samples_split": [2, 3,5,7],
              "min_samples_leaf": [1, 3,5,6]}

model_RF_GS = GridSearchCV(RandomForestRegressor(), param_grid=params_RF)
model_RF_GS.fit(X_train,y_train)

print(model_RF_GS.best_params_)

pred_RF_GS = model_RF_GS.predict(X_test)
sklm.r2_score(y_test,pred_RF_GS)

resid_qq(y_test,pred_RF_GS)
hist_resids(y_test,pred_RF_GS)
print_metrics(y_test,pred_RF_GS,6)
