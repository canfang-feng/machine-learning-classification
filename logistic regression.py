#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""In contrast to regression problems, which involve numeric labels (such as an expected delivery time for a meal order), 
classification problems involve labels which can take on only a finite number of different values. The most simple classification
problems are binary classification problems where the label can take on only two different values, such as  𝑦=0  vs.  𝑦=1 ,  𝑦=``cat'' 
vs.  𝑦=``no cat''  or  𝑦=``red wine''  vs.  𝑦=``white wine'' . In classification problems, the label  𝑦  of a data point indicates to 
which class (or category) the data point belongs to. """

"""method: logistic regression and decision trees."""


# In[1]:


################ load the necessary libraries and data #
######################################################################

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plotchecker import ScatterPlotChecker
from unittest.mock  import patch


# Load the dataset and store data and labels in variables
wine = datasets.load_wine()

X_data = wine['data']
wine_class = wine['target']
categories = wine_class.reshape(-1, 1)

print('data shape\t', X_data.shape, '\nlabels shape \t', categories.shape)
print("Number of samples from Class 0:", sum(wine_class == 0))
print("Number of samples from Class 1:", sum(wine_class == 1))
print("Number of samples from Class 2:", sum(wine_class == 2))

# we can use the Python library "pandas" to show us a preview of the features and labels

features = pd.DataFrame(data=wine['data'], columns=wine['feature_names'])
data = features
data['target'] = wine['target']
data['class'] = data['target'].map(lambda ind: wine['target_names'][ind])
print(data.head(5))


# In[2]:


def feature_matrix():
    """
    Generate a feature matrix representing the chemical measurements of wine samples in the dataset.

    :return: array-like, shape=(m, n), feature-matrix with n features for each of m wine samples. """
    wine = datasets.load_wine() # load the dataset into the variable 'wine'
    features = wine['data']     # read out the features of the wine samples and store in variable 'features' 
    n = features.shape[1]       # set n to the number of colums in features 
    m = features.shape[0]       # set m equal to the number of rows in features    
    X = np.zeros((m,n))
    X=wine.data

    return X


# In[3]:


def labels():
    """ 
    :return: array-like, shape=(m, 1), label-vector
    """
    wine = datasets.load_wine() # load the dataset into the variable 'wine'
    cat = wine['target']         # read out the categories (0,1 or 2) of wine samples and store in vector 'cat' 
    m = cat.shape[0]       # set m equal to the number of rows in features  
    y = np.zeros((m, 1));    # initialize label vector with zero entries
    y=np.select([cat==0],[1],default=0).reshape(-1,1)
    
    
    return y


# In[5]:


######## Visualize Data Points.
y = labels() 
X = feature_matrix()
indx_1 = np.where(y == 1)[0] # index of each class 0 wine.
indx_2 = np.where(y == 0)[0] # index of each not class 0 wine
plt.rc('legend', fontsize=20) 
fig, axes = plt.subplots(figsize=(15, 5))
axes.scatter(X[indx_1, 0], X[indx_1, 1], c='g', marker ='x', label='y =1; Class 0 wine')
axes.scatter(X[indx_2, 0], X[indx_2, 1], c='brown', marker ='o', label='y=0; Class 1 or Class 2 wine')
axes.legend(loc='upper left')
axes.set_xlabel('feature x1')
axes.set_ylabel('feature x2')


# In[7]:


###################### Logistic Regression ####################
"""demo- Logistic Loss"""

def sigmoid_func(x):
    f_x = 1/(1+np.exp(-x))
    return f_x

fig, axes = plt.subplots(1, 1, figsize=(15, 5)) #used only for testing purpose

range_x = np.arange(-5 , 5 , 0.01).reshape(-1,1)
print(range_x.shape)
logloss_y1 = np.empty(len(range_x))
logloss_y0 = np.empty(len(range_x))
#squaredloss_y1 = np.empty(len(range_x))
#squaredloss_y0 = np.empty(len(range_x))
plt.rc('legend', fontsize=20) 
plt.rc('axes', labelsize=20) 
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

for i in range(len(range_x)):
    logloss_y1[i] = -np.log(sigmoid_func(range_x[i]))     # logistic loss when true label y=1
    logloss_y0[i] = -np.log(1-sigmoid_func(range_x[i]))   # logistic loss when true label y=0
     
# plot the results, using the plot function in matplotlib.pyplot.

axes.plot(range_x,logloss_y1, linestyle=':', label=r'$y=1$',linewidth=5.0)
axes.plot(range_x,logloss_y0, label=r'$y=0$',linewidth=5.0)

axes.set_xlabel(r'$\mathbf{w}^{T}\mathbf{x}$')
axes.set_ylabel(r'$\mathcal{L}((y,\mathbf{x});\mathbf{w})$')
axes.set_title("logistic loss",fontsize=20)
axes.legend()


# In[8]:


###################### Logistic Regression ####################
"""demo- Logistic vs. Squared Error Loss."""

def sigmoid_func(x):
    f_x = 1/(1+np.exp(-x))
    return f_x

fig, axes = plt.subplots(1, 1, figsize=(15, 5))

range_x = np.arange(-2 , 2 , 0.01).reshape(-1,1)
print(range_x.shape)

logloss_y1 = np.empty(len(range_x))
logloss_y0 = np.empty(len(range_x))
squaredloss_y1 = np.empty(len(range_x))
squaredloss_y0 = np.empty(len(range_x))

plt.rc('legend', fontsize=20) 
plt.rc('axes', labelsize=40) 
plt.rc('xtick', labelsize=30) 
plt.rc('ytick', labelsize=30) 

for i in range(len(range_x)):
    logloss_y1[i] = -np.log(sigmoid_func(range_x[i]))     # logistic loss when true label y=1
    logloss_y0[i] = -np.log(1-sigmoid_func(range_x[i]))   # logistic loss when true label y=0
    
    squaredloss_y1[i]=np.square(1-range_x[i])
    squaredloss_y0[i]=np.square(0-range_x[i])
# plot the results, using the plot function in matplotlib.pyplot.

axes.plot(range_x,logloss_y1, linestyle=':', label=r'logistic loss $y=1$',linewidth=5.0)
axes.plot(range_x,logloss_y0, label=r'logistic loss $y=0$',linewidth=5.0)
axes.plot(range_x,squaredloss_y0/2, label=r'squared error for $y=0$',linewidth=5.0)
axes.plot(range_x,squaredloss_y1/2, label=r'squared error for $y=1$',linewidth=5.0)

axes.set_xlabel(r'$\mathbf{w}^{T}\mathbf{x}$')
axes.set_ylabel(r'$\mathcal{L}((y,\mathbf{x});\mathbf{w})$')
axes.legend()


# In[9]:


###################### Logistic Regression ####################
"""demo-  Logistic Regression."""
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

wine = datasets.load_wine()         # load wine datasets into variable "wine"
X = wine['data']                    # matrix containing the feature vectors of wine samples
cat = wine['target'].reshape(-1, 1) # vector with wine categories (0,1 or 2)

m = cat.shape[0]         # set m equal to the number of rows in features  
y = np.zeros((m, 1));    # initialize label vector with zero entries
    
for i in range(m):
        if (cat[i] == 0):
            y[i,:] = 1 # Class 0
        else:
            y[i,:] = 0 #Not class 0


print(X.shape, y.shape)

# Split X and y to training and test sets with parameters:test_size=0.2, random_state=0
# X_train, X_test, y_train, y_test = ...
X_train=X[:int(0.8*m),:]
X_test=X[int(0.8*m):,:]
y_train=y[:int(0.8*m),:]
y_test=y[int(0.8*m):,:]

# initialize logistic regression
logReg = LogisticRegression(random_state=0,C=1e6,solver='liblinear')

# Train Logistic Regression Classifier
logReg_fit = logReg.fit(X, y)

#Predict the response for test dataset
y_pred =logReg.predict(X).reshape(-1,1)


# In[10]:


###################### Logistic Regression ####################
"""demo- compute accuracy."""

def calculate_accuracy(y, y_hat):
    """
    Calculate accuracy of your prediction
    
    :param y: array-like, shape=(m, 1), correct label vector
    :param y_hat: array-like, shape=(m, 1), label-vector prediction
    
    :return: scalar-like, percentual accuracy of your prediction
    """
    ### STUDENT TASK ###
    # YOUR CODE HERE
    m=y.shape[0]
    n=0
    for i in range(m):
        if y[i]==y_hat[i]:
            n+=1
    accuracy=n/m*100 
    
    return accuracy

wine = datasets.load_wine()         # load wine datasets into variable "wine"
X = wine['data']                    # matrix containing the feature vectors of wine samples
cat = wine['target'].reshape(-1, 1) # vector with wine categories (0,1 or 2)

m = cat.shape[0]         # set m equal to the number of rows in features  
y = np.zeros((m, 1));    # initialize label vector with zero entries
    
for i in range(m):
        if (cat[i] == 0):
            y[i,:] = 1 # Class 0
        else:
            y[i,:] = 0 #Not class 0

logReg = LogisticRegression(random_state=0)
logReg = logReg.fit(X, y)
y_pred = logReg.predict(X).reshape(-1, 1)
            
# Tests
test_acc = calculate_accuracy(y, y_pred)
print ('Accuracy of the result is: %f%%' % test_acc)


# In[11]:


###################### Logistic Regression ####################
"""multiclass classification"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

wine = datasets.load_wine()         # load wine datasets into variable "wine"
X = wine['data']                    # matrix containing the feature vectors of wine samples
y = wine['target'].reshape(-1, 1)   # vector with wine categories (0,1 or 2)
logReg = LogisticRegression(random_state=0,multi_class="ovr") # set multi_class to one versus rest ('ovr')
logReg = logReg.fit(X, y)
y_pred = logReg.predict(X).reshape(-1, 1)


# In[12]:


###################### Logistic Regression ####################
"""Confusion Matrix"""

# This function is used to plot the confusion matrix and normalized confusion matrix
import itertools
from sklearn.metrics import confusion_matrix
def visualize_cm(cm):
    """
    Function visualizes a confusion matrix with and without normalization
    """
    plt.rc('legend', fontsize=10) 
    plt.rc('axes', labelsize=10) 
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10) 


    fig, axes = plt.subplots(1, 2,figsize=(10,5))

    im1 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im1, ax=axes[0])
    classes = ['Class 0','Class 1','Class 2']
    tick_marks = np.arange(len(classes))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(classes,rotation=45)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    axes[0].set_xlabel('predicted label $\hat{y}$')
    axes[0].set_ylabel('true label $y$')
    axes[0].set_title(r'$\bf{Figure\ 6.}$Without normalization')
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im2, ax=axes[1])
    
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(classes,rotation=45)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[1].text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    axes[1].set_xlabel('predicted label $\hat{y}$')
    axes[1].set_ylabel('true label $y$')
    axes[1].set_title(r'$\bf{Figure\ 7.}$Normalized')
    
    axes[0].set_ylim(-0.5,2.5) 
    axes[1].set_ylim(-0.5,2.5)
    
    plt.tight_layout()
    plt.show()


# In[13]:


# Get the confusion matrix from the test set and your predictions
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y,y_pred)
visualize_cm(cm)


# In[ ]:




