"""
Created on Sat Nov 26 19:48:06 2022

@author: Florian Martin

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import keras
from sklearn.model_selection import KFold
import os 
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.models import Model
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SequentialFeatureSelector
from tensorflow.keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from sklearn.decomposition import PCA
import scipy
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
import matplotlib.image as mpimg
from PIL import Image
import PIL.ImageDraw as ImageDraw  
import imageio
from sklearn.feature_extraction.text import CountVectorizer



def compute_rmse(predict, target):
    if len(target.shape) == 2:
        target = target.squeeze()
    if len(predict.shape) == 2:
        predict = predict.squeeze()
    diff = target - predict
    if len(diff.shape) == 1:
        diff = np.expand_dims(diff, axis=-1)
    rmse = np.sqrt(diff.T@diff / diff.shape[0])
    return float(rmse)

#################################################
############# PREPROCESSING SECTION #############
#################################################

def vectorize_title():
    """
    Create an object CountVectorizer that will transform word into numerical features
    in order to take them into account in ML models.
    """
    
    
    vectorizer = CountVectorizer()
    
    
    X1 = pd.read_csv("X1.csv")
    X1.drop(columns = ["Unnamed: 0"], inplace=True)
    
    X2 = pd.read_csv("X2.csv")
    X2.drop(columns = ["Unnamed: 0"], inplace=True)
    
    
    corpus = []
    
    for i in range(len(X1)):
        item = X1["title"].iloc[i].split(" ")
        for j in item :
            corpus.append(j)
        
    for i in range(len(X2)):
        item = X2["title"].iloc[i].split(" ")
        for j in item :
            corpus.append(j)
    
    
    vectorizer.fit_transform(corpus)
    
    return vectorizer

    
def include_embeddings(X_):
    
    """
    Return a dataframe containing all the samples present in the embeddings
    First they are converted into two dataframe with one columns for each sample per vector
    Then, the two dataframes are concatenated.
    """
    
    X = X_.copy()
    
    X_img = X["img_embeddings"].apply(literal_eval)
    X_text = X["text_embeddings"].apply(literal_eval)
    
    data_img = np.zeros((len(X), len(X_img.iloc[0])))
    data_text = np.zeros((len(X), len(X_text.iloc[0])))

    for i in range(data_img.shape[0]) :
        data_img[i] = np.array(X_img.iloc[i])
        
    for j in range(data_text.shape[0]) :
        data_text[j] = np.array(X_text.iloc[j])
        
    
    return pd.concat([X, pd.DataFrame(data=data_img, columns=[f"img {i}" for i in range(len(X_img.iloc[0]))]), 
                      pd.DataFrame(data=data_text, columns=[f"text {i}" for i in range(len(X_text.iloc[0]))])], axis=1)


def mean_embeddings(X_):
    
    """
    Other method to convert the embeddings. For each vector in embeddings, the mean is computed
    and then replace the list by a single value in the embedding column.
    """
    
    X = X_.copy()
    X["img_embeddings"] = X["img_embeddings"].apply(literal_eval)
    X["text_embeddings"] = X["text_embeddings"].apply(literal_eval)
    
    for i in range(len(X)):
        X["img_embeddings"].iloc[i] = np.mean(X["img_embeddings"].iloc[i])
    
    for i in range(len(X)):
        X["text_embeddings"].iloc[i] = np.mean(X["text_embeddings"].iloc[i])
        
    return X


def remove_outliers(X, Y):
    
    """
    Remove outliers from numerical features (not binary features)
    using the 3-sigma rule such that 99.7% of the data hold in 3-sigma
    
    """
    
    X2 = X.copy()
    y2 = Y.copy()

    X3 = X2.copy()
    y3 = y2.copy()
    
    idx = []

    for i, col in enumerate(X2.columns) :
        
        if (sum(X2[col] == 0.) + sum(X2[col] == 1.)) == len(X2[col]):
            continue

        mean = np.mean(X2[col].values)
        std = np.std(X2[col].values)

        idx += list(X3[X3[col] > mean + 3*std].index.values)
        idx += list(X3[X3[col] < mean - 3*std].index.values)
        
        
    X3.drop(np.unique(idx), inplace=True)
    y3.drop(np.unique(idx), inplace=True)
    
    print("Number of element dropped : ", len(np.unique(idx)))
    
    X3.reset_index(inplace=True)
    y3.reset_index(inplace=True)
    
    X3.drop(columns=["index"], inplace=True)
    y3.drop(columns=["index"], inplace=True)
    
    return X3, y3


def splitting_scaling(X, y):
    
    """
    Method to perform both train_test_split and scaling at once.
    """
    
    X_train, X_test_ = train_test_split(X, shuffle=True, random_state=42, test_size=0.2)
    y_train, y_test  = train_test_split(y, shuffle=True, random_state=42, test_size=0.2)

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(data=scaler.fit_transform(X_train), columns = X_train.columns)
    X_test_scaled  = pd.DataFrame(data=scaler.transform(X_test_), columns = X_test_.columns)
    
    X_train_scaled = X_train_scaled.set_index(X_train.index)
    X_test_scaled = X_test_scaled.set_index(X_test_.index)
    
    return X_train, X_train_scaled, X_test_, X_test_scaled, y_train, y_test, scaler

def power_transform(X):
    
    """
    Method that use PowerTransformer from sklearn. It tries to fit the distribution of a feature
    into a gaussian distribution.
    """
    
    pt = PowerTransformer(standardize=True)
    return pd.DataFrame(data=pt.fit_transform(X.values), columns=X.columns)
    

###############################################
############# PREDICTIONS SECTION #############
###############################################


def perform_grid_search(model_, hyper_params_grid, score_function, X_train, Y_train):
    
    """
    Performs GridSearch to find the best parameters for a given model and a given set of params.
    """
    
    
    scorer = make_scorer(score_function)
    grid = GridSearchCV(estimator=model_, param_grid=hyper_params_grid, scoring = scorer)
    grid.fit(X_train, Y_train)
    
    return grid

def pred_with_decision_tree(X_train, y_train, X_test):
    
    decision_tree = DecisionTreeRegressor()
    params = {'splitter' : ("best", "random"), 'min_samples_split' : [0.5,2,5,10,20] ,'min_impurity_decrease': [0.0,0.5,1.0] ,'criterion' : ("friedman_mse", "squared_error","absolute_error", "poisson"), 'ccp_alpha' : [0.0,0.1,0.5, 0.75],'max_features' : ('auto','sqrt', 'log2'), 'random_state' : [None,0,1,3]}
    grid_search = perform_grid_search(decision_tree, params, compute_rmse, X_train, y_train)
    preds = grid_search.predict(X_test)
    
    return preds, grid_search.best_params_

def pred_with_knn(X_train, y_train, X_test):
    
    knn = KNeighborsRegressor()
    params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12]}
    
    grid_search = perform_grid_search(knn, params, compute_rmse, X_train, y_train)
    preds = grid_search.predict(X_test)
    
    return preds, grid_search.best_params_

def pred_with_SVR(X_train, y_train, X_test):
    
    svr = SVR()
    params = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], "C" : [0.01, 0.1, 10, 1000, 10000]}
    grid_search = perform_grid_search(svr, params, compute_rmse, X_train, y_train)
    preds = grid_search.predict(X_test)
    
    return preds, grid_search.best_params_

def pred_with_random_forest(X_train, y_train, X_test):
    
    RandomForest = RandomForestRegressor()
    params = {'criterion' : ("squared_error","absolute_error", "poisson"), 'min_samples_split' : [2,3,4,5,6], 'max_features' : ("auto", "sqrt", "log2")}
    grid_search = perform_grid_search(RandomForest, params, compute_rmse, X_train, y_train)
    preds = grid_search.predict(X_test)
    
    return preds, grid_search.best_params_



def NN(output_shape = 1, input_shape = None, lr=1e-2) :

    model = Sequential()
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(output_shape, activation="sigmoid"))
    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  metrics = tf.keras.metrics.RootMeanSquaredError(name='RMSE'))
    
    return model

def NN_predict(X_train, y_train, X_test, y_test):
    
    func = lambda x : 10**x
    model = NN()
    if isinstance(model, keras.engine.sequential.Sequential) :
        history = model.fit(X_train, y_train, epochs=10, 
                             batch_size=32, 
                             verbose=2,
                             shuffle=False)
        
        y_preds = y_test.copy()
        for i in range(len(y_preds)) :
            x = tf.convert_to_tensor(X_test.iloc[i].values.reshape(1, X_test.shape[-1]), dtype=tf.int64) 
            y_preds.iloc[i] = model.predict(x)
            
    
        print("MLP preds : ", "{:,}".format(compute_rmse(func(y_preds), func(y_test))))
            
    else:
        raise RuntimeError
        
    

def quick_preds(X_train, y_train, X_test, y_test, scaling = True):
    
    if scaling :
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(data=scaler.fit_transform(X_train), columns = X_train.columns)
        X_test_scaled  = pd.DataFrame(data=scaler.transform(X_test), columns = X_test.columns)
    else :
        X_train_scaled = X_train.copy()
        X_test_scaled  = X_test.copy()


    LR = LinearRegression()
    DTR = DecisionTreeRegressor()
    RFR = RandomForestRegressor()
    KNN = KNeighborsRegressor()
    SVR_m = SVR()
    GBR = GradientBoostingRegressor()
    EN = ElasticNet()
    SGD = SGDRegressor()
    BR = BayesianRidge()
    KR = KernelRidge()


    LR.fit(X_train_scaled, y_train)
    DTR.fit(X_train_scaled, y_train)
    RFR.fit(X_train_scaled, y_train)
    KNN.fit(X_train_scaled, y_train)
    SVR_m.fit(X_train_scaled, y_train)
    GBR.fit(X_train_scaled, y_train)
    EN.fit(X_train_scaled, y_train)
    SGD.fit(X_train_scaled, y_train)
    BR.fit(X_train_scaled, y_train)
    KR.fit(X_train_scaled, y_train)

    func = lambda x : 10**x
    func2 = lambda x : np.exp(x)
    

    y_preds_LR  = LR.predict(X_test_scaled)
    print("LR preds : ", "{:,}".format(compute_rmse(func(y_preds_LR), func(y_test))))
    y_preds_DTR = DTR.predict(X_test_scaled)
    print("DTR preds : ", "{:,}".format(compute_rmse(func(y_preds_DTR), func(y_test))))
    y_preds_RFR = RFR.predict(X_test_scaled)
    print("RFR preds : ", "{:,}".format(compute_rmse(func(y_preds_RFR), func(y_test))))
    y_preds_KNN = KNN.predict(X_test_scaled)
    print("KNN preds : ", "{:,}".format(compute_rmse(func(y_preds_KNN), func(y_test))))
    y_preds_SVR = SVR_m.predict(X_test_scaled)
    print("SVR preds : ", "{:,}".format(compute_rmse(func(y_preds_SVR), func(y_test))))
    y_preds_GBR = GBR.predict(X_test_scaled)
    print("GradientBoosting preds : ", "{:,}".format(compute_rmse(func(y_preds_GBR), func(y_test))))
    y_preds_EN = EN.predict(X_test_scaled)
    print("EleasticNet preds : ", "{:,}".format(compute_rmse(func(y_preds_EN), func(y_test))))
    y_preds_SGD = SGD.predict(X_test_scaled)
    print("SGD preds : ", "{:,}".format(compute_rmse(func(y_preds_SGD), func(y_test))))
    y_preds_BR = BR.predict(X_test_scaled)
    print("BayesianRidge preds : ", "{:,}".format(compute_rmse(func(y_preds_BR), func(y_test))))
    y_preds_KR = KR.predict(X_test_scaled)
    print("KernelRidge preds : ", "{:,}".format(compute_rmse(func(y_preds_KR), func(y_test))))

def Kfold_model(X, y, model, nb_split=5):
    
    """
    X : Training set
    y : Training set target
    
    """
    
    kf = KFold(n_splits=nb_split)
    RMSE = list()
    R2 = list()
    func = lambda x : 10**x
    
    idx = 1
    print("begginning of Kfold")
    for train_index, test_index in kf.split(X):
        
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        
        

        scaler = MinMaxScaler()

        X_train_scaled = pd.DataFrame(data=scaler.fit_transform(X_train_fold), columns = X_train_fold.columns)
        X_test_scaled  = pd.DataFrame(data=scaler.transform(X_test_fold), columns = X_test_fold.columns)    
        
        model.fit(X_train_scaled, y_train_fold)
        y_preds = model.predict(X_test_scaled)

        
        rmse = compute_rmse(func(y_preds), func(y_test_fold))
        r2 = r2_score(func(y_test_fold.to_numpy()), func(y_preds))
        RMSE.append(rmse)
        R2.append(r2)
        
        print(f"The RMSE at iteration {idx} of K-Fold is {rmse}")
        
        idx+=1
    
    print("Mean RMSE : ", np.mean(RMSE), "Std RMSE :", np.std(RMSE))
    print("Mean R2 : ", np.mean(R2), "Std R2 :", np.std(R2))
    
    return y_preds, RMSE, R2

def Kfold_NN(X, y,_epochs=10, nb_split=5):
    
    """
    X : Training set
    y : Training set target
    
    """
    model = NN()
    
    kf = KFold(n_splits=nb_split)
    RMSE = list()
    R2 = list()
    func = lambda x : 10**x
    
    idx = 1
    print("begginning of Kfold")
    for train_index, test_index in kf.split(X):
        
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        history = model.fit(X_train_fold, y_train_fold, epochs=_epochs, 
                             batch_size=32, 
                             verbose=2,
                             shuffle=False)
        
        y_preds = y_test_fold.copy()
        for i in range(len(y_preds)) :
            x = tf.convert_to_tensor(X_test_fold.iloc[i].values.reshape(1, X_test_fold.shape[-1]), dtype=tf.int64) 
            y_preds.iloc[i] = model.predict(x)

        
        rmse = compute_rmse(func(y_preds), func(y_test_fold))
        r2 = r2_score(func(y_test_fold.to_numpy()), func(y_preds))
        RMSE.append(rmse)
        R2.append(r2)
        
        print(f"The RMSE at iteration {idx} of K-Fold is {rmse}")
        
        idx+=1
    
    print("Mean RMSE : ", np.mean(RMSE), "Std RMSE :", np.std(RMSE))
    print("Mean R2 : ", np.mean(R2), "Std RMSE :", np.std(R2))
    
    return y_preds, RMSE, R2, history

def grid_search(X, y) :
    
    kf = KFold(shuffle = True)

    RMSE_decision_tree = list()
    RMSE_knn           = list()
    RMSE_random_forest = list()

    R2_decision_tree   = list()
    R2_knn             = list()
    R2_random_forest   = list()

    old_r2_decision_tree = -np.inf
    old_r2_knn           = -np.inf
    old_r2_random_forest = -np.inf
    
    func = lambda x : 10**x

    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        #Predictions
        y_pred_decision_tree, best_param_decision_tree = pred_with_decision_tree(X_train, y_train, X_test)
        y_pred_knn,           best_param_knn           = pred_with_knn(X_train, y_train, X_test)
        y_pred_random_forest, best_param_random_forest = pred_with_random_forest(X_train, y_train, X_test)
        
        #Compute RMSE
        RMSE_decision_tree.append(compute_rmse(func(y_pred_decision_tree), func(y_test.to_numpy())))
        RMSE_knn.append(          compute_rmse(func(y_pred_knn),           func(y_test.to_numpy())))
        RMSE_random_forest.append(compute_rmse(func(y_pred_random_forest), func(y_test.to_numpy())))

        
        #Compute regression scores
        r2_decision_tree = r2_score(func(y_test.to_numpy()), func(y_pred_decision_tree))
        r2_knn           = r2_score(func(y_test.to_numpy()), func(y_pred_knn))
        r2_random_forest = r2_score(func(y_test.to_numpy()), func(y_pred_random_forest))
        
        #Store the best parameters
        if(r2_decision_tree > old_r2_decision_tree) :
            param_decision_tree = best_param_decision_tree
            old_r2_decision_tree = r2_decision_tree
            
        if(r2_knn > old_r2_knn) :
            param_knn = best_param_knn
            old_r2_knn = r2_knn
            
        if(r2_random_forest > old_r2_random_forest) :
            param_random_forest = best_param_random_forest
            old_r2_random_forest = r2_random_forest
            
        R2_decision_tree.append(r2_decision_tree)
        R2_knn.append(          r2_knn)
        R2_random_forest.append(r2_random_forest)
        
    return RMSE_decision_tree, RMSE_knn, RMSE_random_forest, R2_decision_tree, R2_knn, R2_random_forest, param_decision_tree, param_knn, param_random_forest


########################################################
############# REDUCTION TECHNIQUES SECTION #############
########################################################

class AutoEncoder(Model):
    def __init__(self, input_shape, encode_shape=20):
        super(AutoEncoder, self).__init__()
        self.encode_shape = encode_shape
        self.input_shape_ = input_shape
        
        self.dense1 = Dense(int(self.input_shape_/4), input_shape=(self.input_shape_,), activation='relu')
        self.dense2 = Dense(int(self.input_shape_/8), activation='relu')
        self.dense3 = Dense(self.encode_shape, activation='relu')
        
        self.dense4 = Dense(int(self.input_shape_/8), input_shape=(self.encode_shape,), activation='relu')
        self.dense5 = Dense(int(self.input_shape_/4), activation='relu')
        self.dense6 = Dense(self.input_shape_, activation='sigmoid')
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return x 

    def encode(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        encode = self.dense3(x)
        return encode
    
    def decode(self, x):
        x = self.dense4(x)
        x = self.dense5(x)
        decode = self.dense6(x)
        return decode

def reducing_AE(X, encoder_shape=50):
    
    modelAE = AutoEncoder(X.shape[-1], encode_shape=encoder_shape)
    modelAE.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    history = modelAE.fit(X.values, X.values, epochs=15, shuffle=False, verbose=2)
    
    print("Loss : ", history.history["loss"][-1])
    
    return pd.DataFrame(data=modelAE.encode(X.values).numpy(), columns=[f"{i}" for i in range(modelAE.encode_shape)])    
    
def reducing_PCA(X, nb_col=20):
    
    X_ = X.copy()
    scaling = MinMaxScaler()
    X = pd.DataFrame(data=scaling.fit_transform(X_), columns=X_.columns)
    
    pca = PCA(n_components=nb_col)
    return pd.DataFrame(data=pca.fit_transform(X), columns=[f"principal component {i}" for i in range(pca.n_components)])  
    
#######################################
############# CNN SECTION #############
#######################################

    
    
def dataset_CNN(train = True):
    
    """
    Create a dataset specific for the CNN. It relies only on the embeddings, convert them into 
    an arbitrary chosen shape (64,44) and return the data.
    
    There are two implementation : train and test 
    
    """
    
    if train:
        X1 = pd.read_csv("X1.csv")
        X = X1[["text_embeddings", "img_embeddings"]]
        X_ = include_embeddings(X)
        X_.drop(columns=["text_embeddings", "img_embeddings"], inplace=True)
        Y1 = pd.read_csv("Y1.csv", header = None, names = ["revenue"])
        Y1["revenue"] = np.log(Y1["revenue"])
        
        X_train, X_test = train_test_split(X_, shuffle=True, random_state=42, test_size=0.2)
        y_train, y_test  = train_test_split(Y1, shuffle=True, random_state=42, test_size=0.2)
        
        rows, cols = 64, 44
        X_train = X_train.values.reshape(X_train.shape[0], rows, cols, 1)
        X_test = X_test.values.reshape(X_test.shape[0], rows, cols, 1)
        
        return X_train, X_test, y_train, y_test
    
    else:
        X2 = pd.read_csv("X2.csv")
        X = X2[["text_embeddings", "img_embeddings"]]
        X_ = include_embeddings(X)
        X_.drop(columns=["text_embeddings", "img_embeddings"], inplace=True)
        
        rows, cols = 64, 44
        X_ = X_.values.reshape(X_.shape[0], rows, cols, 1)
        
        return X_
        

def CNN(output_shape = 1, input_shape = (64, 44, 1)) :

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), kernel_initializer='uniform'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))
    
    return model

def r2_keras(y_true, y_pred):
    
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    
    return (1 - SS_res/(SS_tot + K.epsilon()))


def train_CNN(X_train, y_train, epochs=100):
    
    conv = CNN()
    conv.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2_keras])
    
    conv.fit(X_train, y_train)
    
    return conv

def save_CNN(model, name = "model.h5"):
    
    model.save_weights("model.h5")
    

def load_CNN(name = 'best_cnn.h5'):
    
    loaded_cnn = CNN()
    loaded_cnn.load_weights("best_cnn.h5")
    
    return loaded_cnn


############################################
############# PLOTTING SECTION #############
############################################


def plot_revenue(Y1):
    
    
    plt.hist(10**Y1["revenue"], color = "lightcoral")
    plt.title("Histogram of the Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Count")
    #plt.savefig("fig/revenue.svg")
    plt.show()
    
    plt.hist(Y1["revenue"], color = "lightcoral")
    plt.title("Histogram of the Revenue")
    plt.xlabel("Revenue in log-scale")
    plt.ylabel("Count")
    #plt.savefig("fig/revenue_log.svg")
    plt.show()
        
def embeddings_plot(X_):
    
    rows, cols = 64,44
    
    fig, ax = plt.subplots(1,5, figsize=(20,5))
    ax[0].imshow(X_.iloc[0].values.reshape(rows,cols))
    ax[1].imshow(X_.iloc[1].values.reshape(rows,cols))
    ax[2].imshow(X_.iloc[2].values.reshape(rows,cols))
    ax[3].imshow(X_.iloc[3].values.reshape(rows,cols))
    ax[4].imshow(X_.iloc[4].values.reshape(rows,cols))
    plt.savefig("fig/embeddings.svg")
    
    
def plot_anim(df, animate=False, svg=False) :
    
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    
    col = df.columns
    z, x, y = df[col[2]], df[col[1]], df[col[0]] 
    
    ax.scatter(x,y,z, c='b', marker='o') 
    
    ax.set_zlabel(col[2])
    ax.set_xlabel(col[1])
    ax.set_ylabel(col[0])
    
    
    if animate :
        nb = 360
        for i in range(nb) :
            ax.view_init(elev=10., azim=i)
            plt.savefig(f"videos/{i}.png")
            
        tmp_frames = []
        for i in range(nb) :
            img = imageio.imread(f"videos/{i}.png")
            tmp_frames.append(Image.fromarray(img))
    
        imageio.mimwrite(os.path.join('./videos/', 'video.gif'), tmp_frames, fps=10)
    
    if svg:
        ax.view_init(elev=30., azim=30)
        plt.savefig(os.path.join('./figures/', "plot.svg"))
    
    plt.show()  
    
    
def RMSE_plot(example_data1, example_data2) :

    def box_plot(data, edge_color, fill_color):
        bp = ax.boxplot(data, patch_artist=True)
        
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)
    
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)       
            
        return bp
        
    #example_data1 = [RMSE_random_forest, RMSE_decision_tree, RMSE_knn]
    #example_data2 = [RMSE_random_forest_aug, RMSE_decision_tree_aug, RMSE_knn_aug]
    
    fig, ax = plt.subplots()
    bp1 = box_plot(example_data1, 'red', 'lightcoral')
    bp2 = box_plot(example_data2, 'blue', 'lightskyblue')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Original Data', 'Augmented Data'])
    #ax.set_ylim(0, 10)
    plt.title("Boxplot of the RMSE for the different model with both dataset")
    plt.xticks([1, 2, 3], ["Random Forest", "Decision Tree", "KNN"])
    plt.savefig(os.path.join("./figures/", "Boxplot_RMSE.svg"))
    plt.show()
    
def R2_plot(example_data1, example_data2) :

    def box_plot(data, edge_color, fill_color):
        bp = ax.boxplot(data, patch_artist=True)
        
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)
    
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)       
            
        return bp
        
    #example_data1 = [R2_random_forest, R2_decision_tree, R2_knn]
    #example_data2 = [R2_random_forest_aug, R2_decision_tree_aug, R2_knn_aug]
    
    fig, ax = plt.subplots()
    bp1 = box_plot(example_data1, 'red', 'lightcoral')
    bp2 = box_plot(example_data2, 'blue', 'lightskyblue')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Original Data', 'Augmented Data'])
    #ax.set_ylim(0, 10)
    plt.title("Boxplot of the R2-score for the different model with both dataset")
    plt.xticks([1, 2, 3], ["Random Forest", "Decision Tree", "KNN"])
    plt.savefig(os.path.join("./figures/", "Boxplot_R2.svg"))
    plt.show()
    
def jointplots(X, X_pt, col1, col2):


    g0 = sns.jointplot(x=col1, y=col2, data=X, color='#F5B041')
    #plt.title("Before removing outliers", fontweight="bold", fontsize = 30)

    g1= sns.jointplot(x=col1, y=col2, data=X_pt, color='#F5B041')
    #plt.title("After removing outliers", fontweight="bold", fontsize = 30)

    g0.savefig(os.path.join("./jointplot/", f'{col1}_{col2}_before.svg'))
    plt.close(g0.fig)

    g1.savefig(os.path.join("./jointplot/", f'{col1}_{col2}_after.svg'))
    plt.close(g1.fig)
    
def distribution_comparison(X_, X2_) :
    
    
    cols = ["ratings", "n_votes", "production_year", "runtime", "release_year"]
    nb=sum(np.arange(1, len(cols)))
    X = X_[cols]
    X2 = X2_[cols]
    
    
    for i in range(len(cols)):
        for j in range(len(cols)):
            if j > i:
                jointplots(X, X2, cols[i], cols[j])
                
                
def RMSE_R2_plot(RMSE, R2) :

    def box_plot(ax, data, edge_color, fill_color):
        bp = ax.boxplot(data, patch_artist=True)
        
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)
    
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)       
            
        return bp
        
    #example_data1 = [R2_random_forest, R2_decision_tree, R2_knn]
    #example_data2 = [R2_random_forest_aug, R2_decision_tree_aug, R2_knn_aug]
    
    fig, ax = plt.subplots(1,2, figsize=(16,16))
    
    bp1 = box_plot(ax[0], RMSE, 'red', 'lightcoral')
    ax[0].legend([bp1["boxes"][0]], ['RMSE'])
    labels = [item.get_text() for item in ax[0].get_xticklabels()]
    labels = ["Random Forest", "Decision Tree", "KNN"]
    ax[0].set_xticklabels(labels)
    
    
    bp2 = box_plot(ax[1], R2, 'blue', 'lightskyblue')
    ax[1].legend([bp2["boxes"][0]], ['R2'])
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    labels = ["Random Forest", "Decision Tree", "KNN"]
    ax[1].set_xticklabels(labels)
    
    #plt.title("Boxplot of the RMSE and R2-score for the different models.")
    plt.savefig(os.path.join("./figures/", "Boxplot_R2.svg"))
    plt.show()
    
    
def show_outliers(X, Y):  
    
    X2 = X.copy()
    y2 = Y.copy()

    X3 = X2.copy()
    y3 = y2.copy()
    
    cols = ['ratings', 'n_votes', 'production_year', 'runtime', 'release_year']

    for i, col in enumerate(X2.columns) :
        
        if col not in cols : continue

        fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))


        nb_init = len(X2[col])

        mean = np.mean(X2[col].values)
        std = np.std(X2[col].values)

        idx1 = X3[X3[col] > mean + 3*std].index
        idx2 = X3[X3[col] < mean - 3*std].index
        X3.drop(idx1, inplace=True)
        X3.drop(idx2, inplace=True)

        y3.drop(idx1, inplace=True)
        y3.drop(idx2, inplace=True)

        nb_final = len(X3[col])

        print(f"The columns {i} named {col} dropped {nb_init-nb_final} elements !")


        ax1.set_title(f'Before {col}')
        ax1.boxplot(X2[col].values)

        ax2.set_title(f'After {col} with {nb_init-nb_final} dropped')
        ax2.boxplot(X3[col].values)
      
    
    print(f'Dataset was made of {len(X)}. Dataset is now made of {len(X3)} samples') 


def plot_PCA(X):
    
    X_ = X.copy()
    scaling = MinMaxScaler()
    X = pd.DataFrame(data=scaling.fit_transform(X_), columns=X_.columns)

    nb = min(X.shape[0], X.shape[1])
    
    pca = PCA(n_components=nb)
    pca_i = pca.fit_transform(X.values)
    pca_df = pd.DataFrame(data = pca_i, columns = [f'principal component {i}' for i in range(1, pca.n_components+1)])
     
    fig, ax = plt.subplots(1,1, figsize=(4,7))
    
    thresh = 0.8
    
    idx = np.where(np.cumsum(pca.explained_variance_ratio_) > thresh)[0][0]
    
    ax.plot(np.arange(pca.n_components), np.cumsum(pca.explained_variance_ratio_))
    ax.scatter(idx, thresh, color = "orange", marker ='o')
    ax.annotate(str(idx), xy=(idx + 50, thresh-0.025))

    
    ax.set_ylabel("Cumulative sum of the explained variance")
    ax.set_xlabel("Number of principal components")
    ax.grid()
    ax.set_title("Cumulative sum of the explained variance with regards to the number of components")
    
    plt.savefig(os.path.join("./fig/", "explained_var.svg"))
    
    plt.show()
    
    return idx