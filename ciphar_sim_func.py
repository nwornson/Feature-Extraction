# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:10:05 2019

@author: Nick
"""

def exit_sim(n):
    
    import time
    import pandas as pd
    import numpy as np
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    
    start = time.clock()
    
    # load ciphar10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # training parameters
    epochs = 50
    batch_size = int(.25 * n)
    if batch_size > 128:
        batch_size = 128 # batch size caps at 128
    num_train_samples = n
    num_val_samples = 100
    
    # random sampling and pre-processing
    idx_train = np.random.randint(0,50000,num_train_samples)
    idx_test = np.random.randint(0,10000,num_val_samples)
    
    x_train_samp = x_train[idx_train,:,:,:]
    y_train_sampv = y_train[idx_train,:]
    x_test_samp = x_test[idx_test,:,:,:]
    y_test_sampv = y_test[idx_test,:]
    
    y_train_samp = to_categorical(y_train_sampv)
    y_test_samp = to_categorical(y_test_sampv)
    
    input_shape = x_train_samp.shape[1:]
    
    # scale the data
    x_train_samp = x_train_samp.astype('float32')
    x_test_samp = x_test_samp.astype('float32')
    x_train_samp /= 255
    x_test_samp /= 255
    
    ############## the model  ####################################################
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape,
                     padding = 'same',
                     data_format = 'channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
      
    model.add(Conv2D(64, (3, 3),padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3),padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
            
    model.add(Flatten()) 
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,name = 'dense1')) # <-- feature extraction layer
    model.add(Dense(10)) #10 classes
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(x_train_samp,y_train_samp,batch_size=batch_size,epochs=epochs)
    
    nn_time = time.clock() - start
    nn_acc = model.evaluate(x_test_samp,y_test_samp)[1]
    
    # define the layer for feature extraction
    intermediate_layer = Model(inputs = model.input,outputs = model.get_layer('dense1').output)
    
    # get engineered features for training and validation
    feature_engineered_train = intermediate_layer.predict(x_train_samp)
    feature_engineered_train = pd.DataFrame(feature_engineered_train)
    
    feature_engineered_test = intermediate_layer.predict(x_test_samp)
    feature_engineered_test = pd.DataFrame(feature_engineered_test)
    
    # convert labels back to 1d arrays
    y_train_sampv = y_train_sampv.reshape([num_train_samples])
    y_test_sampv = y_test_sampv.reshape([num_val_samples])
    
    # RandomForest
    rfmod = (RandomForestClassifier(n_estimators = 500,max_features = 100,oob_score = True).fit(feature_engineered_train,y_train_sampv))
    rf_acc = rfmod.score(feature_engineered_test,y_test_sampv)
    
    # Support Vector Machine
    rbf_svc = svm.SVC(kernel='rbf', gamma='scale',decision_function_shape='ovo').fit(feature_engineered_train,y_train_sampv)
    svm_acc = rbf_svc.score(feature_engineered_test,y_test_sampv)
    
    stop = time.clock()
    tot_time = stop - start
    
    return(nn_acc,rf_acc,svm_acc,nn_time,tot_time)