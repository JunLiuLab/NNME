#!/usr/bin/env python
# coding: utf-8

# prediction


import numpy as np
from keras.layers import Input, Dense, Lambda, Concatenate, Dropout, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.optimizers import adam
from keras.losses import mse, binary_crossentropy
from sklearn.model_selection import train_test_split #cross_validation
from sklearn import preprocessing
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras import regularizers
from scipy.linalg import sqrtm
from numpy.linalg import inv
import pandas as pd
import sys

dat = pd.read_csv('framingham-1.csv', index_col=0) #,header=None
dat2 = dat.values
dat2[:,0:(dat2.shape[1]-1)] -= dat2[:,0:(dat2.shape[1]-1)].mean(axis=0)

X_train0, X_val0 = train_test_split(dat2[dat2[:,15]==0],test_size=0.20,random_state=int(sys.argv[1]))
X_train1, X_val1 = train_test_split(dat2[dat2[:,15]==1],test_size=0.20,random_state=int(sys.argv[1]))

X_train = np.concatenate((X_train0, X_train1))
np.random.shuffle(X_train)
X_val = np.concatenate((X_val0, X_val1))

#X_train = np.loadtxt("framingham-1_2d_train_" + sys.argv[1] + ".csv",delimiter=',')
#X_val = np.loadtxt("framingham-1_2d_test_" + sys.argv[1] + ".csv", delimiter=',')


prefix="heart"
latent_dim = 2
tot_val = X_train.shape[1] - 1

mc_samples = 50
batch_size = 512  # batch size is 512 for initial fit
epsilon_std = 1.0
#prior_mean = 0
#prior_var = 0.1
Sigma0 = np.array([[0.0126, 0.000673], [0.000673, 0.00846]])
Sigma = sqrtm(inv(Sigma0))
Sigma = K.variable(Sigma)



def build_model(layer, nodes, activ ='relu', input_dim = 1, output_dim = 1, regu = -1, alpha = 0.3):
    model = Sequential()
    if regu > 0:
        model.add(Dense(nodes, input_dim=input_dim, activation=activ, 
                        kernel_regularizer=regularizers.l2(regu), bias_regularizer=regularizers.l2(regu))) #, kernel_initializer='he_normal', bias_initializer='he_normal'))
        for l in np.arange(layer):
            model.add(Dense(nodes, input_dim=nodes, activation=activ, 
                            kernel_regularizer=regularizers.l2(regu), bias_regularizer=regularizers.l2(regu)))#, kernel_initializer='he_normal', bias_initializer='he_normal'))
        model.add(Dense(output_dim, input_dim=nodes, activation = "linear", kernel_regularizer=regularizers.l2(regu)))
    elif activ == 'leakyrelu':
        model.add(Dense(nodes, input_dim=input_dim))
        model.add(LeakyReLU(alpha = alpha))
        for l in np.arange(layer):
            model.add(Dense(nodes, input_dim=nodes))
            model.add(LeakyReLU(alpha = alpha))
        model.add(Dense(output_dim, input_dim=nodes,activation = "linear"))
    else:
        model.add(Dense(nodes, input_dim=input_dim, activation=activ)) 
        for l in np.arange(layer):
            model.add(Dense(nodes, input_dim=nodes, activation=activ))
        model.add(Dense(output_dim, input_dim=nodes, activation = "linear"))
   
    return model


# In[7]:


## prior model ##
def prior_model(layers, nodes, latent_dim2, activ ='relu', regu = -1):
    x1 = Input(shape=(latent_dim2,))
    x2 = Input(shape=(latent_dim2,))
    for l in np.arange(layers):
        px1_shift = build_model(1, nodes , activ='relu', regu = regu, input_dim = latent_dim2, output_dim = latent_dim2) 
        px1_log_scale = build_model(1, nodes , activ='relu', regu = regu, input_dim = latent_dim2, output_dim = latent_dim2)
        if l == 0:
            z_shift = px1_shift(x2)
            z_scale = px1_log_scale(x2)
            z1 = Add()([x1, z_shift])          
        elif l % 2:
            z_shift = px1_shift(z1)
            z_scale = px1_log_scale(z1)
            if l ==1:
                z2 = Add()([x2, z_shift]) 
            else:
                z2 = Add()([z2, z_shift]) 
        else:
            z_shift = px1_shift(z2)
            z_scale = px1_log_scale(z2)
            z1 = Add()([z1, z_shift]) 

        if l==0:
            log_det = z_scale
        else:
            log_det = Add()([log_det, z_scale])

        z_scale = Lambda(lambda t: K.exp(t))(z_scale)
        if l% 2:
            z2 = Multiply()([z2, z_scale])
        else:
            z1 = Multiply()([z1, z_scale])
    nz = Concatenate()([z1, z2]) # normal density


    return Model(inputs = [x1, x2], outputs = [nz, log_det])

def priorLoss(yTrue, yPred):
    prior_loss = K.sum(K.square(nz)/2 - log_det, axis = -1)
    return K.mean(prior_loss)


# In[8]:


class LossLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(LossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, z, fz, w, y, nz, log_det = inputs
       
        w = K.expand_dims(w, axis = 1)
        y = K.expand_dims(y, axis = 1)
        
        fz = K.clip(fz, K.epsilon(), 1.0 - K.epsilon()) # = 1e-7
        reconstruction_loss = K.sum(-y * K.log(fz) - (1.0 - y) * K.log(1.0 - fz), axis = -1)
        #reconstruction_loss = K.sum(-y * K.log(fz + eps_err) - (1.0 - y) * K.log(1.0 - fz + eps_err), axis = -1)
        reconstruction_loss += K.sum(K.square(K.dot(w - z,Sigma)), axis=-1)/2
       
        prior_loss = K.sum(K.square(nz), axis = -1) /2 - K.sum(log_det, axis = -1) 
        
        post_loss = .5 * (K.square(mu - z) /K.exp(log_var) + log_var)
        post_loss = K.sum(post_loss, axis=-1)
      
        return  reconstruction_loss + prior_loss - post_loss


class WeightLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(WeightLayer, self).__init__(*args, **kwargs)

    def call(self, loss):
        
        log_weight = K.stop_gradient(-loss)
        
        log_weight -= K.max(log_weight,axis = 1,keepdims= True)
        
        weight = K.exp(log_weight)
        weight = weight/K.sum(weight,axis = 1,keepdims= True)
        
        return  weight


# In[20]:


#### prediction, test error ####
regu1 = 1e-5 # encoder
regu2 = 1e-4  # decoder
regu = 1e-4
layer1 = 1 # decoder layers
nodes1 = 32 # decoder nodes
layer2 = 1 # encoder layers
nodes2 =  32 # encoder nodes
epochs = 300

results = np.zeros((5, 11)) #nn_loss, nnme_loss, nn_mae, nnme_mae, nnme_weight_mae, nn_fp, nnme_fp, nnme_weight_fp, nn_fn, nnme_fn, nnme_weight_fn
nn_predict_all = np.zeros((X_val.shape[0], 5))
y_pred_all = np.zeros((X_val.shape[0], 5))
yw_pred_all = np.zeros((X_val.shape[0], 5))

for rr in np.arange(5):
    print(rr)
    K.clear_session()
    
    log_sigma2 = K.variable(np.log([0.0126, 0.00846]))
    Sigma = sqrtm(inv(Sigma0))
    Sigma = K.variable(Sigma)

    x_other = Input(shape=(tot_val - latent_dim,))
    x1 = Input(shape=(latent_dim,))

    f2 = Dense(1, kernel_regularizer=regularizers.l2(regu))(x_other)

    model0 = build_model(layer1, nodes1, input_dim = latent_dim, activ='relu', regu = regu) # regu alpha for leaky relu
    f1 = model0(x1)
    output = Add()([f2, f1])
    output = Lambda(lambda t: K.sigmoid(t))(output)

    nn_model = Model(inputs=[x1, x_other], outputs=output)

    ada = adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0002, amsgrad=False)
    nn_model.compile(loss=binary_crossentropy,optimizer=ada, metrics=['mean_absolute_error'])

    history = nn_model.fit([X_train[:,0:latent_dim],X_train[:,latent_dim:tot_val]], X_train[:,tot_val],
                         #validation_data=(X_val[:,0:tot_val], X_val[:, tot_val]),
      batch_size=np.min([512,X_train.shape[0]]),epochs=400,verbose=0, shuffle=True) #, callbacks=[checkpointer]

    nn_predict = nn_model.predict([X_val[:,0:latent_dim],X_val[:,latent_dim:tot_val]] )[:,0]

    org_weight = model0.get_weights()
    results[rr, 0] = history.history['mean_absolute_error'][-1]

    # initialize prior model
    latent_dim2 = int(latent_dim/2)
    pm0 = prior_model(3,16,latent_dim2)
    z1 = Input(shape=(latent_dim2,))
    z2 = Input(shape=(latent_dim2,))
    nz, log_det = pm0([z1,z2])
    pm2 = Model(inputs = [z1, z2], outputs = [nz, log_det])
    pm2.compile(optimizer='adam', loss=priorLoss)

    history = pm2.fit(
      [X_train[:,0:(latent_dim2)],X_train[:,(latent_dim2):(latent_dim)]],
      [X_train[:,0:(latent_dim)],X_train[:,0:(latent_dim)]],
      #validation_data = ([test_dat[:,0:1],test_dat[:,1:2]], [test_dat[:,0:1],test_dat[:,1:2]]),
      shuffle=True,
      epochs=100,
      verbose = 0,
      batch_size=np.min([512,X_train.shape[0]])
    )

    #### vae ####
    model1 = build_model(layer1, nodes1, input_dim = latent_dim, activ='relu', regu = regu2)
    model1.set_weights(org_weight)

    x = Input(shape=(tot_val + 1,))
    x1 = Input(shape=(latent_dim,))
    y1 = Input(shape=(1,))

    model_mu = build_model(layer2, nodes2, activ='relu', regu = regu1, input_dim = tot_val + 1, output_dim = latent_dim)
    z_mu = model_mu(x)
    #z_mu = Add()([z_mu, x1])

    model_var = build_model(layer2, nodes2, activ='relu', regu = regu1, input_dim = tot_val + 1, output_dim = latent_dim)
    z_log_var = model_var(x)

    z_log_var = Lambda(lambda t: t + log_sigma2)(z_log_var)

    encoder = Model([x,x1], [z_mu,z_log_var])

    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

    eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0],mc_samples,latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    x_pred =  model1(z)

    x2 = Input(shape=(tot_val-latent_dim,))
    f2 = Dense(1, kernel_regularizer=regularizers.l2(regu2))(x2)
    x_pred = Add()([x_pred, f2])
    x_pred = Lambda(lambda t: K.sigmoid(t))(x_pred)

    nnme_model = Model(inputs=[x1, x2], outputs=Lambda(lambda t: K.sigmoid(t))(Add()([model1(x1), f2])))

    z_mu = K.expand_dims(z_mu, axis = 1)
    z_log_var = K.expand_dims(z_log_var , axis = 1)


    z_mu0 = K.stop_gradient(z_mu)
    z_log_var0 = K.stop_gradient(z_log_var)
    z0 = K.stop_gradient(z)
    x0_pred =  Add()([model1(z0), f2])
    x0_pred = Lambda(lambda t: K.sigmoid(t))(x0_pred)

    z1 = Lambda(lambda x: x[:,:,0:1])(z)
    z2 = Lambda(lambda x: x[:,:,1:2])(z)

    z10 = K.stop_gradient(z1)
    z20 = K.stop_gradient(z2)

    nz, log_det = pm0([z1,z2])
    nz0, log_det0 = pm0([z10,z20])

    vae_loss = LossLayer(name='LossLayer')([z_mu0, z_log_var0, z, x_pred, x1, y1, nz, log_det])
    weight = WeightLayer(trainable = False,name='WeightLayer')(vae_loss)

    output = Concatenate()([z, x_pred, nz, log_det])
    vae = Model(inputs=[x,x1, x2, y1,eps], outputs=output) # batch * MC * (latent_dim + 1)

    def mae(yTrue, yPred):
        var_y = K.sum(K.abs(yTrue[:,:,latent_dim:(latent_dim+1)]- yPred[:,:,latent_dim:(latent_dim+1)]), axis=-1)
        return K.mean(K.sum(var_y * weight, axis = 1))


    def customLoss(yTrue, yPred):
        loss = K.sum(vae_loss * K.square(weight), axis = 1)
        y = yTrue[:,:,latent_dim:(latent_dim+1)]
        fz = K.clip(x0_pred, K.epsilon(), 1.0 - K.epsilon())
        reconstruction_loss0 = K.sum( -y * K.log(fz) - (1.0 - y) * K.log(1.0 - fz),axis=-1)
        reconstruction_loss0 += K.sum(K.square(nz0), axis = -1) /2 - K.sum(log_det0, axis = -1)

        reconstruction_loss0 = K.sum(reconstruction_loss0 * (weight - K.square(weight)), axis = 1)

        return K.mean(loss + reconstruction_loss0, axis = 0)


    ada = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.001, amsgrad=False) #0.002. 0.004

    vae.compile(optimizer=ada, loss=customLoss, metrics = [mae]) #rmsprop,  weight_entropy
    #vae.summary()

    history = vae.fit([X_train, X_train[:,0:latent_dim], X_train[:,latent_dim:tot_val], X_train[:,tot_val]],
      np.expand_dims(X_train[:,[0,1,tot_val]], axis=1),
      shuffle=True, epochs=epochs, verbose = 0, batch_size=batch_size,
    #   validation_data=(
    #       [X_val, X_val[:,0:latent_dim], X_val_other, X_val[:,tot_val]],
    #       np.expand_dims(X_val[:,[0,1,tot_val]], axis=1)
    #   )
     #,callbacks=[noiseparam ]
    )

    results[rr, 1] = history.history['mae'][-1]

    # prediction
    nsample=200
    nval = X_val.shape[0]
    y_pred = np.zeros((nval, nsample))
    z = np.dot(np.random.normal(0, 1, size=(nval, nsample, latent_dim)), sqrtm(Sigma0)) + np.expand_dims(X_val[:,0:2],axis=1)
    w = np.zeros((nval, nsample))
    # compute the weight for z
    for i in range(nval):
        nz, log_det = pm0.predict([z[i,:,0],z[i,:,1]])
        w1 = np.exp(-np.sum(np.square(nz), axis = -1) /2 + np.sum(log_det, axis = -1))
        w[i,:] = w1/np.sum(w1)

    for i in np.arange(nsample):
        y_pred[:,i] =  nnme_model.predict([z[:,i,:],X_val[:,latent_dim:tot_val]])[:,0]
    

    # get mse
    results[rr, 2] = np.mean(np.abs(nn_predict - X_val[:,15]))
    results[rr, 3] = np.mean(np.abs(np.mean(y_pred, axis = 1) - X_val[:,15]))
    results[rr, 4] = np.mean(np.abs(np.sum(y_pred * w, axis = 1) - X_val[:,15]))
    nn_predict_all[:, rr] = nn_predict
    y_pred_all[:,rr] = np.mean(y_pred, axis = 1)
    yw_pred_all[:,rr] = np.sum(y_pred * w, axis = 1)
    #results[rr, 4] = np.mean(np.abs((nn_predict>0.5) - X_val[:,15]))
    #results[rr, 5] = np.mean(np.abs((np.sum(y_pred * w, axis = 1)>0.5) - X_val[:,15]))
    ind = np.where(X_val[:,15]==0)[0]
    results[rr, 5] = np.sum(nn_predict[ind]>0.5)
    results[rr, 6] = np.sum(np.mean(y_pred[ind,:], axis = 1)>0.5)
    results[rr, 7] = np.sum(np.sum(y_pred[ind,:] * w[ind,:], axis = 1)>0.5)
    
    ind = np.where(X_val[:,15]==1)[0]
    results[rr, 8] = np.sum(nn_predict[ind]<0.5)
    results[rr, 9] = np.sum(np.mean(y_pred[ind,:], axis = 1)<0.5)
    results[rr, 10] = np.sum(np.sum(y_pred[ind,:] * w[ind,:], axis = 1)<0.5)

# select the best one and output
ind = np.argmin(results[:,0], axis=0)
predictions = np.concatenate((nn_predict_all, y_pred_all, yw_pred_all), axis=1)
best_predictions = np.concatenate((nn_predict_all[:,ind], y_pred_all[:,ind], yw_pred_all[:,ind]))
xw = np.concatenate((z[:,:,0],z[:,:,1],w))
                                  
np.savetxt(prefix + "-prior_x-pred-results_" + sys.argv[1] + ".csv", results, delimiter=',')
np.savetxt(prefix + "-prior_x-pred-predictions_" + sys.argv[1] + ".csv", predictions, delimiter=',')
np.savetxt(prefix + "-prior_x-pred-best_" + sys.argv[1] + ".csv", best_predictions, delimiter=',')
np.savetxt(prefix + "-prior_x-pred-xw_" + sys.argv[1] + ".csv", xw, delimiter=',')
