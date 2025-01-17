# -*- coding: utf-8 -*-
"""vae_mc3.ipynb
0425: change loss function (*latent_dim)
Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_LKG42SnWDfjyqqhn-QKF76y4AyrGBcD
"""

import numpy as np
from keras.layers import Input, Dense, Lambda, Concatenate, Layer, Add, Multiply,RepeatVector
from keras.models import Model, Sequential
from keras.optimizers import adam
from keras.losses import mse, mae
#from sklearn.model_selection import train_test_split #cross_validation
#from sklearn import preprocessing
#import pandas as pd
from keras import backend as K
#import matplotlib.pyplot as plt
#from functools import partial
from keras.callbacks import Callback
from keras import regularizers

## read in parameter
import sys
mc_samples = int(sys.argv[1])
#N = int(sys.argv[2])
#random_state = int(sys.argv[3])
suffix = sys.argv[2] #"1_16_0.2_0.1_1"
prefix = sys.argv[3] # "simulation_GP"
layer = int(sys.argv[4]) #7
nodes = int(sys.argv[5]) #32
layer2 = int(sys.argv[6]) #3
nodes2 = int(sys.argv[7]) #16
repeat = sys.argv[8] # 1-5
noise = float(sys.argv[9])**2
log_sigma2 = np.log(noise)

batch_size = 512
epochs = 400
epsilon_std = 1.0
#noise = 0.1**2 # for x
#prior_mean = 0
#prior_var = 4
regu = regu1 = 1e-5
laplace=False

results = np.zeros((5, 7)) # NN_val_err, NN_test_ise, train_ise, val_ise, train_ll, val_ll, test_ise

file1 = prefix + "_train_" + suffix + "_" + sys.argv[9] + "_" + sys.argv[8] + ".txt"
file2 = prefix + "_test_" + suffix + "_0.1_" + sys.argv[8] + ".txt"
latent_dim = 2

train_dat = np.loadtxt(file1) #simulation_GP_train_1_16_0.2_0.1.txt
test_dat = np.loadtxt(file2) #np.concatenate((train_dat[2500:,0:latent_dim], train_dat[2500:, (2*latent_dim+1): (2*latent_dim+2)]), axis=1)#

idx = (test_dat[:,0] > 0.2) * (test_dat[:,1] < -0.2)!=1
test_dat = test_dat[idx,:]

idx = (test_dat[:,0] < -0.5) * (test_dat[:,1] > 0.5)!=1
test_dat = test_dat[idx,:]


def build_model(layer, nodes, activ ='relu', input_dim = 1, output_dim = 1, regu = -1, alpha = 0.3):
    model = Sequential()
    if regu > 0:
        model.add(Dense(nodes, input_dim=input_dim, activation=activ, 
                        kernel_regularizer=regularizers.l2(regu), bias_regularizer=regularizers.l2(regu))) #, kernel_initializer='he_normal', bias_initializer='he_normal'))
        for l in np.arange(layer):
            model.add(Dense(nodes, input_dim=nodes, activation=activ, 
                            kernel_regularizer=regularizers.l2(regu), bias_regularizer=regularizers.l2(regu)))#, kernel_initializer='he_normal', bias_initializer='he_normal'))
        model.add(Dense(output_dim, input_dim=nodes, kernel_regularizer=regularizers.l2(regu)))
    elif activ == 'leakyrelu':
        model.add(Dense(nodes, input_dim=input_dim))
        model.add(LeakyReLU(alpha = alpha))
        for l in np.arange(layer):
            model.add(Dense(nodes, input_dim=nodes))
            model.add(LeakyReLU(alpha = alpha))
        model.add(Dense(output_dim, input_dim=nodes))
    else:
        model.add(Dense(nodes, input_dim=input_dim, activation=activ)) 
        for l in np.arange(layer):
            model.add(Dense(nodes, input_dim=nodes, activation=activ))
        model.add(Dense(output_dim, input_dim=nodes))
   
    return model


class changeNoise(Callback):
  def __init__(self, noisey):
      super(changeNoise, self).__init__()
      self.noisey = noisey 

  def on_epoch_end(self, epoch, logs={}):
      #print("Setting noisey to =", str(K.get_value(self.noisey)))
      if epoch > 10:   
          K.set_value(self.noisey, logs.get('mise2'))

class LossLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(LossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, z, fz, w, y = inputs #
       
        w = K.expand_dims(w, axis = 1)
        y = K.expand_dims(y, axis = 1)
        
        if laplace:
            reconstruction_loss = K.sum(K.square(y - fz), axis=-1)/noisey/2 + K.sum(K.abs(w - z), axis=-1) /noise
        else:
            reconstruction_loss = K.sum(K.square(y - fz), axis=-1)/noisey/2 + K.sum(K.square(w - z), axis=-1) /noise/2
       
        z1 = K.square((z - prior_mu1)/K.exp(prior_sigma1))/2 + prior_sigma1
        z2 = K.square((z - prior_mu2)/K.exp(prior_sigma2))/2 + prior_sigma2
        prior_loss = prior_pi * K.exp(-K.sum(z1, axis=-1))
        prior_loss += (1-prior_pi) * K.exp(-K.sum(z2, axis=-1))
        prior_loss = -K.log(prior_loss)
        
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

for i, fold_size in enumerate([125, 250, 500, 1000,2000]): # [50, 125, 250, 500] for mixed kernel
  for f in [4]: #np.arange(5):
    X_train = np.concatenate((train_dat[0: f*fold_size,:], train_dat[(f+1)*fold_size:5*fold_size,:]))
    X_val = train_dat[f*fold_size : (f+1)*fold_size,:]
    
    K.clear_session()

    noisey =  K.variable(0.1)
    noiseparam = changeNoise(noisey)
     
    val_err = 1e8
    # initialization by direct regression model (repeat 3 times, get the smallest validation error)
    for kk in np.arange(5):
      model0 = build_model(layer, nodes, input_dim = latent_dim, regu = regu)
      model0.compile(loss=mse,optimizer='adam')  

      history = model0.fit( X_train[:,latent_dim : (2*latent_dim)] , X_train[:,2*latent_dim],
          batch_size=np.min([1024,X_train.shape[0]]),epochs=500,verbose=0, shuffle=True)

      temp = np.mean((model0.predict(X_val[:,latent_dim : (2*latent_dim)]).transpose() - X_val[:,2*latent_dim])**2) #history.history['loss'][-1]
      if temp < val_err:
          val_err = temp
          org_weights = model0.get_weights()
    
    model0.set_weights(org_weights)
    #results[i, 1] = history.history['val_loss'][-1]
    results[i, 0] += val_err #np.mean((model0.predict(X_val[:,latent_dim : (2*latent_dim)]).transpose() - X_val[:,2*latent_dim])**2) #history.history['loss'][-1]

    model00_predict = model0.predict(test_dat[:,0:latent_dim])
    results[i, 1] += np.mean((model00_predict.transpose() - test_dat[:,latent_dim])**2)

    x = Input(shape=(latent_dim + 1,))
    x1 = Input(shape=(latent_dim,))
    y1 = Input(shape=(1,))
    
    prior_mu1 = K.variable(np.array([[[-0.4,-0.2]]]))
    prior_mu2 = K.variable(np.array([[[0.2,0.4]]]))
    prior_sigma1 = K.variable(np.log(np.array([[[0.2,0.3]]])))  # log
    prior_sigma2 = K.variable(np.log(np.array([[[0.3,0.2]]])))
    prior_pi = K.variable(0.7)


    model_mu = build_model(layer2, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = latent_dim) #x1
    z_mu = model_mu(x)
    z_mu = Add()([z_mu, x1])

    model_var = build_model(layer2, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = latent_dim)
    z_log_var = model_var(x)

    encoder = Model([x,x1], [z_mu,z_log_var])

    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

    eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                       shape=(K.shape(x)[0],
                                              mc_samples,
                                              latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])
    z = Lambda(lambda x: K.clip(x,-1,1))(z)

    x_pred =  model0(z)
   
    z_mu = K.expand_dims(z_mu, axis = 1)
    z_log_var = K.expand_dims(z_log_var , axis = 1)

    z_mu0 = K.stop_gradient(z_mu)
    z_log_var0 = K.stop_gradient(z_log_var)
    z0 = K.stop_gradient(z)
    x0_pred =  model0(z0)

    vae_loss = LossLayer(name='LossLayer')([z_mu0, z_log_var0, z, x_pred, x1, y1])  #
    weight = WeightLayer(trainable = False,name='WeightLayer')(vae_loss) 

    output = Concatenate()([z, x_pred])
    vae = Model(inputs=[x,x1, y1, eps], outputs=[output]) # batch * MC * (latent_dim + 1)


      
    def mise2(yTrue, yPred):
      var_y = K.sum(K.square(yTrue[:,:,latent_dim:(latent_dim+1)]- yPred[:,:,latent_dim:(latent_dim+1)]), axis=-1)

      return K.mean(K.sum(var_y * weight, axis = 1))
            

    noisey =  K.variable(0.1)
    noiseparam = changeNoise(noisey)

    def customLoss(yTrue, yPred):
      loss = K.sum(vae_loss * K.square(weight), axis = 1)

      reconstruction_loss0 =  (K.sum(K.square(yTrue[:,:,latent_dim:(latent_dim+1)]- x0_pred), axis=-1)) /noisey/2
      
      z1 = K.square((z0 - prior_mu1)/K.exp(prior_sigma1))/2 + prior_sigma1
      z2 = K.square((z0 - prior_mu2)/K.exp(prior_sigma2))/2 + prior_sigma2
      prior_loss0 = prior_pi * K.exp(-K.sum(z1, axis=-1, keepdims=False))
      prior_loss0 += (1-prior_pi) * K.exp(-K.sum(z2, axis=-1, keepdims=False))
      prior_loss0 = -K.log(prior_loss0)
          
      reconstruction_loss0 += prior_loss0
      
      reconstruction_loss0 = K.sum(reconstruction_loss0 * (weight - K.square(weight)), axis = 1) 
      
      if laplace:      
          return K.mean(loss + reconstruction_loss0, axis = 0) + K.log(noise)*latent_dim + K.log(noisey)/2
      else:
          return K.mean(loss + reconstruction_loss0, axis = 0) + K.log(noise)*latent_dim/2 + K.log(noisey)/2
  

    #ada = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.004, amsgrad=False)
    vae.compile(optimizer='adam', loss=customLoss, metrics = [mise2]) #rmsprop, 
    #vae.summary() 


    history = vae.fit(
      [X_train[:,latent_dim:(latent_dim*2+1)],X_train[:,latent_dim:(latent_dim*2)],X_train[:,latent_dim*2]],
      np.expand_dims(X_train[:,latent_dim:(latent_dim*2+1)], axis=1),
      shuffle=True,
      epochs=epochs,
      verbose = 0, 
      batch_size=np.min([batch_size,X_train.shape[0]]),
      validation_data=(
          [X_val[:,latent_dim:(latent_dim*2+1)],X_val[:,latent_dim:(latent_dim*2)],X_val[:,latent_dim*2]],
          np.expand_dims(X_val[:,latent_dim:(latent_dim*2+1)], axis=1)
      ),
      callbacks=[noiseparam ]
    )


    model0_predict = model0.predict(test_dat[:,0:(latent_dim)])

    results[i, 6] += np.mean((model0_predict.transpose() - test_dat[:,latent_dim])**2)
    results[i, 5] += history.history['val_loss'][-1]
    results[i, 4] += history.history['loss'][-1]
    results[i, 3] += history.history['val_mise2'][-1]
    results[i, 2] += history.history['mise2'][-1]


  if repeat == '1':
    filename = prefix + "_priorx_true_param_" + repeat + "_" + str(mc_samples)+ "_" + str(fold_size) +"_" + str(layer)+"_" +str(nodes) + "_" + str(layer2)+"_" +str(nodes2)+"_"+suffix+"_" + sys.argv[9]

    np.savetxt(filename + "_predict0.txt", np.squeeze(model00_predict))
    np.savetxt(filename + "_predict.txt", np.squeeze(model0_predict))


filename = prefix + "_priorx_true_param_" + repeat + "_" + str(mc_samples) +"_" + str(layer)+"_" +str(nodes) + "_" + str(layer2)+"_" +str(nodes2) + "_" + suffix + "_" + sys.argv[9]
np.savetxt(filename + ".txt", results)


