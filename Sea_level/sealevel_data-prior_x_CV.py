#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Input, Dense, Lambda, Concatenate, Dropout, Layer, Add, Multiply,RepeatVector
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import mse, mae
from sklearn.model_selection import train_test_split #cross_validation
from sklearn import preprocessing
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras import regularizers
from scipy.stats import gamma, describe
import sys


# In[2]:


dat2 = np.loadtxt("Data_NC_m_shuffle_" + sys.argv[1] + ".csv", delimiter=',') #Age   RSL Age.Error RSL.Error  V6
dat2[:,0] /= 1000
dat2[:,2] = (dat2[:,2]/2000)**2
dat2[:,3] **= 2
yocc = 2.01


# In[3]:


prefix="sealevel"
latent_dim = 1
layer1 = 4 # decoder layers 4
nodes1 = 32 # decoder nodes 32

layer2 = 2 # encoder layers
nodes2 =  32 # encoder nodes

mc_samples = 50
batch_size = 128  # batch size is 512 for initial fit
epochs = 500
epsilon_std = 1.0
laplace = False

results = np.zeros((3, 6)) 
regu = regu2 = -1 #1e-5  # decoder
regu1 = -1 #1e-4 # encoder
KC = 2


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
    def __init__(self, tau2):
        super(changeNoise, self).__init__()
        self.tau2 = tau2 

    def on_epoch_end(self, epoch, logs={}):
        #print("Setting noisey to =", str(K.get_value(self.noisey)))
        if epoch > 9: #and epoch % 10 == 0:  
            tt = logs.get('mise2')
            if(tt < 1e-6):
                tt = 1e-6
            K.set_value(self.tau2, tt) #


class LossLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(LossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, z, fz, w, y, noisex, noisey, gia, params = inputs
       
        w = K.expand_dims(w, axis = 1)
        y = K.expand_dims(y, axis = 1)
        
        params = K.expand_dims(params, axis = 1)
           
        gia = K.expand_dims(gia, axis = 1)
        
        reconstruction_loss = K.sum(K.square(y - fz + (yocc - z)*gia), axis = -1)/(noisey + tau2)/2 
        reconstruction_loss += K.sum(K.square(w - z), axis = -1)/noisex/2
        
        prior_loss = 0
        for k in np.arange(KC):
            alpha = Lambda(lambda x: x[:,:,(latent_dim *2 *k) : (latent_dim*(2*k+1))])(params)
            beta = Lambda(lambda x: x[:,:,(latent_dim *(2*k+1)) : (latent_dim*2*(k+1))])(params)
            ppi = Lambda(lambda x: x[:,:,2*latent_dim *KC + k])(params)
            z1 = -(alpha-1) * K.log(2.01 - z) + beta * (2.01 - z) - alpha * K.log(beta) + tf.math.lgamma(alpha)
            prior_loss += ppi * K.exp(-K.sum(z1, axis=-1))
        
        prior_loss = -K.log(prior_loss)
        
        post_loss = .5 * (K.square(mu - z) /K.exp(log_var) + log_var)
        post_loss = K.sum(post_loss, axis=-1)
      
        return  reconstruction_loss  - post_loss + prior_loss


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


# cross validation
fold_size = 34
#nn_loss, nn_val_loss, nnme_tau, nnme_val_tau, nn_predict, nnme_predict, nnme_prior_weighted_predict
results = np.zeros((5, 7, 5)) 
gamma_params = np.zeros((5, 6, 5)) 


for fold in np.arange(5):
    print(fold)
    for rr in np.arange(5): 
        print(rr)
        K.clear_session()
        tau2 =  K.variable(0.005)
        noiseparam = changeNoise(tau2) # will change according to "noise", "noisey"

        X_train = np.concatenate((dat2[0: fold*fold_size,:], dat2[(fold+1)*fold_size : dat2.shape[0],:]))
        X_val = dat2[(fold*fold_size) : (fold+1)*fold_size, ]

        model0 = build_model(layer1, nodes1, input_dim = latent_dim, activ='tanh', regu = -1) # 1e-6
        #ada = adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0002, amsgrad=False)
        ada = Adam(lr=0.04, beta_1=0.9, beta_2=0.999, decay=0.0007, epsilon=1e-7, amsgrad=False) #
        model0.compile(loss=mse,optimizer=ada)  

        history = model0.fit( X_train[:,0 : latent_dim] , X_train[:,latent_dim] + (yocc - X_train[:,0]) * X_train[:,4],  #0: latent_dim
                           validation_data=(X_val[:,0 : latent_dim], X_val[:, latent_dim] + (yocc - X_val[:,0]) * X_val[:,4]), 
          batch_size=np.min([512,X_train.shape[0]]),epochs=1000,verbose=0, shuffle=True) #, callbacks=[checkpointer]

        org_weight = model0.get_weights()

        results[rr, 0, fold] = history.history['loss'][-1]
        results[rr, 1, fold] = history.history['val_loss'][-1]

        #### vae ####
        model1 = build_model(layer1, nodes1, activ='tanh', regu = -1) #regu2
        model1.set_weights(org_weight)

        x = Input(shape=(latent_dim + 1,))
        x1 = Input(shape=(latent_dim,))
        y1 = Input(shape=(1,))
        
        placeholder1 = Input(shape=(1,), name="placeholder1") # Gamma param
        placeholder2 = Input(shape=(1,), name="placeholder2") # pi param

        params1 = Dense(latent_dim*KC * 2, name="param1",use_bias=False)(placeholder1) 
        params1 = Lambda(lambda t: K.exp(t))(params1)
        params2 = Dense(KC, name="param2",use_bias=False, activation = "softmax")(placeholder2) 
        params = Concatenate()([params1, params2])

        noisex = Input(shape=(latent_dim,))
        noisey = Input(shape=(1,))
        gia = Input(shape=(1,))

        model_mu = build_model(layer2, nodes2, activ='relu', regu = -1, input_dim = latent_dim + 1, output_dim = latent_dim) #regu1
        z_mu = model_mu(x)
        z_mu = Add()([z_mu, x1])

        model_var = build_model(layer2, nodes2, activ='relu', regu = -1, input_dim = latent_dim + 1, output_dim = latent_dim) #regu1, 
        z_log_var = model_var(x)
        log_noisex = Lambda(lambda t: K.log(t))(noisex)
        z_log_var = Add()([z_log_var, log_noisex])

        encoder = Model([x,x1,noisex], [z_mu,z_log_var])

        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

        eps = Input(tensor=K.random_normal(stddev=epsilon_std,shape=(K.shape(x)[0],mc_samples, latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])
        z = Lambda(lambda x: K.clip(x,-100,2.005))(z)

        x_pred =  model1(z) 

        z_mu = K.expand_dims(z_mu, axis = 1)
        z_log_var = K.expand_dims(z_log_var , axis = 1)

        z_mu0 = K.stop_gradient(z_mu)
        z_log_var0 = K.stop_gradient(z_log_var)
        z0 = K.stop_gradient(z)
        x0_pred =  model1(z0)

        vae_loss = LossLayer(name='LossLayer')([z_mu0, z_log_var0, z, x_pred, x1, y1, noisex, noisey, gia, params]) 
        weight = WeightLayer(trainable = False,name='WeightLayer')(vae_loss) 

        params_out = RepeatVector(mc_samples)(params)
        output = Concatenate()([z, x_pred, params_out])
        vae = Model(inputs=[x, x1, y1,noisex, noisey, gia, placeholder1,placeholder2, eps], outputs=output) # batch * MC * (latent_dim + 1)

        def mise2(yTrue, yPred):
            var_y = yTrue[:,:,latent_dim:(latent_dim+1)]- yPred[:,:,latent_dim:(latent_dim+1)] + (yocc - z0)*K.expand_dims(gia, axis = 1)
            var_y = K.sum(K.square(var_y), axis = -1)
            return K.mean(K.sum(var_y * weight, axis = 1) - noisey)


        def customLoss(yTrue, yPred):
            loss = K.sum(vae_loss * K.square(weight), axis = 1) 
            reconstruction_loss0 = K.square(yTrue[:,:,latent_dim:(latent_dim+1)]- x0_pred + (yocc - z0)*K.expand_dims(gia, axis = 1))
            reconstruction_loss0 = K.sum(reconstruction_loss0, axis = -1)
            reconstruction_loss0 /= (noisey + tau2) * 2
            
            params_t = K.expand_dims(params, axis = 1)
            prior_loss0 = 0
            for k in np.arange(KC):
                alpha = Lambda(lambda x: x[:,:,(latent_dim *2 *k) : (latent_dim*(2*k+1))])(params_t)
                beta = Lambda(lambda x: x[:,:,(latent_dim *(2*k+1)) : (latent_dim*2*(k+1))])(params_t)
                ppi = Lambda(lambda x: x[:,:,2*latent_dim *KC + k])(params_t)
                z1 = -(alpha-1) *K.log(2.01 - z0) + beta * (2.01 - z0) - alpha * K.log(beta) + tf.math.lgamma(alpha)
                prior_loss0 += ppi * K.exp(-K.sum(z1, axis=-1))

            prior_loss0 = -K.log(prior_loss0)
            
            reconstruction_loss0 += prior_loss0
            reconstruction_loss0 = K.sum(reconstruction_loss0 * (weight - K.square(weight)), axis = 1) 

            return K.mean(loss + reconstruction_loss0, axis = 0) + K.mean(K.log(noisex))*latent_dim/2 + K.mean(K.log(noisey + tau2))/2


        #ada = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.002, amsgrad=False)
        ada = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.003, amsgrad=False)
        
        vae.compile(optimizer=ada, loss=customLoss, metrics = [mise2])

        history = vae.fit(
          [X_train[:,0:2],X_train[:,0:1],X_train[:,1], X_train[:,2],X_train[:,3], X_train[:,4], 
           np.ones(X_train.shape[0]),  np.ones(X_train.shape[0])],
          np.expand_dims(X_train[:,0:2], axis=1),
          shuffle=True,
          epochs=1000,
          verbose = 0, 
          batch_size=batch_size,
          validation_data=(
              [X_val[:,0:2],X_val[:,0:1],X_val[:,1], X_val[:,2], X_val[:,3], X_val[:,4],
               np.ones(X_val.shape[0]),  np.ones(X_val.shape[0])],
              np.expand_dims(X_val[:,0:2], axis=1)
          )
          ,callbacks=[noiseparam ]
        )
        
        results[rr, 2, fold] = history.history['mise2'][-1]
        results[rr, 3, fold] = history.history['val_mise2'][-1]
        
        
        # prediction
        nsample=200
        y_pred0 = np.zeros((X_val.shape[0], nsample))
        y_pred = np.zeros((X_val.shape[0], nsample))
        z = np.random.normal(0, 1, size=(X_val.shape[0], nsample)) * np.sqrt(X_val[:,2:3]) + X_val[:,0:1]
        # weighted by the prior
        a = np.exp(vae.layers[12].get_weights())[0,0]
        temp = np.exp(vae.layers[16].get_weights())[0,0]
        pi = temp/np.sum(temp)
        w = pi[0] * gamma.pdf(2.01 - z, a[0], scale=1/a[1]) + pi[1] * gamma.pdf(2.01-z, a[2], scale=1/a[3])
        w /= np.sum(w, axis=1, keepdims=True)
        for i in np.arange(nsample):
            y_pred0[:,i] =  model0.predict(z[:,i])[:,0] - (yocc - z[:,i])*X_val[:,4]
            y_pred[:,i] =  model1.predict(z[:,i])[:,0] - (yocc - z[:,i])*X_val[:,4]

        # get mse
        results[rr, 4, fold] = np.mean((np.mean(y_pred0, axis = 1) - X_val[:,1])**2)
        results[rr, 5, fold] = np.mean((np.mean(y_pred, axis = 1) - X_val[:,1])**2)
        results[rr, 6, fold] = np.mean((np.sum(y_pred * w, axis = 1) - X_val[:,1])**2)
        print(a)
        print(pi)

        gamma_params[rr, 0:4, fold] = a
        gamma_params[rr, 4:6, fold] = pi

np.save("sea_level_cv_results2-tanh-2gamma-prior_"+ sys.argv[1] + ".npy", results)
np.save("sea_level_cv_params-tanh-2gamma-prior_"+ sys.argv[1] + ".npy", gamma_params)









