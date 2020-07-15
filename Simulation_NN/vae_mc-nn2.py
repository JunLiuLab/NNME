# -*- coding: utf-8 -*-
"""vae_nn2-test.ipynb
for simu_nn
sequential propose x
"""

#@title
import numpy as np
from keras.layers import Input, Dense, Lambda, Concatenate, Dropout, Layer, Add, Multiply, RepeatVector
from keras.models import Model, Sequential
from keras.optimizers import adam
from keras.losses import mse, mae
from sklearn.model_selection import train_test_split #cross_validation
from sklearn import preprocessing
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras import regularizers

import sys
prefix = sys.argv[1] 
regu = 1e-4 #float(sys.argv[2]) #for NN
regu2 = 1e-4 #float(sys.argv[3]) #for decoder weights
regu1 = -1 #for encoder weights

repeat = sys.argv[4]
n = int(sys.argv[5]) #1250

#regu = 0.0005
layer1 = 5 # decoder layers
nodes1 = 32 # decoder nodes

layer2 = 1 # encoder layers
nodes2 =  32 # encoder nodes

mc_samples = 200
batch_size = 512  # batch size is 512 for initial fit
epochs = 100
epsilon_std = 1.0
##noise = 0.2/np.sqrt(2) # for x, 0.1**2 #
prior_mean = 0
prior_var = 0.5
laplace = False
noisex = float(sys.argv[2])
#beta = float(sys.argv[3])
sy = float(sys.argv[3])

X_train = np.loadtxt("simu_nn2/nn_5_32_train_0.2_0.2_" +repeat +".txt")
X_train = X_train[0:n,:]
X_train[:,2:4] = X_train[:,0:2] + np.random.normal(size=(X_train.shape[0],2)) * noisex
X_train[:,4] = X_train[:,5] + np.random.normal(size=X_train.shape[0]) * sy
test_dat = np.loadtxt("simu_nn2/nn_5_32_test_" + repeat +".txt")


results = np.zeros((3, 7)) #NN_train_ise, NN_test_ise, train_ise, t`rain_ll, test_ise, NN_test_iae, test_iae
best_predict = np.zeros((test_dat.shape[0], 2))

latent_dim = 2

ise_min = 1e8


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
    def __init__(self, noisey, noise):
        super(changeNoise, self).__init__()
        self.noisey = noisey 
        self.noise = noise

    def on_epoch_end(self, epoch, logs={}):
        #print("Setting noisey to =", str(K.get_value(self.noisey)))
        if epoch > 19: #and epoch % 10 == 0:   
          K.set_value(self.noisey,logs.get('mise2')) #
#        if epoch == 200:  
#          K.set_value(self.noise, 0.3**2)
#         elif epoch == 600:  
#           K.set_value(self.noise, 0.2**2)
        

noisey =  K.variable(0.1)
noise = K.variable(noisex**2)
#if anneal:
#	noise =  K.variable(0.1**2)
#else:
#	noise =  K.variable(0.3**2)
noiseparam = changeNoise(noisey, noise) # will change according to "noise", "noisey"



class LossLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(LossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, z, fz, w, y = inputs
       
        w = K.expand_dims(w, axis = 1)
        y = K.expand_dims(y, axis = 1)
        
        if laplace:
          reconstruction_loss = K.sum(K.square(y - fz), axis=-1)/noisey/2 + K.sum(K.abs(w - z), axis=-1) /noise
        else:
          reconstruction_loss = K.sum(K.square(y - fz), axis=-1)/noisey/2 + K.sum(K.square(w - z), axis=-1) /noise/2
       
        prior_loss = 1.5 * K.log(1 + K.square(z - prior_mean)/prior_var/2) # v = 2
        prior_loss = K.sum(prior_loss, axis=-1)
        
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



for i in np.arange(3): # repeat 

  model0 = build_model(layer1, nodes1, input_dim = latent_dim, activ='relu', regu = regu) # alpha for leaky relu

  ada = adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0004, amsgrad=False)

  model0.compile(loss=mse,optimizer=ada)  
  #model0.summary()

  history = model0.fit( X_train[:,latent_dim : (2*latent_dim)] , X_train[:,2*latent_dim],  #0: latent_dim
                       #validation_data=(X_val[:,latent_dim : (2*latent_dim)], X_val[:,2*latent_dim]), 
      batch_size=np.min([512,X_train.shape[0]]),epochs=200,verbose=0, shuffle=True) #, callbacks=[checkpointer]

  
  model00_predict = model0.predict(test_dat[:,0:latent_dim])
  results[i, 1] = np.mean((model00_predict.transpose() - test_dat[:,latent_dim])**2)
  results[i, 0] = history.history['loss'][-1]
 #np.mean((model0.predict(X_val[:,latent_dim : (2*latent_dim)]).transpose() - X_val[:,2*latent_dim])**2) 

  org_weight = model0.get_weights()

  if sy < 0.3:
    K.set_value(noisey, 0.1)
  else:
    K.set_value(noisey, 0.2)

  model1 = build_model(layer1, nodes1, input_dim = latent_dim, activ='relu', regu = regu2)

  model1.set_weights(org_weight)


  x = Input(shape=(latent_dim + 1,))
  x2 = Input(shape=(latent_dim,))
  x1 = Input(shape=(latent_dim,))
  y1 = Input(shape=(1,))

  model_mu = build_model(layer2, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = 1) #x1
  z_mu1 = model_mu(x)
  #z_mu = Add()([z_mu, x1])

  model_var = build_model(layer2-1, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = 1)
  z_log_var1 = model_var(x)

  z_sigma1 = Lambda(lambda t: K.exp(.5*t))(z_log_var1)

  eps1 = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0],mc_samples, 1)))

  z_eps1 = Multiply()([z_sigma1, eps1])
  z1 = Add()([z_mu1, z_eps1])

  model_mu2 = build_model(layer2+1, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = 1) #x2
  model_var2 = build_model(layer2-1, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = 1)

  input2 = Concatenate()([RepeatVector(mc_samples)(x2), z1])  # x need expand_dim

  z_mu2 = model_mu2(input2)
  z_log_var2 = model_var2(input2)
             
  z_sigma2 = Lambda(lambda t: K.exp(.5*t))(z_log_var2)
  eps2 = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0],
                                        mc_samples, 1)))

  z_eps2 = Multiply()([z_sigma2, eps2])
  z2 = Add()([z_mu2, z_eps2])
             

  z = Concatenate()([z1, z2])
  x_pred =  model1(z) 

#encoder = Model([x,eps1], [z_mu1,z_log_var1,model_mu2()])

  z_mu1 = RepeatVector(mc_samples)(z_mu1)
  z_log_var1 = RepeatVector(mc_samples)(z_log_var1)

  z_mu = Concatenate()([z_mu1, z_mu2])
  z_log_var = Concatenate()([z_log_var1, z_log_var2])


  z_mu0 = K.stop_gradient(z_mu)
  z_log_var0 = K.stop_gradient(z_log_var)
  z0 = K.stop_gradient(z)
  x0_pred =  model1(z0)

  vae_loss = LossLayer(name='LossLayer')([z_mu0, z_log_var0, z, x_pred, x1, y1]) 
  weight = WeightLayer(trainable = False,name='WeightLayer')(vae_loss) 

  output = Concatenate()([z, x_pred])
  vae = Model(inputs=[x,x2,x1, y1,eps1, eps2], outputs=output) # batch * MC * (latent_dim + 1)


  def mise2(yTrue, yPred):
    var_y = K.sum(K.square(yTrue[:,:,latent_dim:(latent_dim+1)]- yPred[:,:,latent_dim:(latent_dim+1)]), axis=-1)

    return K.mean(K.sum(var_y * weight, axis = 1))


  def customLoss(yTrue, yPred):
    loss = K.sum(vae_loss * K.square(weight), axis = 1) 

    reconstruction_loss0 =  (K.sum(K.square(yTrue[:,:,latent_dim:(latent_dim+1)]- x0_pred), axis=-1)) /noisey/2
    reconstruction_loss0 = K.sum(reconstruction_loss0 * (weight - K.square(weight)), axis = 1) 


    if laplace:      
        return K.mean(loss + reconstruction_loss0, axis = 0) + K.log(noise)*latent_dim + K.log(noisey)/2
    else:
        return K.mean(loss + reconstruction_loss0, axis = 0) + K.log(noise)*latent_dim/2 + K.log(noisey)/2


  ada = adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0004, amsgrad=False) #0.005 for noisex = 0.005

  vae.compile(optimizer=ada, loss=customLoss, metrics = [mise2]) #rmsprop,  weight_entropy
#vae.summary() 



  history = vae.fit([X_train[:,latent_dim:(latent_dim*2+1)],X_train[:,(latent_dim+1):(latent_dim*2+1)],
    X_train[:,latent_dim:(latent_dim*2)], X_train[:,latent_dim*2]],
    np.expand_dims(X_train[:,latent_dim:(latent_dim*2+1)], axis=1),
    shuffle=True,
    epochs=epochs,
    verbose = 0, 
    batch_size=np.min([batch_size,X_train.shape[0]]),
#   validation_data=(
#       [X_val[:,latent_dim:(latent_dim*2+1)],X_val[:,latent_dim:(latent_dim*2)],X_val[:,latent_dim*2]],
#       np.expand_dims(X_val[:,latent_dim:(latent_dim*2+1)], axis=1)
#   ),
  callbacks=[noiseparam ])

  model0_predict = model1.predict(test_dat[:,0:(latent_dim)]) 
  results[i, 4] = np.mean((model0_predict.transpose() - test_dat[:,latent_dim])**2)
  results[i, 6] = np.mean(np.abs(model0_predict.transpose() - test_dat[:,latent_dim]))
  results[i, 5] = np.mean(np.abs(model00_predict.transpose() - test_dat[:,latent_dim]))

  results[i, 3] = history.history['loss'][-1]
  results[i, 2] = history.history['mise2'][-1]
  
  if results[i,4] < ise_min:
    best_predict[:,1] = model0_predict[:,0]
    best_predict[:,0] = model00_predict[:,0]
    ise_min = results[i,4]
     

filename = prefix + "_" + repeat + "_" + str(noisex) +"_" +str(n)
np.savetxt(filename + ".txt", results)
# save prediction
if repeat == '0':
  np.savetxt(filename + "_prediction.txt", best_predict)

