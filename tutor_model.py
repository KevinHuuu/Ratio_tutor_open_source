'''
RNN model for  ratio tutor
'''
import keras
from keras.models import Model
from keras.layers import Input, Dropout, Masking, Dense, Embedding,Dropout,Bidirectional
from keras.layers import Embedding
from keras.layers.core import Flatten, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import merge
from keras.layers.merge import multiply
from keras.callbacks import EarlyStopping
from keras import backend as K
from theano import tensor as T
from theano import config
from theano import printing
from theano import function
from keras.layers import Lambda
import theano
import numpy as np
import pdb
from math import sqrt
from keras.callbacks import Callback
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from PhasedLSTM import PhasedLSTM as PLSTM
from numpy.random import seed
seed(1)

class TestCallback(Callback):
    def __init__(self, test_data,  file_name):
        self.x_test, self.y_test = test_data
        print('TestCallback initialized!')
        print('file_name ',file_name)
        self.file_name = file_name
        self.count = 0 # For the output files
        
    def on_epoch_begin(self, epoch, logs={}):

        y_pred = self.model.predict(self.x_test)
        #avg_rmse, avg_acc = self.rmse_masking(self.y_test, y_pred)
        #self.output_info(self.y_test, y_pred)
        #print('output file' + self.file_name)
        #print('\nTesting avg_rmse: {}\n'.format(avg_rmse))
        #print('\nTesting avg_acc: {}\n'.format(avg_acc))

    def output_info(self, y_true, y_pred):
        '''output_info'''
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        seq_true = y_true[y_true!=-1]
        seq_pred = y_pred[y_true!=-1]
        
        true_name = ('true'+str(self.count))
        pred_name = ('pred'+str(self.count))
        df = pd.DataFrame({true_name:seq_true,\
                          pred_name:seq_pred})
        self.count += 1
        #path = '/research/kevin/ratio_tutor/logs/'+ self.file_name
        #df.to_csv(path,mode = 'a')
        

    def rmse_masking(self, y_true, y_pred):
        #mask_matrix = np.sum(self.y_test_order, axis=2).flatten()
        num_users, max_sequences = np.shape(self.x_test)[0], np.shape(self.x_test)[1]
        #we want y_pred and y_true both to be matrix of 2 dim.
        if len(y_pred.shape) and len(y_true.shape) == 3:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        rmse = []
        acc = []
        padding_num = 0
        for user in range(num_users):
            diff_sq, response, correct = 0, 0, 0
            for i in range(user * max_sequences, (user + 1) * max_sequences):
                if y_true[i] == -1:
                    continue
                if y_true[i] == 1 and y_pred[i] >0.5:
                    correct += 1
                elif y_true[i] == 0 and y_pred[i] < 0.5:
                    correct += 1
                response += 1
                diff_sq += (y_true[i] - y_pred[i]) ** 2
            if response != 0:
                acc.append(correct/float(response))
                rmse.append(sqrt(diff_sq/float(response)))
        # print ('padding_num',padding_num)
        try:
            return sum(rmse)/float(len(rmse)), sum(acc)/float(len(acc))
        except:
            pdb.set_trace()


            

class tutor_net():
    def __init__(self,cut_mode, batch_size, epoch, hidden_layer_size,input_dim, output_dim, optimizer_mode, RNN_mode):
        self.input_dim = input_dim # we don't need to specify input_dim here
        self.output_dim = output_dim 
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_layer_size = hidden_layer_size
        

        self.optimizer_mode = optimizer_mode
        self.RNN_mode = RNN_mode
        self.cut_mode = cut_mode
        
        '''Choose optimizer and learning rate'''
        if self.optimizer_mode == 'RMSprop':
            self.optimizer =keras.optimizers.RMSprop()
            
        elif self.optimizer_mode == 'Adagrad':
            self.optimizer = keras.optimizers.Adagrad()
            
        elif self.optimizer_mode == 'Adamax':
            self.optimizer = keras.optimizers.Adamax()
        else:
            print('Lack choice of optimizer or learning_rate')
 
        print ("Model Initialized")
    
    #def custom_bce(self, y_true, y_pred):
        #b = K.not_equal(y_true, -K.ones_like(y_true))
        #b = K.cast(b, dtype='float32')
        #losses = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1) * K.mean(b, axis=-1)
        #count =  K.not_equal(losses, 0).sum()
        #return  losses.sum()/count

    '''
    def custom_cce(self, y_true, y_pred):
        b = K.not_equal(y_true, -K.ones_like(y_true))
        b = K.cast(b, dtype='float64')
        losses =K.categorical_crossentropy(target = y_true, output = y_pred)*b/K.sum(b)
        #print('custom_cce loss ',losses.sum().eval())
        return  losses.sum()
    '''
    def custom_cce(self, y_true, y_pred):
        b = K.not_equal(y_true, -K.ones_like(y_true))
        b = K.cast(b, dtype='float32')
        ans = K.categorical_crossentropy(target = y_true, output = y_pred) * K.mean(b, axis=-1)
        #ans = K.mean(K.categorical_crossentropy(target = y_true, output = y_pred), axis=-1) * K.mean(b, axis=-1)
        #count =  K.not_equal(ans, 0).sum()
        #return  ans.sum()/count
        return ans
    

    def build(self, train_index, train_response, \
                         val_index, val_response, file_name):
        self.train_index = train_index
        self.train_response = train_response
        self.val_index = val_index
        self.val_response = val_response
        self.file_name = file_name
        
        x = Input(batch_shape = (None, None, self.input_dim), name='x')
        masked = (Masking(mask_value= -1, input_shape = (None, None, self.input_dim)))(x)
        #RNN_out = SimpleRNN(self.hidden_layer_size, input_shape = (None, None, self.input_dim), return_sequences = True)(masked)
        
        #Notice dropout_W, dropout_U or dropout, recurrent_dropout
        if self.RNN_mode == 'SimpleRNN':
            #print('test return_sequences==False')
            RNN_out = SimpleRNN(self.hidden_layer_size, input_shape =\
                                (None, None, self.input_dim), return_sequences=True)(masked)
            #print('test PLSTM!!!')
            #RNN_out = PLSTM(self.hidden_layer_size, input_shape =\
                                #(None, None, self.input_dim), return_sequences=True)(masked)
            #RNN_out = SimpleRNN(self.hidden_layer_size, input_shape =\
                                #(None, None, self.input_dim), return_sequences=True)(masked)
            
        elif self.RNN_mode == 'LSTM':
            #print('test return_sequences == False with LSTM')
            #print('Test BLSTM!!!')
            #RNN_out = Bidirectional(LSTM(self.hidden_layer_size, input_shape =\
                                    #(None, None, self.input_dim), return_sequences = False), merge_mode='sum')(masked)
            RNN_out = LSTM(self.hidden_layer_size, input_shape =\
                                    (None, None, self.input_dim), return_sequences = True)(masked)
            
            #RNN_out = LSTM(self.hidden_layer_size, input_shape =\
                                    #(None, None, self.input_dim), return_sequences = True)(masked)
        elif self.RNN_mode == 'GRU':
            #print('test return_sequences == False with GRU')
            RNN_out = GRU(self.hidden_layer_size, input_shape =\
                                        (None, None, self.input_dim), return_sequences = True)(masked)           
            #RNN_out = GRU(self.hidden_layer_size, input_shape =\
                                        #(None, None, self.input_dim), return_sequences = True)(masked)
        else:
  
            print('Sth wrong with the RNN_mode')
        #Dropout
        drop_out = Dropout(rate = 0.5)(RNN_out)
        
        #activation selects softmax
        #print('test return_sequences==False,remove one dim ')
        #dense_out = Dense(self.output_dim, input_shape = (None, self.hidden_layer_size), activation='softmax')(drop_out)
        dense_out = Dense(self.output_dim, input_shape = (None, None, self.hidden_layer_size), activation='softmax')(drop_out)
        
        #earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        if self.cut_mode == 'no_cut':
            print("Earlystopping monitor is loss")
            earlyStopping = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto')
        else:
            print('Earlystopping monitor is val_loss')
            earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')   
        model = Model(inputs=x, outputs=dense_out)
        model.compile( optimizer = 'rmsprop',\
                       loss = self.custom_cce,\
                      #loss = 'categorical_crossentropy',\
                      metrics=['accuracy'])

        model.fit(self.train_index, self.train_response, batch_size = self.batch_size, \
                  epochs=self.epoch, \
                  callbacks = [ earlyStopping, \
                                TestCallback([self.val_index, \
                                self.val_response],self.file_name)],\
                                validation_data = [self.val_index,self.val_response], shuffle = True)
        return model


    def predict(self, model, X_test, Y_test):

        predictions = model.predict(X_test)
        Y_test = np.array(Y_test)
        if len(Y_test.shape) == 3:
            predictions = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1], predictions.shape[2]))
            Y_test = np.reshape(Y_test, (Y_test.shape[0]*Y_test.shape[1], Y_test.shape[2]))
        elif len(Y_test.shape)==2:
            predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
            Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))
        else:
            print('Sth wrong in predict_label()!!')
        
        
        
        print (predictions.shape)
        print (Y_test.shape)
        correct, total = 0, 0
        for i in range(len(predictions)):
            if np.sum(Y_test[i]) / Y_test[i].shape[0]== -1:
                continue
            y = np.where(np.array(Y_test[i])==np.max(np.array(Y_test[i])))[0][0]
            p = np.where(predictions[i]==np.max(predictions[i]))[0][0]
            if y == p:
                correct += 1
            total += 1
        print (correct, total)
        print (correct / float(total))
        return (correct / float(total))


    def predict_label(self, model, X_test, Y_test):

        predictions = model.predict(X_test)
        Y_test = np.array(Y_test)
        if len(Y_test.shape) == 3:
            predictions = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1], predictions.shape[2]))
            Y_test = np.reshape(Y_test, (Y_test.shape[0]*Y_test.shape[1], Y_test.shape[2]))
        elif len(Y_test.shape)==2:
            predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
            Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))
        else:
            print('Sth wrong in predict_label()!!')
        label_to_integer = pd.read_csv("label_to_integer.csv", sep=",")
        label_list = list(label_to_integer.columns.str.strip())
        integer_list = list(label_to_integer.iloc[0])
        #label_to_integer = label_to_integer.iloc[0].to_dict()
        #integer_to_label = dict(zip(label_to_integer.values(), label_to_integer.keys()))

        #label_list = list(label_to_integer.keys())
        #integer_list = list(label_to_integer.values())

        print (predictions.shape)
        print (Y_test.shape)
        y_true = []
        y_pred = []
        for i in range(len(predictions)):
            if np.sum(Y_test[i]) / Y_test[i].shape[0] == -1:
                continue
            y = np.where(np.array(Y_test[i])==np.max(np.array(Y_test[i])))[0][0]
            p = np.where(predictions[i]==np.max(predictions[i]))[0][0]
            y_true.append(y)
            y_pred.append(p)

        confusion = confusion_matrix(y_true, y_pred, labels=integer_list)
        recall = recall_score(y_true, y_pred, labels=integer_list, average=None)
        accuracy = accuracy_score(y_true, y_pred)

        info_matrix = pd.DataFrame(np.vstack((confusion, recall)), index=(label_list+['accu']), columns=label_list)
        info_matrix['sum'] = info_matrix.sum(axis=1)
        info_matrix.loc['accu','sum'] = accuracy

        print (info_matrix)
        return info_matrix

    def predict_distribution(self, model, X_test):
        # return model.predict
        predictions = model.predict(X_test)     
        return predictions
        
        
