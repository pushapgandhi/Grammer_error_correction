from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')


class Encoder(tf.keras.layers.Layer):

    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
    '''
    
    def __init__(self, vocab_size,emb_dims, enc_units, input_length,batch_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.input_length = input_length
        # INITIALIZING THE REQUIRED VARIABLES
        self.batch_size=batch_size # BATHCH SIZE
        self.enc_units = enc_units # ENCODER UNITS

        # EMBEDDING LAYER
        self.embedding= layers.Embedding(vocab_size ,emb_dims) 
        # LSTM LAYER WITH RETURN SEQ AND RETURN STATES
        self.lstm = layers.LSTM(self.enc_units,return_state= True,return_sequences =  True) 
        
    def call(self, enc_input , states):
        '''
        This function takes a sequence input and the initial states of the encoder.
        Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
        returns -- encoder_output, last time step's hidden and cell state
        '''
        # FORMING THE EMBEDDED VECTOR 
        emb = self.embedding(enc_input)
        # PASSING THE EMBEDDED VECTIO THROUGH LSTM LAYERS 
        enc_output,state_h,state_c = self.lstm(emb,initial_state=states)
        #RETURNING THE OUTPUT OF LSTM LAYER
        return enc_output,state_h,state_c 
    
    def initialize(self,batch_size):

        '''
        Given a batch size it will return intial hidden state and intial cell state.
        If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
        '''
        return tf.zeros(shape=(batch_size,self.enc_units)),tf.zeros(shape=(batch_size,self.enc_units))
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"vocab_size":self.vocab_size,"emb_dims":self.emb_dims, "enc_units":self.enc_units,"input_length":self.input_length,"batch_size":self.batch_size})
        return config


class Monotonic_Attention(tf.keras.layers.Layer):
    '''THIS FUNCTION RETURNS THE CONTEXT VECTOR AND ATTENTION WEIGHTS (ALPHA VALUES)'''
    def __init__(self,units,att_mode):
        super().__init__()
        self.units = units
        self.att_mode = att_mode
        # INITIALIZING THE DENSE LAYER W1
        self.W1 = layers.Dense(units)
        # INITIALIZING THE DENSE LAYER W2
        self.W2 = layers.Dense(units)
        # INITIALIZING THE DENSE LAYER V
        self.v = layers.Dense(1)
        self.mode = att_mode
        
    def call(self,enc_output,dec_state,prev_att):
        # HERE WE ARE COMPUTING THE SCORE 

        if self.mode == "dot":
        # FINDING THE SCORE FOR DOT MODEL
            dec_state =  tf.expand_dims(dec_state,axis=-1)
            score = tf.matmul(enc_output,dec_state)
            score = tf.squeeze(score, [2])
            
            
        if self.mode == "general":
        # FINDING THE SCORE FOR GENERAL MODEL
            dec_state =  tf.expand_dims(dec_state,axis=-1)
            dense_output = self.W1(enc_output)
            score = tf.matmul(dense_output , dec_state)
            score = tf.squeeze(score, [2])
            
            
        if self.mode == "concat":
        # FINDING THE SCORE FOR CONCAT MODEL
            dec_state =  tf.expand_dims(dec_state,axis=1)
            score = self.v(tf.nn.tanh(
                self.W1(dec_state)+ self.W2(enc_output)))
            score = tf.squeeze(score, [2])
        
        # AFTER THE SOCRES ARE COMPUTED THE SIGMOID IS USED ON IT
        probabilities = tf.sigmoid(score)

        # ATTENTION WEIGHTS FOR PRESENT TIME STEP
        probabilities = probabilities*tf.cumsum(tf.squeeze(prev_att,-1), axis=1)
        attention = probabilities*tf.math.cumprod(1-probabilities, axis=1, exclusive=True)
        attention = tf.expand_dims(attention,axis=-1)
        
        # CONTEXT VECTOR
        context_vec  =  attention  * enc_output
        context_vec = tf.reduce_sum(context_vec,axis=1)
        
        # RETURN CONTEXT VECTOR AND ATTENTION
        return context_vec, attention
    def get_config(self):
        config = super(Monotonic_Attention, self).get_config()
        config.update({"units":self.units,"att_mode":self.att_mode})
        return config

class Onestepdecoder(tf.keras.Model):
    '''THIS MODEL OUTPUTS THE RESULT OF DECODER FOR ONE TIME SETP GIVEN THE INPUT FOR PRECIOVE TIME STEP'''

    def __init__(self, vocab_size,emb_dims, dec_units, input_len,att_units,batch_size, att_mode):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.dec_units = dec_units
        self.input_len = input_len
        self.att_units = att_units
        self.batch_size = batch_size
        self.att_mode = att_mode

        # INTITALIZING THE REQUIRED VARIABLES
        # EMBEDDING LAYERS
        self.emb = layers.Embedding(vocab_size,emb_dims,input_length= input_len)
        # ATTENTION LAYER
        self.att = Monotonic_Attention(att_units,att_mode)
        # LSTM LAYER
        self.lstm = layers.LSTM(dec_units,return_sequences=True,return_state=True)
        # DENSE LAYER
        self.dense = layers.Dense(vocab_size,activation="softmax")

    def call(self, encoder_output , input , state_h,state_c,previous_attention):
        # FORMING THE EMBEDDED VECTOR FOR THE WORD
        # (32,1)=>(32,1,12)
        emb = self.emb(input)

        dec_output,dec_state_h,dec_state_c = self.lstm(emb, initial_state = [state_h,state_c] )

        # GETTING THE CONTEXT VECTOR AND ATTENTION WEIGHTS BASED ON THE ENCODER OUTPUT AND  DECODER STATE_H
        context_vec,alphas = self.att(encoder_output,dec_state_h,previous_attention)
        
        # CONCATINATING THE CONTEXT VECTOR(BY EXPANDING DIMENSION) AND ENBEDDED VECTOR
        dense_input =  tf.concat([tf.expand_dims(context_vec,1),dec_output],axis=-1)
        
        # PASSING THE DECODER OUTPUT THROUGH DENSE LAYER WITH UNITS EQUAL TO VOCAB SIZE
        fc = self.dense(dense_input)
        
        # RETURNING THE OUTPUT
        return fc , dec_state_h , dec_state_c , alphas


    def get_config(self):
        config=({ "vocab_size":self.vocab_size,"emb_dims":self.emb_dims,"dec_units": self.dec_units,"input_len": self.input_len,"att_units":self.att_units,"batch_size":self.batch_size, "att_mode":self.att_mode})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Decoder(tf.keras.Model):
    '''THIS MODEL PERFORMS THE WHOLE DECODER OPERATION FOR THE COMPLETE SENTENCE'''
    def __init__(self, vocab_size,emb_dims, dec_units, input_len,att_units,batch_size,att_mode):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.dec_units = dec_units
        
        self.att_units = att_units
        self.batch_size = batch_size
        self.att_mode = att_mode

        # INITIALIZING THE VARIABLES
        # LENGTH OF INPUT SENTENCE
        self.input_len = input_len
        # ONE STEP DECODER
        self.onestepdecoder = Onestepdecoder(vocab_size,emb_dims, dec_units, input_len,att_units,batch_size,att_mode)

    def call(self,dec_input,enc_output,state_h,state_c,initial_attention):
        # THIS VATIABLE STORES THE VALUE OF STATE_H FOR THE PREVIOUS STATE
        current_state_h = state_h 
        current_state_c = state_c
        previous_attention = initial_attention
        # THIS STORES THE DECODER OUTPUT FOR EACH TIME STEP
        pred = []
        # THIS STORED THE ALPHA VALUES
        alpha_values = []
        # FOR EACH WORD IN THE INPUT SENTENCE
        for i in range(self.input_len):
            
            # CURRENT WORD TO INPUT TO ONE STEP DECODER
            current_vec = dec_input[:,i]

            # EXPANDING THE DIMENSION FOR THE WORD
            current_vec = tf.expand_dims(current_vec,axis=-1)

            # PERFORMING THE ONE STEP DECODER OPERATION 
            dec_output,dec_state_h,dec_state_c,alphas = self.onestepdecoder(enc_output ,current_vec,current_state_h,current_state_c,previous_attention)

            #UPDATING THE CURRENT STATE_H
            current_state_h = dec_state_h
            current_state_c = dec_state_c
            previous_attention = alphas
            
            #APPENDING THE DECODER OUTPUT TO "pred" LIST
            pred.append(dec_output)

            # APPENDING THE ALPHA VALUES
            alpha_values.append(alphas)
            
        # CONCATINATING ALL THE VALUES IN THE LIST
        output = tf.concat(pred,axis=1)
        # CONCATINATING ALL THE ALPHA VALUES IN THE LIST
        alpha_values = tf.concat(alpha_values,axis = -1)
        # RETURNING THE OUTPUT
        return output , alpha_values
    def get_config(self):
      config = ({ "vocab_size":self.vocab_size,"emb_dims":self.emb_dims,"dec_units": self.dec_units, "input_len":self.input_len,"att_units":self.att_units,"batch_size":self.batch_size,"att_model":self.att_mode})
      return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
class encoder_decoder(tf.keras.Model):
    '''THIS MODEL COMBINES ALL THE LAYERS AND FORM IN ENCODER DECODER MODEL WITH ATTENTION MECHANISM'''
    def __init__(self,enc_vocab_size,enc_emb_dim,enc_units,enc_input_length,
            dec_vocab_size,dec_emb_dim,dec_units,dec_input_length ,att_units, batch_size,att_mode):
        # INITAILIZING ALL VARIABLES
        super().__init__()
        self.enc_vocab_size= enc_vocab_size
        self.enc_emb_dim=enc_emb_dim
        self.enc_units= enc_units
        self.enc_input_length =enc_input_length
        self.dec_vocab_size=dec_vocab_size
        self.dec_emb_dim=dec_emb_dim
        self.dec_units=dec_units
        self.dec_input_length =dec_input_length
        self.att_units=att_units
        self.att_mode=att_mode

        # BATCH SIZE
        self.batch_size = batch_size
        # INITIALIZING ENCODER LAYER
        self.encoder = Encoder(enc_vocab_size, enc_emb_dim,enc_units, enc_input_length,batch_size)
        # INITALIZING DECODER LAYER
        self.decoder = Decoder(dec_vocab_size ,dec_emb_dim,dec_units,dec_input_length  ,att_units, batch_size,att_mode)
        self.input_len = enc_input_length
        
        
    def call(self,data):
        # THE INPUT OF DATALOADER IS IN A LIST FORM FOR EACH BATCH IT GIVER TWO INPUTS
        # INPUT1 IS FOR ENCODER
        # INPUT2 IS FOR DECODER
        inp1 , inp2 = data
        # PASSING THE INPUT1 TO ENCODER LAYER
        enc_output, enc_state_h, enc_state_c = self.encoder(inp1,self.encoder.initialize(self.batch_size))
        # PASSING INPUT2 TO THE DECODER LAYER
        initial_attention = np.zeros(shape = (self.batch_size,self.input_len,1),dtype="float32")
        initial_attention[:,1] = 1 
        dec_output , alphas = self.decoder(inp2 , enc_output,enc_state_h,enc_state_c ,initial_attention)
        # THE OUTPUT OF MODEL IS ONLY DECODER OUTPUT THE ALPHA VALUES ARE IGNORED HERE
        return dec_output

    def get_config(self):
        config = ({"enc_vocab_size":self.enc_vocab_size, 
                  "enc_emb_dim":self.enc_emb_dim,"enc_units":self.enc_units,"enc_input_length":self.enc_input_length,\
            "dec_vocab_size":self.dec_vocab_size,
            "dec_emb_dim":self.dec_emb_dim,
            "dec_units":self.dec_units,
            "dec_input_length":self.dec_input_length ,\
            "att_units":self.att_units, "batch_size":self.batch_size,"att_mode":self.att_mode})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
