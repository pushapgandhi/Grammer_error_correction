import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
import pickle
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
import model
from flask import Flask,jsonify,request,redirect,url_for
import flask
from  tensorflow.keras.preprocessing.sequence import pad_sequences
import io
import urllib, base64
import os
from datetime import datetime


app = Flask(__name__)


##### PREPROCESSING ############
def remove_spaces(text):
    text = re.sub(r" '(\w)",r"'\1",text)
    text = re.sub(r" \,",",",text)
    text = re.sub(r" \.+",".",text)
    text = re.sub(r" \!+","!",text)
    text = re.sub(r" \?+","?",text)
    text = re.sub(" n't","n't",text)
    text = re.sub("[\(\)\;\_\^\`\/]","",text)
    
    return text

def decontract(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

def preprocess(text):
    text = re.sub("\n","",text)
    text = remove_spaces(text)   # REMOVING UNWANTED SPACES
    text = re.sub(r"\.+",".",text)
    text = re.sub(r"\!+","!",text)
    text = decontract(text)    # DECONTRACTION
    text = re.sub("[^A-Za-z0-9 ]+","",text)
    text = text.lower()
    return text   

## Plot for alpha values ## 
def plot( input_sent , output_sent , alpha ) :

    input_words = input_sent.split() 
    output_words = output_sent.split() 
    fig, ax = plt.subplots()
    sns.set_style("darkgrid")
    sns.heatmap(alpha[:len(input_words),:], xticklabels= output_words , yticklabels=input_words,linewidths=0.01)
    ax.xaxis.tick_top() 

    
    for filename in os.listdir('static/images'):
        if filename.startswith('plt'):  # not to remove other images
            os.remove('static/images/' + filename)

    path =  "static/images/plt_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".jpg"
    plt.savefig(path)

    return  path


@app.route("/")
def hello():
    return flask.render_template("home.html", filename = ["static/images/book.jpg","static/images/blog.jpg","static/images/model.jpg"])

@app.route("/home")
def home():
    return flask.render_template("home.html", filename = ["static/images/book.jpg","static/images/blog.jpg","static/images/model.jpg"])


@app.route("/index")
def index():
    return flask.render_template("index.html",filename = "static/images/book.jpg") 

@app.route("/predict",methods=["POST","GET"])
def predict():
    text = request.form["sent"]
    text = preprocess(text)

    seq = tk_inp.texts_to_sequences([text])
    seq = pad_sequences(seq,maxlen = 20 , padding="post")
    state = model1.layers[0].initialize(1)
    enc_output,state_h,state_c= model1.layers[0](seq,state)

    pred = []
    
    input_state_h = state_h
    input_state_c = state_c
    prev_attention = np.zeros(shape = (1,20,1),dtype="float32")
    prev_attention[:,1] = 1 
    
    current_vec = tf.ones((1,1))
   
    alpha_values = []

    for i in range(20):
       
        fc , dec_state_h ,dec_state_c, alphas = model1.layers[1].layers[0](enc_output , current_vec ,input_state_h ,input_state_c,prev_attention)
       
        alpha_values.append(alphas)
       
        current_vec = np.argmax(fc , axis = -1)
        
        input_state_h = dec_state_h
        input_state_c = dec_state_c
        prev_attention = alphas
      
        pred.append(tk_out.index_word[current_vec[0][0]])
       
        if tk_out.index_word[current_vec[0][0]]=="<end>":
              break
 
    pred_sent = " ".join(pred)
   
    alpha_values = tf.squeeze(tf.concat(alpha_values,axis=-1),axis=0)

    html = plot( text , pred_sent , alpha_values ) 
    
    return flask.render_template("output.html",filename="static/images/book.jpg",input =request.form["sent"] ,output = pred_sent ,alpha= html)



if __name__ ==  "__main__":
    print("#######  LOADING ###########")
    tk_inp = pickle.load(open("load/tk_inp","rb"))

    tk_out = pickle.load(open("load/tk_out","rb"))

    model1 = model.encoder_decoder(enc_vocab_size=len(tk_inp.word_index)+1,
                            enc_emb_dim = 300,
                            enc_units=256,enc_input_length=35,
                            dec_vocab_size =len(tk_out.word_index)+1,
                            dec_emb_dim =300,
                            dec_units=256,
                            dec_input_length = 35,
                            
                            att_units=256,
                            batch_size=512,
                              att_mode = "dot")

    model1.compile(optimizer="adam",loss='sparse_categorical_crossentropy')
    model1.build([(512,35),(512,35)])
    
    model1.load_weights("load/best.h5")
    print("####### MODEL LOADED ###########")


    app.run(debug=True)