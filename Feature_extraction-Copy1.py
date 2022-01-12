#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# # Feature Extraction

# In[3]:


header = 'filename chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmonics_mean harmonics_var perceptual_mean perceptual_var spectral_contrast1_mean spectral_contrast1_var spectral_constrast2_mean spectral_contrast2_var spectral_constrast3_mean spectral_contrast3_var  spectral_constrast4_mean spectral_contrast4_var spectral_constrast5_mean spectral_contrast5_var spectral_contrast6_mean spectral_contrast6_var spectral_contrast7_mean spectral_contrast7_var spectral_flatness_mean spectral_flatness_var tonnetz_1_mean tonnetz_1_var tonnetz_2_mean tonnetz_2_var tonnetz_3_mean tonnetz_3_var tonnetz_4_mean tonnetz_4_var tonnetz_5_mean tonnetz_5_var tonnetz_6_mean tonnetz_6_var '
for i in range(1, 21):
    header += f'mfcc_mean{i} '
    header+=  f'mfcc_var{i} '
for j in range(1,129):
    header += f'mel_specgram_mean{j} '
    header += f'mel_specgram_var{j} '
header +='  tempo'
header += ' label'
header = header.split()


# In[4]:


len(header)


# In[7]:


file = open('music.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'final_blues final_classical final_country final_disco final_hiphop final_jazz final_metal final_pop final_reggae final_rock'.split()
for g in genres:
    for filename in os.listdir(f'./final_genres/{g}'):
        songname = f'./final_genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        harmonics, perceptual = librosa.effects.hpss(y)
        tempo=librosa.beat.tempo(y=y,sr=sr)
        spectral_contrast=librosa.feature.spectral_contrast(y,sr=sr)
        #Spectral flatness
        spectral_flatness=librosa.feature.spectral_flatness(y=y)
        mel_spectrogram=librosa.feature.melspectrogram(y=y,sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.mean(rms)} {np.var(rms)} {np.mean(spec_cent)} {np.var(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)} {np.mean(harmonics)} {np.var(harmonics)} {np.mean(perceptual)} {np.var(perceptual)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
            to_append += f' {np.var(e)}'
        for j in range(0,7):
            to_append += f' {np.mean(spectral_contrast[j])}'
            to_append += f' {np.var(spectral_contrast[j])}'
            j=0
    
        to_append += f' {np.mean(spectral_flatness)}'
        to_append += f' {np.var(spectral_flatness)}'
        y_harmonic=librosa.effects.harmonic(y=y)
        tonnetz=librosa.feature.tonnetz(y=y_harmonic,sr=sr)
        for k in range(0,6):
            to_append += f' {np.mean(tonnetz[j])}'
            to_append += f' {np.var(tonnetz[j])}'
            k=0
        for l in range(0,128):
            to_append += f' {np.mean(mel_spectrogram[j])}'
            to_append += f' {np.var(mel_spectrogram[j])}'
        to_append += f' {float(tempo)}'
        to_append += f' {g}'
        file = open('music.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


# # Extracting Feature vector Target vector

# In[76]:


import pandas as pd
data=pd.read_csv('music.csv')
data.head()


# In[77]:


# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
data.head()


# In[78]:


data.isnull().sum().sum()


# In[79]:


#Target column
target=data['label']


# In[80]:


#Feature vector
feature=data.drop(columns='label',axis=1)


# In[81]:


#One hot  encoding the target variable
target=pd.get_dummies(target)
target.head()


# In[101]:


def split_transform_data(X,Y,test_size,random_state):
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(feature,target,test_size=test_size,random_state=random_state)
    print("The number of training examples is :{}\n".format(X_train.shape[0]))
    print("The number of testing  examples is :{}".format(X_test.shape[0]))
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    usable_data={
        "training_data_feature":X_train,
        "training_data_target":Y_train,
        "testing_data_feature":X_test,
        "testing_data_target":Y_test
    }
    return usable_data
    


# In[117]:


usable_data=split_transform_data(feature,target,0.1,42)


# In[118]:


#Training data
X_train=usable_data['training_data_feature']
Y_train=usable_data['training_data_target']
X_test=usable_data['testing_data_feature']
Y_test=usable_data['testing_data_target']


# In[119]:


def train_model(X_train,Y_train,epochs,validation_split):
    model=Sequential()
    model.add(Dense(512,input_shape=(341,),activation='relu'))
    model.add(Dense(512,input_shape=(341,),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    history=model.fit(x=X_train,y=Y_train,validation_split=validation_split,steps_per_epoch=20,epochs=epochs)
    validation_acc=history.history['val_accuracy'][-1]
    training_loss=history.history['loss'][-1]
    training_acc=history.history['accuracy'][-1]
    validation_loss=history.history['val_loss'][-1]
    trained_param={
        "trained_model":model,
        "training_history":history,
        "training_loss":training_loss,
        "training_acc":training_acc,
        "val_loss":validation_loss,
        "validation_acc":validation_acc
    }
    return trained_param
    


# In[121]:


histo=train_model(X_train,Y_train,300,0.2)


# In[122]:


model=histo['trained_model']


# In[123]:


model.evaluate(X_test,Y_test)

