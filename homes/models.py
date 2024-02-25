import cv2
import mediapipe as mp
import time
import numpy as np

from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from tensorflow.keras.callbacks import *
import tensorflow as tf
from tensorflow.keras import *

from django.db import models
from .config import Config


def poses_diff(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    x = tf.subtract(x[:,1:,...],x[:,:-1,...])
    x = tf.image.resize(x,size=[H,W]) 
    return x
def poses_diff_2(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    x = tf.image.resize(x,size=[H,W]) 
    return x
def pose_motion_2(D, frame_l):
    x_1 = Lambda(lambda x: poses_diff_2(x))(D)
    x_1 = Reshape((frame_l,-1))(x_1)
    return x_1

def pose_motion(P,frame_l):
    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l,-1))(P_diff_slow)
    P_fast = Lambda(lambda x: x[:,::2,...])(P)
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l/2),-1))(P_diff_fast)
    x_1 = Reshape((frame_l,-1))(P)
    return P_diff_slow,P_diff_fast, x_1

def c1D(x,filters,kernel):
    x = Conv1D(filters, kernel_size=kernel,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def block(x,filters):
    x = c1D(x,filters,3)
    x = c1D(x,filters,3)
    return x
    
def d1D(x,filters):
    x = Dense(filters,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_FM(frame_l=32,joint_n=20,joint_d=3,feat_d=190,filters=16, nd=60, drop_rate=0.1): 
    P = Input(shape=(frame_l,joint_n,joint_d))
    diff_slow,diff_fast, x_1 = pose_motion(P,frame_l)

    x_1 = c1D(x_1, filters*2,1)
    x_1 = SpatialDropout1D(drop_rate)(x_1)
    x_1 = c1D(x_1, filters, 3)
    x_1 = SpatialDropout1D(drop_rate)(x_1)
    x_1 = c1D(x_1, filters,1)
    x_1 = MaxPooling1D(2)(x_1)
    x_1 = SpatialDropout1D(drop_rate)(x_1)

    x_d_slow = c1D(diff_slow,filters*2,1)
    x_d_slow = SpatialDropout1D(drop_rate)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,3)
    x_d_slow = SpatialDropout1D(drop_rate)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,1)
    x_d_slow = MaxPool1D(2)(x_d_slow)
    x_d_slow = SpatialDropout1D(drop_rate)(x_d_slow)

    x_d_fast = c1D(diff_fast,filters*2,1)
    x_d_fast = SpatialDropout1D(drop_rate)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,3) 
    x_d_fast = SpatialDropout1D(drop_rate)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,1) 
    x_d_fast = SpatialDropout1D(drop_rate)(x_d_fast)
   
    x = concatenate([x_1,x_d_slow,x_d_fast])
    x = block(x,filters*2)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(drop_rate)(x)
    
    x = block(x,filters*4)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(drop_rate)(x)

    x = block(x,filters*8)
    x = SpatialDropout1D(drop_rate)(x)
    
    return Model(inputs=[P],outputs=x)


def build_DD_Net(C):
    P = Input(name='P', shape=(C.frame_l,C.joint_n,C.joint_d)) 
    FM = build_FM(C.frame_l,C.joint_n,C.joint_d,C.feat_d,C.filters)
    
    x = FM([P])

    x = GlobalMaxPool1D()(x)
    
    x = d1D(x,128)
    x = Dropout(0.5)(x)
    x = d1D(x,128)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=[P], outputs=x)
    return model

def data_generator_rt(T):
    X = []

    T = np.expand_dims(T, axis = 0)
    for i in range(len(T)): 
        p = T[i]
        X.append(p)

    X = np.stack(X)
    return X

def get_data_info():
    labels_text = ['Xin chào, rất vui được gặp bạn!', 'Tạm biệt, hẹn gặp lại!', 'Xin cảm ơn, bạn thật tốt bụng!', 'Tôi xin lỗi, bạn có sao không','Tôi yêu gia đình và bạn bè.', 'Tôi là học sinh.', 'Tôi thích động vật.', 'Tôi ăn cơm.', 'Tôi sống ở Việt Nam.','Tôi là người Điếc']
    labels = ['xin chao rat vui duoc gap ban', 'tam biet hen gap lai', 'xin cam on ban that tot bung', 'toi xin loi ban co sao khong', 'toi yeu gia dinh va ban be', 'toi la hoc sinh', 'toi thich dong vat', 'toi an com', 'toi song o viet nam', 'toi la nguoi diec']

    return labels_text, labels

def setup():
    # load config
    C = Config()
    
    # load model
    model = build_DD_Net(C)
    
    # load weights
    print('Loading model')
    model.load_weights('pretrained_model/ddnet.h5')
    print('Loaded model')
    
    # load mediapipe config
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
    return model, mp_drawing, mp_drawing_styles, mp_pose
