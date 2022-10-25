import os
import sys
import glob
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/img_oxt.pkl"

#train_size=15000
img_size=784
img_dim = (1, 28, 28)

def _change_one_hot_label(X):
    T = np.zeros((X.size, 3))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

def make_mnist(Label):
    datalist=[]
    path="./img/%s/"%(Label)
    fname=path+'*.jpg'
    img_files = glob.glob(fname)                                 #모든 이미지 불러오기
    
    for idx in range(len(img_files)):
        img=cv2.imread(img_files[idx], cv2.IMREAD_GRAYSCALE)     #Gray_Scale
        img=~img                                                 #Invert
        datalist.append(img)
        
    return datalist

def _load_img():
    global c0, c1, c2
    data0=make_mnist('0')
    data1=make_mnist('1')
    data2=make_mnist('2')
    c0=len(data0)
    c1=len(data1)
    c2=len(data2)
    data=data0+data1+data2
    data=np.array(data)
    print("이미지 갯수:", len(data), '(', c0, c1, c2, ')')
    data = data.reshape(-1, img_size)
    
    return data

##  라벨링
def _load_label():
    labels=['0', '1', '2']  #0: o, 1: x, 2: delta
    label_size=len(labels)

    train_size=c0+c1+c2
    #print(train_size)

    # train_label 을 0으로 초기화
    train_label=[0 for i in range(train_size)]      # train_label을 0으로 초기화    

    for i in range(c0, c0+c1):
        train_label[i]=1
    for i in range(c0+c1, train_size):
        train_label[i]=2
        
    train_label=np.array(train_label)
    
    #print(train_label[0])
    #print(train_label[5000])
    #print(train_label[10000])
    
    return train_label

def init_oxt():
    dataset={}
    dataset['train_img'] =  _load_img()
    dataset['train_label'] = _load_label()

    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")
    
def load_oxt(normalize=True, flatten=True, one_hot_label=False):
    
    if not os.path.exists(save_file):
        init_oxt()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        dataset['train_img'] = dataset['train_img'].astype(np.float32)
        dataset['train_img'] /= 255.0
    
    if one_hot_label:
        dataset['train_label']=_change_one_hot_label(dataset['train_label'])
    
    if not flatten:
        dataset['train_img'] = dataset['train_img'].reshape(-1, 1, 28, 28)
    
    # 종속변수(target)의 컬럼을 target으로의 선언이 필요합니다.
    train = dataset['train_img']
    target = dataset['train_label']
    
    # train data를 8:2로 train data와 test data로 분리
    x_train, x_test, t_train, t_test = train_test_split(train, target, 
                                                      test_size=0.2,
                                                      random_state=83,
                                                      shuffle=True,
                                                      stratify=target)
    
    return (x_train, t_train), (x_test, t_test)
