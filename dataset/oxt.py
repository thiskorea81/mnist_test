import os
import sys
import glob
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/oxt.pkl"

#train_size=3000
img_size=784
img_dim = (1, 28, 28)

def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.

    Parameters
    ----------
    x : 훈련 데이터
    t : 정답 레이블
    
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def del_bbox10(img30):
    
    b=img30
    # 행과 열 확인하기
    # h, w=img30.shape[:2]
    #print('h, w:', h, w)
    # 0열과 29열의 검은라인 지우기
    for i in range(w//30, 0, -1):
        right = 30*i-1
        left=30*(i-1)
        b=np.delete(b, right, axis=1)
        b=np.delete(b, left, axis=1)
    
    # 각 행에 검은라인 지우기
    for i in range(h//30, 0, -1):
        bottom = 30*i-1
        top=30*(i-1)
        b=np.delete(b, bottom, axis=0)
        b=np.delete(b, top, axis=0)
        
    return b 

def _change_one_hot_label(X):
    T = np.zeros((X.size, 3))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

def make_mnist(Label):
    global h, w
    datalist=[]
    path="./img_ox_10/%s/"%(Label)
    fname=path+'*.*'
    img_files = glob.glob(fname)                                 #모든 이미지 불러오기
    #print(len(img_files))
    for idx in range(len(img_files)):
        img=cv2.imread(img_files[idx], cv2.IMREAD_GRAYSCALE)     #Gray_Scale
        img=~img                                                 #Invert
        # 행과 열 확인하기
        h, w=img.shape[:2]
        data=del_bbox10(img)
        
    
        for i in range(h//30):
            for j in range(w//30):
                #dataset.setdefault('train_img', []).append()
                datalist.append(data[i*28:(i+1)*28, j*28:(j+1)*28])
        
    #print(len(datalist))
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
    print("이미지 갯수:", len(data), c0, c1, c2)
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
