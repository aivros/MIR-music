# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
import pyaudio
import librosa
from scipy.signal import kaiserord, firwin, butter, lfilter, freqz
import scipy.signal as signal
import scipy
import numpy as np
import matplotlib.pyplot as plt
import operator
import time                
import multiprocessing
import threading
from pylive import live_plotter
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import functools
import random as rd
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time

vibration_command = False
prediction_list = [0]
gate_ = 0
gate_high = 1
gate_low = 0
mean_high = 1
mean_low = 0
low_frequency_ = []
gate_send = 0
low_frequency_send = []
silence = False
correction_time = 0

data_list = [] #list()
data = ''
data_time = 0
audio = np.zeros(1)
audio_chunk_time = [0]
low_hight = True
#------------------------------------------------------------------------------
def getParity(n):
    parity = 0
    while n:
        parity = ~parity
        n = n & (n - 1)
    return parity

#------------------------------------------------------------------------------
#返回当前时间
def current_milli_time():
    current_milli_time = int(round(time.time() * 1000))
    return current_milli_time

#-------------------------------------------------------------------------------------


#滤波器
#@jit#(nopython=True) # Set "nopython" mode for best performance
    
def low_pass(data,sample_rate):     
    # 设计滤波器
    #sample_rate = 100.0
    nyq_rate = sample_rate / 2.0
    width = 10.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 90.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # The cutoff frequency of the filter.
    cutoff_hz = 300.0
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, data)         
    return filtered_x

def low_filter(y,sr):

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    # Filter requirements.
    order = 1
    fs = sr  # sample rate, Hz
    cutoff = 300  # desired cutoff frequency of the filter, Hz
    data = y
    # Get the filter coefficients so we can check its frequency response.
    #b, a = butter_lowpass(cutoff, fs, order)
    # Filter the data, and plot both the original and filtered signals.
    low_frequency_data = butter_lowpass_filter(data, cutoff, fs, order)
    return low_frequency_data

def high_filter(y,sr):

    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Filter requirements.
    order = 1
    fs = sr  # sample rate, Hz
    cutoff = 2000  # desired cutoff frequency of the filter, Hz
    data = y
    # Get the filter coefficients so we can check its frequency response.
    #b, a = butter_highpass(cutoff, fs, order)
    # Filter the data, and plot both the original and filtered signals.
    high_frequency_data = butter_highpass_filter(data, cutoff, fs, order)
    return high_frequency_data

def highpass_filter(y, sr):
    filter_stop_freq = 2000  # Hz
    filter_pass_freq = 3000  # Hz
    filter_order = 2
    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = scipy.signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
    # Apply high-pass filter
    filtered_audio = scipy.signal.filtfilt(filter_coefs, [1], y)
    return filtered_audio

#------------------------------------------------------------------------------------
#峰值检测
def peak_track(x,sr):
    hop_length = 256
    onset_envelope = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length)
    N = len(x)
    T = N / float(sr)
    t = numpy.linspace(0, T, len(onset_envelope))
    onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, 5)
    return onset_envelope
#-------------------------------------------------------------------------------------
#节奏检测
def beat_track(x,sr):
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, units='time')
    print(tempo)
    print(beat_times)
    return tempo, beat_times

def tempo_track(x,sr):
    hop_length = 200  # samples per frame
    onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length, n_fft=2048)
    S = librosa.stft(onset_env, hop_length=1, n_fft=512)
    fourier_tempogram = np.absolute(S)
    tmp = np.log1p(onset_env[n0:n1])
    r = librosa.autocorrelate(tmp)
    tempo = librosa.beat.tempo(x, sr=sr)
    T = len(x) / float(sr)
    seconds_per_beat = 60.0 / tempo[0]
    beat_times = numpy.arange(0, T, seconds_per_beat)
    return beat_times

#-----------------------------------------------------------------------------------
#鼓转录
def drum_track(x,r):
    drum_onsets = ADT([filename])[0]
    clicks = librosa.clicks(times=drum_onsets['Kick'], sr=sr, length=len(x))
    clicks = librosa.clicks(times=drum_onsets['Snare'], sr=sr, length=len(x))
    clicks = librosa.clicks(times=drum_onsets['Hihat'], sr=sr, length=len(x))
    # Compute average drum beat signal.
    frame_sz = int(0.100 * sr)
    def normalize(z):
        return z / scipy.linalg.norm(z)
    drum_type = ['Kick', 'Snare', 'Hihat']
    drum_track_list = []
    for iii in drum_type:
        onset_samples = librosa.time_to_samples(iii, sr=sr)
        x_avg = numpy.mean([normalize(x[i:i + frame_sz]) for i in onset_samples], axis=0)
        # Compute average spectrum.
        X = librosa.spectrum.fft.fft(x_avg)
        Xmag = librosa.amplitude_to_db(abs(X))
#-----------------------------------------------------------------------------------
#联合高频热力谱，光谱识别
#-----------------------------------------------------------------------------------

def point_index_get(y,sr,time_zero):
    time_zero = time_zero - int(CHUNK/44.1)
    #print(f'time_zero: {time_zero}')
    if not isinstance(y,np.ndarray):
       y = np.array(y, dtype = np.float64)
    #print(f'y_shape: {y.shape}, y: {y}')
    sr = sr*2#??????????????????????????????????????????????????????????
    #print(f'crrenttime: {int(round(time.time() * 1000))}')
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    #print(f'crrenttime: {int(round(time.time() * 1000))}')
    beats = np.around(beats * 1000)
    beats = beats.astype(int)
    beats = beats + time_zero
    #print(f'time: {time_zero},coreenttime: {int(round(time.time() * 1000))}, beats: {beats}')
    return beats

#------------------------------------------------------------------------------------
#F幅值过滤
def amplitude_filter(filtered,gate):    
    #filtered_mute = filtered*1.0/(max(abs(filtered)))#wave幅值归一化
    #print(filtered)
    filtered[abs(filtered) < gate * 0.8] = 0
    #print(filtered)
    if filtered.any():
        xxx =  1
    else:
        xxx =  0            
    return xxx

def pre_amplitude_filter(filtered, gate):
    # filtered_mute = filtered*1.0/(max(abs(filtered)))#wave幅值归一化
    # print(filtered)
    filtered[abs(filtered) < gate * 0.8] = 0
    # print(filtered)
    if filtered.any():
        xxx =  filtered
    else:
        xxx = 0
    return xxx

#----------------------------------------------------------------------------------------------------
def difference_func(data):
    comparison_list = []
    for i in range(0,len(data)-1,1):
        comparison_ = data[i+1] - data[i]
        comparison_list.append(comparison_)
    return comparison_list

def multiple_func(x_nonzero,distance,comparison_list_num):
    multiple_list = []
    comparison_list_num = comparison_list_num[x_nonzero]
    for item,value in enumerate(comparison_list_num):
        if item == len(comparison_list_num)-1:
            break
        else:
            if abs(distance*2 - (comparison_list_num[item+1] + value)) < 20:
                multiple_list.append(comparison_list_num[item+1])
                multiple_list.append(value)                
    return multiple_list

def distance_func(x_zero,data):    
    #print('orgin: ',x_zero)
    distance_list = []
    x_zero = np.array(x_zero)
    x_zero = x_zero[0]    
    #print('ooo: ',x_zero)     
    if x_zero.any() != 0:            
        x_zero_continuous,last_index = continuous_max(x_zero)
        #print(x_zero_continuous)                   
        for item,value in enumerate(data):
            if item in x_zero_continuous:
                distance_list.append(value)                
        if len(distance_list) < 3:            
            return 0,0,0
        else:            
            distance = np.mean(np.array(distance_list))            
            distance_len = len(distance_list)        
            return distance,distance_len,last_index        
    else:        
        return 0,0,0

def continuous_max(data):
    #print(data)
    data = np.insert(data,0,data[0] + 123)
    data = np.append(data, data[-1] + 123)
    #print(data)
    data_0 = data[0:-1]
    data_1 = data[1:]    
    data_x = data_1 - data_0
    #print(data_x)    
    data_x_0 = data_x[0:-1]
    data_x_1 = data_x[1:]    
    data_x_x =  data_x_1 - data_x_0
    #print(data_x_x)    
    x_nonzero = data_x_x.nonzero()  #取出矩阵中的非零元素的坐标
    #print(x_nonzero)    
    x_nonzero_num = np.array(x_nonzero)
    x_nonzero_num = x_nonzero_num[0]
    #print(x_nonzero_num)        
    if x_nonzero_num.any() != 0:
        #x_nonzero_num_ = data_x_x[x_nonzero]
        #print(x_nonzero_num_)        
        x_nonzero_num_0 = x_nonzero_num[0:-1]
        x_nonzero_num_1 = x_nonzero_num[1:]        
        #print(x_nonzero_num_0,x_nonzero_num_1)        
        x_nonzero_num_x = x_nonzero_num_1 - x_nonzero_num_0
        #print(x_nonzero_num_x)        
        #x_nonzero_num_x_max = max(x_nonzero_num_x)
        #print(x_nonzero_num_x_max)        
        x_nonzero_num_x_max_index = np.argmax(x_nonzero_num_x)
        #print(x_nonzero_num_x_max_index)        
        continuous_max_star_index = x_nonzero_num[x_nonzero_num_x_max_index] + 1
        continuous_max_end_index = x_nonzero_num[x_nonzero_num_x_max_index + 1] + 2          
        #print(continuous_max_star_index,continuous_max_end_index)        
        data_ = data[continuous_max_star_index:continuous_max_end_index]
        #print(data_)        
        last_index = continuous_max_end_index                
    else:
        data_ = data
        last_index = len(data) - 1    
    return data_,last_index
    
def start_time(time_box,last_index,distance):    
    startime = time_box[last_index]  + (len(time_box) - 1 - last_index)*distance    
    return startime

def rhythm_point(time_box):
    zero_len = {}    
    distance_dict = {}
    time_point = {}
    time_box_ = time_box[::-1]
    if not all(time_box_):
        time_box = [i for i in time_box if i != 0]
    #print(f'time_box: {time_box}, corrent_time: {int(round(time.time() * 1000))}')
    #print(time_box_)
    #print(len(self.time_box))    
    for ii,vv in enumerate(time_box_):
        distance = time_box_[0] - vv        
        #print('distance: ',distance)
        if distance < 2000 and distance:
            #time_len = time_box[-1] - time_box[0] 
            distance_list = [time_box_[0] - i*distance for i in range(len(time_box_)) if
                             time_box_[0] - i*distance > time_box_[-1]]
            #print(self.time_box[-1])
            #print(f'distance_list: {distance_list}')
            _ = []
            time_ = []
            for m in distance_list:
                for n,v in enumerate(time_box_):
                    if m > v - 20 and m < v + 20 :
                        _.append(n)
                        time_.append(v)
            zero_len[m] = len(_)            
            distance_dict[m] = distance
            time_point[m] = time_            
            #print(zero_len, distance_dict, time_point)
        else:
            #break
            None
    #将最多个数与其匹配的原数据取出，并重新计算距离，生成list，再一次匹配，直到取得最大匹配数。将所有最大匹配原数据做距离平均值计算，作为输出距离。
    if bool(zero_len):        
        max_key = max(zero_len.items(), key=operator.itemgetter(1))[0]
        max_num_list = time_point[max_key]
        #print(max_num_list)
        comparison_list = difference_func(max_num_list)  # 一阶差分
        #print(comparison_list)
        comparison_diff = difference_func(comparison_list)  # 二阶差分
        # print(comparison_diff)
        comparison_diff = np.array(comparison_diff)
        #print(comparison_diff)
        comparison_diff[abs(comparison_diff) == 0] = 1001
        comparison_diff[abs(comparison_diff) < 50] = 0
        #print(comparison_diff)
        comparison_list_num = np.array(comparison_list)
        x_zero = np.where(comparison_diff == 0)
        # print(x_zero)
        x_zero_ = comparison_list_num[x_zero]
        #print(x_zero_)
        distance, distance_len, last_index_ = distance_func(x_zero, comparison_list_num)
        #distance = np.mean(np.array(difference_func(max_num_list[::-1])))
        #print(distance)
        len_num = zero_len[max_key]
        #print('max_: ',max_key)
        #max_time_list = {}
        #print(distance,distance_dict[max_key],len_num)
        def distance_dynamic(distance,len_num):
            distance_list = [int(time_box_[0] - i * distance) for i in range(len(time_box_)) if
                             time_box_[0] - i * distance > time_box_[-1]]
            # print(self.time_box[-1])
            #print(f'distance_list: {distance_list}')
            _ = []
            for m in distance_list:
                for n, v in enumerate(time_box_):
                    if m > v - 5 and m < v + 5:
                        _.append(v)
            #max_time_list[m] = _
            distance = np.mean(np.array(difference_func(_[::-1])))
            #print(distance)
            if len(_) > len_num:
                len_num = len(_)
                return distance_dynamic(distance, len_num)
            else:
                len_num = len(_)
                return distance, len_num
        #print(f'hhh: {distance,len_num}')
        if zero_len[max_key] > 4:   #符合节奏的点的个数            
            #print('len: :',zero_len[max_key])    
            return distance,max_key
    else:
        return 0,0
    return 0,0

def format_rate(format_value):
    format_ = 0
    #format_value = str(format_value)
    if format_value == 16:
        format_ = 1
    elif format_value == 8:
        format_ = 2
    elif format_value == 4:
        format_ = 3        
    elif format_value == 2:
        format_ = 4
    return format_   

def audio_set(RATE):
    RATE_0 = RATE/8000 
    RATE_1 = RATE/11025    
    try:
        if RATE_0 - int(RATE_0) == 0:
            b = 2972*3
            r = RATE_0
        elif RATE_1 - int(RATE_1)  == 0:
            b = 4096*3
            r = RATE_1
        else:
            raise AssertionError()                 
    except:
        print("At present, only integer multiple sampling rates of 8000 Hz and 11025 Hz can be identified.")
        print("Please convert the sampling rate first.")
        print()   
    finally:        
        assert (RATE_0 - int(RATE_0) == 0) or (RATE_1 - int(RATE_1))  == 0,'The sampling rate does not match.'    
    return b,r
   
def block_length_func(CHANNELS,FORMAT_,RATE):    
    RATE_0 = RATE/8000 
    RATE_1 = RATE/11025
    x = 1
    bitx = 0
    try:
        if CHANNELS == 1:
            bitx = FORMAT_
        elif CHANNELS == 2:
            bitx = FORMAT_*2       
        else:
            x = 0
            raise Exception("Currently only 1 to 2 channels can be processed.")
    except Exception as error:
        print(error)
        print()
        assert x, "Please convert channels."
    finally:
        assert (RATE_0 - int(RATE_0) == 0) or (RATE_1 - int(RATE_1))  == 0,'The sampling rate does not match.'
    return bitx   

def chunk_block(chunk,size):
    num = len(chunk)
    b_ = 0
    new_chunk = []
    while True:
        if b_ + size <= len(chunk):
            unit = chunk[b_::b_+size]
            unit_mean = np.mean(unit)
            #print(unit_mean)
            unit_ = [unit_mean]*size
            new_chunk.extend(unit_)
        else:
            break
        b_ += size
    if len(new_chunk) < num:
        ex = [i - i for i in range(num - len(new_chunk))]
        new_chunk.extend(ex)
    new_chunk = np.array(new_chunk)
    return new_chunk
#-------------------------------------------------------------------------------------------------------------
# self.starting_point,self.prediction_num历史数据计算模块
# 时间节点均匀度计算
# 时间节点起始点计算
# 距离误差成立
# 平均距离值计算
#--------------------------------------------------------------------------------------------------------------
#主数据流程
ch_ = 72 #read_music_data
#ch_ = 36 #pre_read_music_data
CHUNK = 128*ch_ #3072
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
FORMAT_ = format_rate(FORMAT)
#------------------------------------------------------------------------------------            
#静态预读检测模块
class pre_read_music_data(object):
    def __init__(self):
        self.b, self.r = audio_set(RATE)
        self.audio = np.zeros(int(((self.b * self.r) * 10)))
        self.audio_chunk_time = [0]*len(self.audio)
        self.bitx = block_length_func(CHANNELS, FORMAT_, RATE)
        self.block_length = int(((CHUNK / 2 * self.bitx)))
        self.time_list = [0] * int((self.b * self.r)/CHUNK) * 10
        self.sys_delay = 150
        self.drum_point_delay = 80
        self.bpm_time = []
        self.time_box = [0] * 20
        # self.time_box_main = [0]*4
        self.last_time = 0
        # back_play = []
        self.back_play_point = []
        self.prediction_num_x = 1
        self.starting_point = [0]
        self.prediction_num = [0]
        self.point_check = 0
        self.loop_main_time = 0
        self.prediction_list = [0]
        self.low_frequency_ = []
        # self.prediction_generator = False
        self.prediction = False
        self.vibration_command_prediction = False
        # drum_point_get = False
        # pause = False
        self.vibration_command = False
        # self.vibration_command_plot = []
        self.gate = 0
        self.ggg = 0
        self.lll = 0
        self.hhh = 1
        self.back_play_point.append(self.hhh)
        self.lll += 1
        self.vvv = 0
        print(f'self: {int(round(time.time() * 1000))}')

    def main_func(self, data, data_time):
        global vibration_command
        global gate_
        global low_frequency_
        global prediction_list
        global silence
        global gate_send
        global low_frequency_send
        global gate_high
        global gate_low
        global mean_low
        global mean_low
        global correction_time
        #print(f'start: {int(round(time.time() * 1000))},--------------------------------------------------------------')
        #currnt_time_chunk = data_time
        self.audio = data
        self.audio_chunk_time = data_time
        sample_rate = RATE
        #print(f'silence: {len(self.audio[5520::])}, {self.audio[5520::]}')
        if sum(abs(self.audio[5520::])) < 300000000:
            print(f'静音🔇')
            silence = True
            return
        else:
            silence = False
        if len(self.audio) > 303:
            high_frequency_ = high_filter(self.audio, sample_rate)
            #print(current_milli_time())
            #low_filter(y, sr)
            self.low_frequency_ = low_filter(self.audio, sample_rate)
            #self.low_frequency_ = low_pass(self.audio, sample_rate)
            #print(f'low_frequency: {self.low_frequency_}')
            #print(f'filter : {current_milli_time()}')
        else:
            print('opps')
        if self.audio.any() != 0:
            # self.audio = self.audio*1.0/(max(abs(self.audio)))#wave幅值归一化
            self.gate = max(abs(self.audio))
            # gate_high = max(abs(high_frequency_))
            # gate_low = max(abs(low_frequency_))
        else:
            print('continue')
        # print(gate,gate_high,gate_low,gate_high/gate)
        gate_ = self.gate
        # 选定频率放大
        # 幅值过滤
        # print(current_milli_time())
        #print(f'crrenttime: {int(round(time.time() * 1000))}')
        amplitude_filter_list = pre_amplitude_filter(self.low_frequency_, gate_)
        #print(f'crrenttime: {int(round(time.time() * 1000))}')
        if isinstance(amplitude_filter_list, np.ndarray):
            max_ = max(amplitude_filter_list)
        else:
            max_ = 0
        #print(f'amplitude_filter_list: {amplitude_filter_list},max: {max_},gate: {gate_}')
        if isinstance(amplitude_filter_list, np.ndarray):
            #print(f'mmm')
            if amplitude_filter_list.any() and any(self.audio_chunk_time[0:self.block_length]):
                #print(f'crrenttime: {int(round(time.time() * 1000))}')
                self.time_box = point_index_get(amplitude_filter_list, sample_rate, self.audio_chunk_time[self.block_length])
                #print(f'self.time_box: {self.time_box}')
                #print(f'self.audio_chunk_time end: {self.audio_chunk_time[-1]}')
                #print(f'crrenttime: {int(round(time.time() * 1000))}')
                self.vibration_command = True
            else:
                #print(f'kkkk')
                self.vibration_command = False
                return
        else:
            #print(f'uuuu')
            self.vibration_command = False
            return
        if self.vibration_command:
            self.vibration_command = False
            #print('ooo')
            #print(f'crrenttime: {int(round(time.time() * 1000))}')
            rhythm_point_, last_index_ = rhythm_point(self.time_box)
            #print(f'crrenttime: {int(round(time.time() * 1000))}')
            #print(f'rhythm_point_: {rhythm_point_}, last_index_: {last_index_}')
            if rhythm_point_:
                self.prediction = True
                distance = rhythm_point_
                startime = last_index_
                #print(distance, startime)
                #print(f'crrenttime: {int(round(time.time() * 1000))}')
            '''
            comparison_list = difference_func(self.time_box) #一阶差分
            print(comparison_list)

            comparison_diff = difference_func(comparison_list) #二阶差分
            # print(comparison_diff)

            comparison_diff = np.array(comparison_diff)
            print(comparison_diff)

            comparison_diff[abs(comparison_diff) == 0] = 1001

            comparison_diff[abs(comparison_diff) < 50] = 0
            print(comparison_diff)

            x_nonzero = comparison_diff.nonzero()  # 取出矩阵中的非零元素的坐标
            # print(x_nonzero)
            print(comparison_diff[x_nonzero])
            x_nonzero_ = len(comparison_diff[x_nonzero])

            comparison_list_num = np.array(comparison_list)
            x_zero = np.where(comparison_diff == 0)
            # print(x_zero)
            x_zero_ = comparison_list_num[x_zero]
            print(x_zero_)

            if x_nonzero_ < 15:
                print('uuu')

                distance_, distance_len, last_index_ = distance_func(x_zero, comparison_list_num)

                if x_nonzero_ < 3:
                    distance = int(np.mean(np.array(x_zero_)))

                    startime = start_time(self.time_box, last_index_, distance)
                    print('kkk')

                    self.prediction = True


                elif distance_len > 4:

                    self.prediction = True

                    distance = distance_

                    startime = start_time(self.time_box, last_index_, distance)

                else:
                    print('mmm')

                    # distance,distance_len = distance_func(x_zero,comparison_list_num)
                    # print('distance: ',distance_)

                    if distance_:

                        multiple_list = multiple_func(x_nonzero, distance_, comparison_list_num)
                    else:
                        multiple_list = []

                    print(multiple_list)

                    if len(multiple_list) > 10:
                        self.prediction = True

                        distance = distance_

                        startime = start_time(self.time_box, last_index_, distance)'''
            #print(f'crrenttime: {int(round(time.time() * 1000))}')
            if self.prediction:
                print('666')
                self.starting_point.append(int(startime))
                self.prediction_num.append(int(abs(distance)))
                self.pred_ = self.starting_point[-1] - self.sys_delay - self.drum_point_delay
                self.prediction_list = [self.pred_ + self.prediction_num[-1] * xxx for xxx in range(60)]
                prediction_list = self.prediction_list
                #print(f'self.prediction_list: {self.prediction_list}')
                #print(f'666_crrenttime: {int(round(time.time() * 1000))}')
                self.prediction = False
                gate_send = gate_
                low_frequency_send = self.low_frequency_
                correction_time = self.audio_chunk_time[-1]

                gate_high = max(abs(high_frequency_))
                mean_high = np.mean(abs(high_frequency_))

                gate_low = max(abs(self.low_frequency_))
                mean_low = np.mean(abs(self.low_frequency_))
            else:
                None
        else:
            None
read_data = pre_read_music_data()
#read_data.main_func(data,data_time)
thread_condition = threading.Condition()
class Low_Hight_Thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global low_hight
        while True:
            dataTem = b''
            thread_condition.acquire()
            #if(len(data_list) == 0):
                #thread_condition.wait()
            #dataTem = data_list.pop(0)
            #if not data:
                #thread_condition.wait()
            dataTem = audio[111520::]
            if len(dataTem) > 303:
                high_frequency_ = highpass_filter(dataTem, RATE)
                low_frequency_ = low_filter(dataTem, RATE)
                low_hight_ = sum(abs(low_frequency_))/sum(abs(high_frequency_))
                if low_hight_ > 1:
                    low_hight = True
                else:
                    low_hight = False
            thread_condition.release()
            #print(f'eee: {int(round(time.time() * 1000))}')

class AudioRecordThread(threading.Thread,pre_read_music_data):
    def __init__(self):
        threading.Thread.__init__(self)
        pre_read_music_data.__init__(self)
    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        global data_time, data, audio, audio_chunk_time
        audio = np.zeros(int(((self.b * self.r) * 10)))
        audio_chunk_time = [0] * len(audio)
        while True:
            #print(f'0000000: {int(round(time.time() * 1000))}')
            # currnt_time_chunk = int(round(time.time() * 1000))
            data = stream.read(CHUNK, exception_on_overflow=False)
            thread_condition.acquire()
            #print('data: ',data)
            #data_list.insert(0,data)
            #print('data_list: ',data_list)
            data_time = int(round(time.time() * 1000))
            #print(f'sss: {int(round(time.time() * 1000))}')
            time_data = [data_time] * self.block_length
            #print(len(audio_chunk_time))
            del audio_chunk_time[0:self.block_length]
            audio_chunk_time.extend(time_data)
            #print(len(audio_chunk_time))
            #del self.time_list[0]
            #self.time_list.append(time_data)
            str_data = data
            #print(f'str_data: {len(str_data)}')
            wave_data = np.frombuffer(str_data,
                                      dtype=np.int16)  # np.fromstring(str_data,dtype=np.int16)#将字符串转化为int
            #print(f'wave_data: {len(wave_data)}')
            block_size = 88
            wave_data = chunk_block(wave_data,block_size)
            wave_data = list(wave_data)
            audio = list(audio)
            #print(len(audio))
            # print(len(wave_data))
            del audio[0:self.block_length]
            audio.extend(wave_data)
            #print(len(audio))
            audio = np.array(audio, dtype=int)
            thread_condition.notify(1)
            thread_condition.release()
            #print(f'9999999999: {int(round(time.time() * 1000))}')
        stream.stop_stream()
        stream.close()
        p.terminate()
'''
class AudioProcessThread(threading.Thread,pre_read_music_data):
    def __init__(self):
        threading.Thread.__init__(self)
        pre_read_music_data.__init__(self)
    def run(self):
        global data_time, data, audio, audio_chunk_time

        audio = np.zeros(int(((self.b * self.r) * 10)))
        audio_chunk_time = [0] * len(audio)

        while True:
            thread_condition.acquire()
            #print(f'sss: {int(round(time.time() * 1000))}')
            time_data = [data_time] * self.block_length

            #print(len(audio_chunk_time))
            del audio_chunk_time[0:self.block_length]

            audio_chunk_time.extend(time_data)
            #print(len(audio_chunk_time))

            #del self.time_list[0]

            #self.time_list.append(time_data)
            if data:

                str_data = data
                print(f'str_data: {len(str_data)}')

                wave_data = np.frombuffer(str_data,
                                          dtype=np.int16)

                #print(f'wave_data: {len(wave_data)}')
                block_size = 88
                wave_data = chunk_block(wave_data,block_size)
                wave_data = list(wave_data)

                audio = list(audio)

                #print(len(audio))
                # print(len(wave_data))

                del audio[0:self.block_length]

                audio.extend(wave_data)

                #print(len(audio))

                audio = np.array(audio, dtype=int)

            #print(audio_chunk_time,audio)

            #thread_condition.notify(1)
            thread_condition.release()
            #print(f'9999999999: {int(round(time.time() * 1000))}')

        stream.stop_stream()
        stream.close()
        p.terminate()'''

class FilterThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        #pre_read_music_data.__init__(self)
    def run(self):
        while True:
            dataTem = b''
            curent_time = 0
            thread_condition.acquire()
            #if(len(data_list) == 0):
                #thread_condition.wait()
            #dataTem = data_list.pop(0)
            #if not data:
                #thread_condition.wait()
            dataTem = audio
            #print(f'datatem: {dataTem}')
            curent_time = audio_chunk_time
            #print(f'curent_time: {curent_time}')
            #print(f'curent_time[-1]: {curent_time[-3:-1]},{curent_time[0]}')
            thread_condition.release()
            #print(len(dataTem))
            read_data.main_func(dataTem,curent_time)
            #print(f'eee: {int(round(time.time() * 1000))}')

class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self):
        self.addedData = []
        print(matplotlib.__version__)
        # The data
        self.xlim = 200
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        a = []
        b = []
        a.append(2.0)
        a.append(4.0)
        a.append(2.0)
        b.append(4.0)
        b.append(3.0)
        b.append(4.0)
        self.y = (self.n * 0.0) + 50
        # The window
        self.fig = Figure(figsize=(13,5), dpi=200)
        self.ax1 = self.fig.add_subplot(111)
        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')
        self.line1 = Line2D([], [], color='blue')
        self.line1_tail = Line2D([], [], color='red', linewidth=4)
        self.line1_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(0, 100)
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50, blit = True)
        return

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1, self.line1_tail, self.line1_head]
        for l in lines:
            l.set_data([], [])
        return

    def addData(self, value):
        self.addedData.append(value)
        return

    def zoomIn(self, value):
        bottom = self.ax1.get_ylim()[0]
        top = self.ax1.get_ylim()[1]
        bottom += value
        top -= value
        self.ax1.set_ylim(bottom,top)
        self.draw()
        return

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])
        self.line1.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 0 : self.n.size - margin ])
        self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]
        return

# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QObject):
    data_signal = pyqtSignal(float)

def dataSendLoop(addData_callbackFunc):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)
    last_time = 0
    # Simulate some data
    n = np.linspace(0, 499, 500)
    #y = 50 + 25*(np.sin(n / 8.3)) + 10*(np.sin(n / 7.5)) - 5*(np.sin(n / 1.5))
    global vibration_command
    i = 0
    show_bool = True
    while(True):
        current_time = int(round(time.time() * 1000))
        global vibration_command
        #print(f'prediction_list: {prediction_list}')
        #print(f'play_current_time: {int(round(time.time() * 1000))}')
        if prediction_list[-1] > current_time:
            #print('````````````````````')
            for ii in prediction_list:
                #print(f'play_current_time: {int(round(time.time() * 1000))}')
                if current_time < int(ii + 10) and current_time > (ii - 10) and (current_time - last_time) > 200:
                    #print('888')
                    last_time = int(round(time.time() * 1000))
                    vibration_command_prediction = True
                    # self.vvv += 1
                    # print(self.starting_point, self.prediction_num)
                    #print(f'888: {int(round(time.time() * 1000))},')
                    #print(f'{gate_send}   {max(abs(low_frequency_send))}')
                    if gate_send/max(abs(low_frequency_send)) < 1.8 and not silence and low_hight and abs(correction_time - current_time) > 15 and gate_low/gate_high > 0.5 and mean_low/mean_high > 5:
                        #print(f'false: {gate_, max(abs(low_frequency_))}')
                        vibration_command = True
                        #print(current_time, '预测鼓点:@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                    else:
                        #print(f'Ture: {gate_send, max(abs(low_frequency_send))}')
                        vibration_command = False
                    #vibration_command = True
        else:
            None
        if vibration_command:
            show_bool = True
            last_time = int(round(time.time() * 1000))
            y = np.full(500, 50 + 25 * (np.sin(n / 8.3)) + 10 * (np.sin(n / 7.5)) - 5 * (np.sin(n / 1.5)))
            #print(f'yyyyyyyyyyyy: {y}')
            vibration_command = False
            print(f'预测执行：&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            for _ in range(20):
                mySrc.data_signal.emit(y[_])  # <- Here you emit a signal!
        else:
            y = np.full(500, 50)
            if show_bool:
                show_bool = False
                for _ in range(200):
                    mySrc.data_signal.emit(y[_])  # <- Here you emit a signal!

        '''  
        else:

            y = np.full(500,50)# + 25 * (np.sin(n / 8.3)) + 10 * (np.sin(n / 7.5)) - 5 * (np.sin(n / 1.5)))
        for _ in range(50):
            mySrc.data_signal.emit(y[_])  # <- Here you emit a signal!
        if(i > 499):
            i = 0
        time.sleep(0.001)
        #if
        #mySrc.data_signal.emit(y[i]) # <- Here you emit a signal!
        i += 1'''
    ###
class CustomMainWindow(QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("drum_show")
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        # Place the zoom button
        self.zoomBtn = QPushButton(text = 'zoom')
        self.zoomBtn.setFixedSize(100, 50)
        self.zoomBtn.clicked.connect(self.zoomBtnAction)
        self.LAYOUT_A.addWidget(self.zoomBtn, *(0,0))
        # Place the matplotlib figure
        self.myFig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(0,1))
        # Add the callbackfunc to ..
        audioThread = AudioRecordThread()
        #audioprocessThread = AudioProcessThread()
        filterThread = FilterThread()
        low_hight_Thread = Low_Hight_Thread()
        #predictionThread = PredictionThread()
        # pUIThread = UIThread()
        audioThread.start()
        #audioprocessThread.start()
        filterThread.start()
        #low_hight_Thread.start()
        #predictionThread.start()
        # pUIThread.start()
        myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        myDataLoop.start()
        self.show()
        return

    def zoomBtnAction(self):
        print("zoom in")
        self.myFig.zoomIn(5)
        return

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)
        return

if __name__== '__main__':
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())
