# -*- coding:utf-8 -*-

import pyaudio
import wave
import struct
import numpy as np
import csv

def audio_init(RATE=64*64*4,fs=1):
    fr=RATE #サンプリング周波数
    p=pyaudio.PyAudio()
    fs=fs #0.25
    CHUNK=int(fr*fs) #サンプリング数
    p=pyaudio.PyAudio()
    
    stream=p.open(	format = pyaudio.paInt16,
            channels = 1,
            rate = fr,
            frames_per_buffer = CHUNK,
            input = True,
            output = True) # inputとoutputを同時にTrueにする
    return stream,CHUNK,p,fr

def read_wav(path = './fft_sound/out_sin_',sk=0):
    wavfile = path+str(sk)
    wr = wave.open(wavfile+'.wav', "rb")
    ch = wr.getnchannels()
    width = wr.getsampwidth()
    fr = wr.getframerate() #sampling freq ; RATE
    fn = wr.getnframes()  #sampling No. of frames; CHUNK
    fs = fn / fr  #sampling time
    origin = wr.readframes(fn)
    return origin

def save_wav(p,fr,input, path='./fft_sound/out_sin_',sk=0):
    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0
    sin_wave = [int(x * 32767.0) for x in sig] 
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    w = wave.Wave_write(path+str(sk)+'.wav')
    p = (1, 2, fr, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()
    return sig

# listをCSVファイルで出力
def writecsv(path,sk,output_data):
    with open(path+str(sk)+'.txt', 'w', newline='\n', encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerow(output_data)

def readcsv(path,sk):
    with open(path+str(sk)+'.txt', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        sin_wav=[]
        for row in spamreader:
            sin_wav = (float(x) for x in row)
    sin_wave = [int(float(x)* 32767.0 ) for x in sin_wav]    
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)        
    return binwave,sin_wave    

sk=0
r= 1.059463094
r12=r*r*r*r
fs=0.25 #sampling interval
fr=64*64*4 #sampling rate

stream,CHUNK,p,fr=audio_init(RATE=fr,fs=fs)
stream1,CHUNK1,p1,fr1=audio_init(RATE=int(fr*r12),fs=fs)
path='./aiueo/sig_0730/sentence_1/uwan2/out_sin_'

while 1:
    input = stream.read(CHUNK)
    #output = stream.write(input)
    
    sig = save_wav(p,fr,input,path,sk)
    
    writecsv(path,sk,sig)
    bin_wave,sin_wave = readcsv(path,sk)
    #output =stream1.write(bin_wave)
    sk += 1
    if sk > 39:
        break

skl = 40 #sk
sin_wave = []
origin_wave = read_wav(path,sk=0)
bin_wave,x_ = readcsv(path,sk=0)
sin_wave += x_
for i in range(1,skl,1):
        b,x_ = readcsv(path,sk=i)
        sin_wave += x_
        bin_wave += b
        origin= read_wav(path,sk=i)
        origin_wave += origin

for i in range(len(sin_wave)):
    sin_wave[i]=sin_wave[i]/32762.

#output=stream.write(bin_wave)    
#output=stream1.write(bin_wave)
output=stream.write(origin_wave)

import matplotlib.pyplot as plt
def plot_signal(path,t1,sig):
    plt.plot(t1, sig)
    plt.grid(True)
    #plt.xlim([0, 1])
    #plt.ylim([0, 5])
    plt.savefig(path+"plot_signal.jpg", dpi=200)
    plt.pause(1)
    plt.close()
    
fs=fs*skl
fn=CHUNK*skl
t1=np.linspace(0,fs,fn)
sig=sin_wave
plot_signal(path,t1,sig)

def plot_signal_multi(path,t1,sig,mf=2):
    m=0.5 #3 #4 #0.25 #0.5 #1 #2
    los=int(len(sig)/(mf-2))
    fig, axs = plt.subplots(mf+1)
    fig = plt.figure(figsize=(16, 8))
    axs[0]=fig.add_subplot(int(mf/4),4,1)
    axs[0].plot(t1,sig)
    axs[1] = fig.add_subplot(int(mf/4),4,1+1)
    axs[1].plot(t1[los*(1-1):los*(1-1)+int(4096*m*m)],sig[los*(1-1):los*(1-1)+int(4096*m*m)])    
    for i in range(2,mf,1):
        axs[i] = fig.add_subplot(int(mf/4),4,i+1)
        axs[i].plot(t1[los*(i-1)-int(4096*m*m):los*(i-1)],sig[los*(i-1)-int(4096*m*m):los*(i-1)])
    
    plt.pause(1)
    plt.savefig(path+'signal_multi_'+str(m)+'.jpg')
    plt.clf()

plot_signal_multi(path,t1,sig,mf=12)

def plot_picture_multi(path,t1,origin_wave,sig,mf=2):
    m=0.5 #3 #4 #0.25 #0.5 #1 #2
    print(len(sig),len(origin_wave))
    sig_origin = np.frombuffer(origin_wave, dtype="int16")  /32768.0
    loo=int(len(sig_origin)/(mf-2))
    print(len(sig),loo)
    fig, axs = plt.subplots(mf+1)
    fig = plt.figure(figsize=(16, 8))
    axs[0]=fig.add_subplot(int(mf/4),4,1)
    axs[0].plot(t1,sig)
    axs[0].set_title('original_wave(0-10 sec)')
    axs[1] = fig.add_subplot(int(mf/4),4,1+1)
    sig1=sig_origin[loo*(1-1):loo*(1-1)+int(4096*m*m)]
    sig2=sig1.reshape(int(64*m),int(64*m))
    axs[1].imshow(np.clip(sig2,0,1))
    axs[1].set_title(str(0)+'sec')
    for i in range(2,mf,1):
        axs[i] = fig.add_subplot(int(mf/4),4,i+1)
        sig1=sig_origin[loo*(i-1)-int(4096*m*m):loo*(i-1)]
        print(sig1.shape)
        if sig1.shape[0]==0:
            continue
        else:
            sig1=sig1.reshape(int(64*m),int(64*m))
            axs[i].imshow(np.clip(sig1,0,1))
            axs[i].set_title(str(i-1))
    plt.pause(1)
    plt.savefig(path+'picture_signal_'+str(m)+'.jpg')
    plt.clf()
    
origin_wave
plot_picture_multi(path,t1,origin_wave,sig,mf=12)

from scipy.fftpack import fft, ifft

def FFT(sig,fn,fr):
    freq =fft(sig,int(fn))
    Pyy = np.sqrt(freq*freq.conj())/fn
    f = np.arange(0,fr,fr/fn)
    pabs=np.abs(Pyy)
    return pabs,f

def plot_FFT(sig,mf=12):
    mf=mf
    m=0.5
    fn=4096*m*m
    fr=44100
    los=int(len(sig)/(mf-2))
    fig, axs = plt.subplots(mf+1)
    fig = plt.figure(figsize=(16, 8))
    axs[0]=fig.add_subplot(int(mf/4),4,1)
    axs[0].plot(t1,sig)
    axs[0].set_title('original_wave(0-10 sec)')
    axs[1] = fig.add_subplot(int(mf/4),4,1+1)
    sig1=sig[los*(1-1):los*(1-1)+int(4096*m*m)]
    pabs,f=FFT(sig1,fn,fr)
    axs[1].plot(f,pabs)
    axs[1].set_xlim(20,20000)
    axs[1].grid(True)
    axs[1].set_xscale('log')
    for i in range(2,mf,1):
        axs[i] = fig.add_subplot(int(mf/4),4,i+1)
        sig1=sig[los*(i-1):los*(i-1)+int(4096*m*m)]
        pabs,f=FFT(sig1,fn,fr)
        axs[i].plot(f,pabs)
        axs[i].set_xlim(20,20000)
        axs[i].grid(True)
        axs[i].set_xscale('log')
    plt.pause(1)
    plt.savefig(path+'FFT_signal_'+str(m)+'.jpg')
    plt.clf()
    
plot_FFT(sig,mf=12)

from scipy import signal

def FFT_annotation(sig,mf=12):
    mf=mf
    m=0.5
    fn=4096*m*m
    fr=44100
    freq =fft(sig,int(fn))
    Pyy = np.sqrt(freq*freq.conj())/fn
    f = np.arange(0,fr,fr/fn)
    ld = signal.argrelmax(Pyy, order=2) #相対強度の最大な番号をorder=10で求める
    ssk=0
    fsk=[]
    Psk=[]
    maxPyy=max(np.abs(Pyy))
    for i in range(len(ld[0])):  #ピークの中で以下の条件に合うピークの周波数fと強度Pyyを求める
        if np.abs(Pyy[ld[0][i]])>0.25*maxPyy and f[ld[0][i]]<20000 and f[ld[0][i]]>20:
            fssk=f[ld[0][i]]
            Pssk=np.abs(Pyy[ld[0][i]])
            fsk.append(fssk)
            Psk.append(Pssk)
            ssk += 1
    
    print('{}'.format(np.round(fsk[:len(fsk)],decimals=2))) #標準出力にピーク周波数fskを小数点以下二桁まで出力する
    print('{}'.format(np.round(Psk[:len(fsk)],decimals=4))) #標準出力にピーク強度Pskを小数点以下6桁まで出力する
                
    pabs=np.abs(Pyy)
    return pabs,f,fsk,Psk

def cyclic_annotation(sig,mf=12):
    m=0.5 #3 #4 #0.25 #0.5 #1 #2
    los=int(len(sig)/(mf-2))
    fig, axs = plt.subplots(mf+1)
    fig = plt.figure(figsize=(16, 8))
    axs[0]=fig.add_subplot(int(mf/4),4,1)
    axs[0].plot(t1,sig)
    axs[1] = fig.add_subplot(int(mf/4),4,1+1)
    
    sig1=sig[los*(1-1):los*(1-1)+int(4096*m*m)]
    pabs,f,fsk,Psk = FFT_annotation(sig1,mf)    
    axs[1].plot(f,pabs)
    axs[1].axis([min(fsk)-100, max(fsk)+100, 0,max(pabs)*1.5])  #max(Pyy)])
    axs[1].grid(True)
    axs[1].set_xscale('log')
    axs[1].set_ylim(0,max(pabs)*1.5)
    #axs[1].title('{}'.format(np.round(fsk[:len(fsk)],decimals=0))+'\n'+'{}'.format(np.round(Psk[:len(fsk)],decimals=4)),size=10)
    axs[1].plot(fsk[:len(fsk)],Psk[:len(fsk)],'ro')  #ピーク周波数、ピーク強度の位置に〇をつける
    # グラフにピークの周波数をテキストで表示
    for i in range(len(fsk)):
        axs[1].annotate('{0:.0f}(Hz)'.format(fsk[i]),  #np.round(fsk[i],decimals=2) でも可
                 xy=(fsk[i], Psk[i]),
                 xytext=(10, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
     
    for i in range(2,mf,1):
        axs[i] = fig.add_subplot(int(mf/4),4,i+1)
        sig1=sig[los*(i-1):los*(i-1)+int(4096*m*m)]
        pabs,f,fsk,Psk = FFT_annotation(sig1,mf)    
        axs[i].plot(f,pabs)
        #axs[i].axis([min(fsk)-100, max(fsk)+100, 0,max(pabs)*1.5])  #max(Pyy)])
        axs[i].set_xlim(20,20000)
        axs[i].grid(True)
        axs[i].set_xscale('log')

        #axs[i].title('{}'.format(np.round(fsk[:len(fsk)],decimals=0))+'\n'+'{}'.format(np.round(Psk[:len(fsk)],decimals=4)),size=10)  #グラフのタイトルにピーク周波数とピーク強度を出力する
        axs[i].plot(fsk[:len(fsk)],Psk[:len(fsk)],'ro')  #ピーク周波数、ピーク強度の位置に〇をつける
    # グラフにピークの周波数をテキストで表示
        for j in range(len(fsk)):
            axs[i].annotate('{0:.0f}(Hz)'.format(fsk[j]),  #np.round(fsk[i],decimals=2) でも可
                 xy=(fsk[j], Psk[j]),
                 xytext=(10, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
        
    plt.pause(1)
    plt.savefig(path+'FFT_annotation_'+str(m)+'.jpg')
    plt.clf()

cyclic_annotation(sig,mf=12)
