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
        
    stream=p.open(	format = pyaudio.paInt16,
            channels = 1,
            rate = fr,
            frames_per_buffer = CHUNK,
            input = True,
            output = True) # inputとoutputを同時にTrueにする
    return stream,CHUNK,p,fr

def read_wav(path = './fft_sound/out_sin_',sk=0):
    wavfile = path +str(sk)
    wr = wave.open(wavfile+'.wav', "rb")
    ch = wr.getnchannels()
    width = wr.getsampwidth()
    fr = wr.getframerate() #sampling freq ; RATE
    fn = wr.getnframes()  #sampling No. of frames; CHUNK
    fs = fn / fr  #sampling time
    origin = wr.readframes(fn)
    sig =[]
    sig = np.frombuffer(origin, dtype="int16")  /32768.0
    return origin,sig,fr,fn,fs

path='./aiueo/sig_0730/sentence_1/uwan2/u_p100f'
skl=1
sin_wave = []
origin_wave,sin_origin,fr,fn,fs = read_wav(path,sk=100)
stream,CHUNK,p,fr=audio_init(RATE=fr,fs=fs)

output=stream.write(origin_wave)

import matplotlib.pyplot as plt
def plot_signal(path,t1,sig,sig_name='default'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t1, sig[:len(t1)])
    ax.grid(True)
    #plt.xlim([0, 1])
    #plt.ylim([0, 5])
    plt.savefig(path+"plot_signal_"+sig_name+".jpg", dpi=100)
    plt.pause(1)
    plt.close()
    
fs=fs*skl
fn=CHUNK*skl
t1=np.linspace(0,fs,fn)
#sig=sin_wave
sig=sin_origin
plot_signal(path,t1,sig,'original')


from scipy.fftpack import fft, ifft
from scipy import signal

def FFT(sig,fn,fr):
    freq =fft(sig,int(fn))
    f = np.arange(0,fr,fr/fn)
    return freq,f
    
def iFFT(freq):
    sig=ifft(freq)
    return sig

def plot_FFT(path,sig,mf=12):
    mf=mf
    m=1  #0.5
    fn=4096*m
    fr=44100
    los=int(len(sig)/(mf-2))
    fig,axs = plt.subplots(mf+1)
    plt.close()
    fig = plt.figure(figsize=(16, 8))
    axs[0].plot(t1,sig[:len(t1)])
    axs[0].set_title('original_wave(0-10 sec)')
    sig1=sig[los*(1-1):los*(1-1)+int(4096*m)]
    Freq=[]
    freq,f=FFT(sig1,fn,fr)
    Freq.append(freq)
    Pyy = np.sqrt(freq*freq.conj())/fn
    pabs=np.abs(Pyy)
    axs[1].plot(f,pabs)
    axs[1].set_xlim(20,20000)
    #axs[1].grid(True)
    axs[1].set_xscale('log')
    for i in range(2,mf,1):
        sig1=sig[los*(i-1):los*(i-1)+int(4096*m)]
        freq,f=FFT(sig1,fn,fr)
        Freq.append(freq)
        Pyy = np.sqrt(freq*freq.conj())/fn
        pabs=np.abs(Pyy)
        axs[i].plot(f,pabs)
        axs[i].set_xlim(20,20000)
        #axs[i].grid(True)
        axs[i].set_xscale('log')
        
    plt.pause(1)
    plt.savefig(path+'FFT_signal_'+str(m)+'.jpg')
    plt.close()
    return Freq,f
    
Freq,f = plot_FFT(path,sig,mf=12)

def FFT_annotation(sig,mf=12):
    mf=mf
    m=1
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

def cyclic_annotation(path,sig,mf=12):
    m=1 #3 #4 #0.25 #0.5 #1 #2
    los=int(len(sig)/(mf-2))
    fig, axs = plt.subplots(mf+1)
    plt.close()
    fig = plt.figure(figsize=(16, 8))
    axs[0]=fig.add_subplot(int(mf/4),4,1)
    axs[0].plot(t1,sig[:len(t1)])
    axs[1] = fig.add_subplot(int(mf/4),4,1+1)
    
    sig1=sig[los*(1-1):los*(1-1)+int(4096*m*m)]
    pabs,f,fsk,Psk = FFT_annotation(sig1,mf)    
    axs[1].plot(f,pabs)
    #axs[1].axis([min(fsk)-100, max(fsk)+100, 0,max(pabs)*1.5])  #max(Pyy)])
    #axs[1].grid(True)
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
        #axs[i].grid(True)
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
    plt.savefig(path+'FFT_annotation_'+str(mf)+'_'+str(m)+'.jpg')
    plt.close()

cyclic_annotation(path,sig,mf=12)

def cyclic_annotation2(path,sig,mf=12):
    m=1 #3 #4 #0.25 #0.5 #1 #2
    Fsk=[]
    los=int(len(sig)/(mf-2))
    fig, axs = plt.subplots(mf+1)
    plt.close()
    fig = plt.figure(figsize=(16, 8))
    axs[0]=fig.add_subplot(int(mf/4),4,1)
    sig1=sig[0]
    pabs,f,fsk,Psk = FFT_annotation(sig1,mf)
    Fsk.append(fsk[0])
    axs[0].plot(f,pabs)
    axs[0].set_xlim(20,20000)
    #axs[0].grid(True)
    axs[0].set_xscale('log')
    axs[0].set_ylim(0,max(pabs)*1.5)
    #axs[1].title('{}'.format(np.round(fsk[:len(fsk)],decimals=0))+'\n'+'{}'.format(np.round(Psk[:len(fsk)],decimals=4)),size=10)
    axs[0].plot(fsk[:len(fsk)],Psk[:len(fsk)],'ro')  #ピーク周波数、ピーク強度の位置に〇をつける
    # グラフにピークの周波数をテキストで表示
    for i in range(len(fsk)):
        axs[0].annotate('{0:.0f}(Hz)'.format(fsk[i]),  #np.round(fsk[i],decimals=2) でも可
                 xy=(fsk[i], Psk[i]),
                 xytext=(10, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
    
    for i in range(1,len(sig),1):
        axs[i] = fig.add_subplot(int(mf/4),4,i+1)
        sig1=sig[i]
        pabs,f,fsk,Psk = FFT_annotation(sig1,mf)
        Fsk.append(fsk[0])
        axs[i].plot(f,pabs)
        #axs[i].axis([min(fsk)-100, max(fsk)+100, 0,max(pabs)*1.5])  #max(Pyy)])
        axs[i].set_xlim(20,20000)
        #axs[i].grid(True)
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
    plt.savefig(path+'FFT_annotation2_'+str(mf)+'_'+str(m)+'.jpg')
    plt.close()
    return Fsk

def plot_iFFT(path,Freq,sig,mf=12):
    m=1
    fs=4096*m/44100
    fn=4096*m
    mf=mf
    Sig=[]
    t=np.linspace(0,fs,fn)
    fig, axs = plt.subplots(mf+1)
    plt.close()
    fig=plt.figure(figsize=(16, 8))
    axs[0] = fig.add_subplot(int(mf/4),4,1)
    axs[0].plot(t1,sig[:len(t1)])
    axs[0].set_title('original_wave(0-10 sec)')
    sig=iFFT(Freq[0])
    Sig.append(sig)
    axs[1] = fig.add_subplot(int(mf/4),4,1+1)
    axs[1].plot(t,sig)
    for i in range(2,mf-1,1):
        axs[i] = fig.add_subplot(int(mf/4),4,i+1)
        sig=iFFT(Freq[i-1])
        axs[i].plot(t,sig)
        Sig.append(sig)
    plt.pause(1)
    plt.savefig(path+'iFFT_signal_'+str(mf)+'_'+str(m)+'.jpg')
    plt.close()
    return Sig
    
plot_iFFT(path,Freq,sig,mf=12)
Sig=plot_iFFT(path,Freq[1:12],sig,mf=9)
Fsk=cyclic_annotation2(path,Sig[1:12],mf=8)
fsk=Fsk[2]
pitch=int(44100/fsk)+8
print(fsk,pitch)
sig2=[]
j=0
ps=2
pn=50
for ssl in range(pn):
    while 1:
        sig_real=sig[j]
        if sig_real>0.3:
            sig_real=0
            break
        j += 1        
    si=sig[j:pitch*ps+j]
    sig2.append(si)
    j=j+pitch*ps

def plot_signal2(path,t1,sig):
    fig = plt.figure(figsize=(16, 8))
    for i in range(len(sig)):
        plt.plot(t1, sig[i])
        plt.grid(True)
        #plt.xlim([0, 1])
        #plt.ylim([0, 5])
    plt.savefig(path+"plot_signal_signal2.jpg", dpi=100)
    plt.pause(1)
    plt.close()
    
#t1=np.linspace(0,fs,fn)    
t2=np.linspace(0,ps/fsk,ps*pitch)
plot_signal2(path,t2,sig2)

def plot_signal3(path,t1,sig,sig_name='default'):
    fig = plt.figure(figsize=(16, 8))
    plt.plot(t1, sig)
    plt.title(sig_name,size=20)
    #plt.grid(True)
    #plt.xlim([0, 1])
    #plt.ylim([0, 5])
    plt.savefig('./aiueo/sig_0730/sentence_1/uwan2/u_p100f_'+sig_name+'.jpg')
    plt.pause(1)
    plt.close()

sig3=np.zeros((pn-1)*pitch)
for j in range(0,pn-1):
    if j==0:
        for i in range(0,pitch):
            sig3[j*pitch+i]=sig2[j][i]*2
            #print(sig2[j][i],sig3[i])
    elif j==pn-1:
        for k in range(0,(ps-1)*pitch):
            sig3[j*pitch+k]= sig2[j+1][k]*2
    else:
        for m in range(0,(ps-1)*pitch,1):
            #print(sig2[j])
            sig3[j*pitch+m] = sig2[j][pitch+m] + sig2[j+1][m]
    
    
print(sig3)   
t3=np.linspace(0,(pn-1)/fsk,(pn-1)*pitch)
plot_signal3(path,t3,sig3,sig_name='sig3_original')

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

#kurikaeshi sk_m , record_time=(pn-1)/fsk
sk=0
r= 1.059463094
r12=1 #r*r*r*r
fs=(pn-1)/fsk #0.25 #sampling interval
fr=44100*1 #64*64*4 #sampling rate

#stream,CHUNK,p,fr=audio_init(RATE=fr,fs=fs)
stream,CHUNK1,p1,fr1=audio_init(RATE=int(fr*r12),fs=fs)
sin_wave = [np.clip(int(float(x)* 32767),-32767,32767)  for x in sig3] #32767.0   
bw = struct.pack("h" * len(sin_wave), *sin_wave)        
sk=0
sk_m=20
sinwave=np.zeros((sk_m+1)*(pn-1)*pitch)
while 1:
    output=stream.write(bw)
    for i in range(0,(pn-1)*pitch):
            sinwave[sk*(pn-1)*pitch+i]=sig3[i]
    
    sk += 1
    if sk >sk_m:
        break

def save_wav(p,fr,input, path='./fft_sound/out_sin_',sk=0):
    #sig =[]
    #sig = np.frombuffer(input, dtype="int16")  /32768.0
    sin_wave = [np.clip(int(float(x) * 32767.0),-32767,32767) for x in input] 
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    w = wave.Wave_write(path+str(sk)+'.wav')
    p = (1, 2, fr, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()
    return sig

save_wav(p1,fr,sinwave, path+'iFFT_wav_'+str(pitch)+'_'+str(m)+'_',sk)

#sekisan version
sig4=np.zeros(pitch)
for i in range(0,pitch):
    for j in range(0,pn-1):
        sig4[i] += sig2[j][i]/(pn-1)

sig5=np.zeros((pn-1)*pitch)
for j in range(pn-1):
    for i in range(0,pitch):
        sig5[j*pitch+i]=sig4[i]

t3=np.linspace(0,(pn-1)/fsk,(pn-1)*pitch)
plot_signal3(path,t3,sig5,'sig5_sekisan')

sin_wave = [np.clip(int(float(x)* 32767),-32767,32767)  for x in sig5] #32767.0   
bw = struct.pack("h" * len(sin_wave), *sin_wave)        
sk=0
sk_m=20
sinwave=np.zeros((sk_m+1)*(pn-1)*pitch)
while 1:
    output=stream.write(bw)
    for i in range(0,(pn-1)*pitch):
            sinwave[sk*(pn-1)*pitch+i]=sig5[i]
    sk += 1
    if sk >sk_m:
        break
        
save_wav(p1,fr,sinwave, path+'iFFT_wav_sig5_'+str(pitch)+'_'+str(m)+'_',sk) 

#interpolation of data
sig6=np.zeros(2*pitch)
for i in range(0,2*pitch,1):
    if i%2==0:
        sig6[i]=sig4[int(i/2)]
    else:
        sig6[i] = (sig4[int(i/2)-1]+sig4[int(i/2)])/2

sig7=np.zeros(2*(pn-1)*pitch)
for j in range(pn-1):
    for i in range(0,2*pitch):
        sig7[j*2*pitch+i]=sig6[i]

t3=np.linspace(0,(pn-1)/fsk,(pn-1)*2*pitch)
plot_signal3(path,t3,sig7,'sig7_interpolation')

sin_wave = [np.clip(int(float(x)* 32767),-32767,32767)  for x in sig7] #32767.0   
bw = struct.pack("h" * len(sin_wave), *sin_wave)        
sk=0
sk_m=10
sinwave=np.zeros((sk_m+1)*(pn-1)*2*pitch)
stream2,CHUNK2,p2,fr2=audio_init(RATE=int(fr*r12*2),fs=fs)
while 1:
    output=stream2.write(bw)
    for i in range(0,(pn-1)*2*pitch):
            sinwave[sk*(pn-1)*pitch+i]=sig7[i]
    sk += 1
    if sk >sk_m:
        break
        
save_wav(p2,fr2,sinwave, path+'iFFT_wav_sig7_'+str(pitch)+'_'+str(m)+'_',sk)                    
    