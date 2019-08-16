import pyaudio
import wave
import struct
import numpy as np
import csv
import matplotlib.pyplot as plt
import numpy as np

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

origin_wave,sin_origin,fr,fn,fs = read_wav(path,sk=100)
stream,CHUNK,p,fr=audio_init(RATE=fr,fs=fs)
sig=sin_origin[4096*5:4096*15]
print(len(sig))
sin_wave = [np.clip(int(float(x)* 32767),-32767,32767)  for x in sig] #32767.0   
bw = struct.pack("h" * len(sin_wave), *sin_wave)
for i in range(1):
    output=stream.write(bw)

fsk=129.19921875
pitch=349
print(fsk,pitch) #129.19921875 349
sig2=[]
j=0
ps=2 #3 #2
pn=50 #20 #50
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
    
print('len(sig2)=',len(sig2))   
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
    
print('len(sig3)=',len(sig3)) 
t3=np.linspace(0,(pn-1)/fsk,(pn-1)*pitch)
plot_signal3(path,t3,sig3,sig_name='sig3_original')
plot_signal3(path,t3[:4096],sig3[:4096],sig_name='sig3_original_')

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

#sekisan version
sig4=np.zeros(pitch)
for i in range(0,pitch):
    for j in range(0,pn-1):
        sig4[i] += sig2[j][i]/(pn-1)

sig5=np.zeros((pn-1)*pitch)
mado=np.zeros((pn-1)*pitch)
for j in range(pn-1):
    for i in range(0,pitch):
        sig5[j*pitch+i]=sig4[i]
        if i<pitch*0.5:
            mado[j*pitch+i]=2*i/pitch
        else:
            mado[j*pitch+i]=1-2*(i-pitch/2)/pitch
sig5_mado=np.zeros((pn-1)*pitch)
for j in range(pn-1):
    for i in range(0,pitch):
        sig5_mado[j*pitch+i]=sig5[j*pitch+i]*mado[j*pitch+i]
            
print('len(sig5)=',len(sig5))         
t3=np.linspace(0,(pn-1)/fsk,(pn-1)*pitch)
plot_signal3(path,t3,sig5,'sig5_sekisan')
plot_signal3(path,t3[:4096],sig5[:4096],'sig5_sekisan_')
plot_signal3(path,t3,mado,'mado_function')
plot_signal3(path,t3[:4096],mado[:4096],'mado_function_')
plot_signal3(path,t3,sig5_mado,'sig5_mado')
plot_signal3(path,t3[:4096],sig5_mado[:4096],'sig_mado_')
        
def plot_signal(path,t1,sig,sig_name='default'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t1, sig[:len(t1)])
    ax.grid(True)
    #plt.xlim([0, 1])
    #plt.ylim([0, 5])
    plt.savefig(path+"plot_signal_"+sig_name+".jpg", dpi=100)
    plt.pause(1)
    plt.close()

def interpolation(x0,y0,x1,y1,x):
    dn = (x0-x1)
    return y0*(x-x1)/dn + y1*(x0-x)/dn

def interpolation2(x0,y0,x1,y1,x2,y2,x):
    dn1 = (x0-x1)*(x0-x2)
    dn2 = (x1-x2)*(x1-x0)
    dn3 = (x2-x0)*(x2-x1)
    return y0*(x-x1)*(x-x2)/dn1+y1*(x-x2)*(x-x0)/dn2+y2*(x-x0)*(x-x1)/dn3


pitch=len(sig5)
x=np.linspace(0,fs,len(sig5))
y=sig5 #_mado
plt.plot(x[:4096],y[:4096])
plt.savefig('./aiueo/sig_0730/sentence_1/uwan2/xvsy_.jpg')
plt.pause(1)
plt.close()

m=5
sigxm=np.zeros(m*pitch-(m-1))
sigxm[0]=y[0]
sigxm[m*pitch-m]=y[pitch-1]
for i in range(1,m*pitch-m,1):
    if i%4==0:
        sigxm[i]=y[int(i/m)]
    if i > m*pitch-(2*m+1):
        sigxm[i] = interpolation(int(i/m),y[int(i/m)],int(i/m)+1,y[int(i/m)+1],int(i/m)+(i%m)/m)
    else:
        sigxm[i] = interpolation2(int(i/m),y[int(i/m)],int(i/m)+1,y[int(i/m)+1],int(i/m)+2,y[int(i/m)+2],int(i/m)+(i%m)/m)

stream1,CHUNK,p,Fr=audio_init(RATE=fr*5,fs=fs)        
sin_wave = [np.clip(int(float(x)* 32767),-32767,32767)  for x in sigxm] #32767.0   
bw = struct.pack("h" * len(sin_wave), *sin_wave)
sin_=[]
for i in range(10):
    output=stream1.write(bw)
    sin_+=sin_wave
    
print(len(sigxm))        
x4=np.linspace(0,fs,m*pitch-(m-1))

plt.plot(x4[:4096], sigxm[:4096])
plt.pause(1)
plt.savefig('./aiueo/sig_0730/sentence_1/uwan2/x4vssigxm_.jpg')
plt.close()

def save_wav(p,fr,input, path='./fft_sound/out_sin_',sk=0):
    #sig =[]
    #sig = np.frombuffer(input, dtype="int16")  /32768.0
    #sin_wave = [np.clip(int(float(x) * 32767.0),-32767,32767) for x in input] 
    sin_wave = [int(float(x) * 1) for x in input] 
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    w = wave.Wave_write(path+str(sk)+'.wav')
    p = (1, 2, fr, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()
    return sig
print(max(sin_),min(sin_))
sk=1
save_wav(p,Fr,sin_, path+'x4vssigxm_'+str(pitch)+'_'+str(m)+'_',sk)

for j in range(2,10,1):
    sk=0
    sigym=[]
    for i in range(len(sigxm)):
        if i%j==0:
            continue
        else:
            sy=sigxm[i]
            sigym.append(sy)
            sk += 1
    sin_=[]
    stream2,CHUNK,p,fr=audio_init(RATE=int(5*Fr/5),fs=fs)        
    sin_wave = [np.clip(int(float(x)* 32767),-32767,32767)  for x in sigym] #32767.0   
    bw = struct.pack("h" * len(sin_wave), *sin_wave)
    for i in range(10):
        output=stream2.write(bw)
        sin_ += sin_wave
        
    print(len(sigym))        
    x4=np.linspace(0,fs,4*(m*pitch-(m-1))/5)

    plt.plot(x4[:4096], sigym[:4096])
    plt.savefig('./aiueo/sig_0730/sentence_1/uwan2/x4vssigym_'+str(j)+'.jpg')
    plt.pause(1)
    plt.close()
    
    print(max(sin_),min(sin_))
    sk=j
    save_wav(p,fr,sin_, path+'x4vssigym_'+str(pitch)+'_'+str(m)+'_',sk)
    
for j in range(1,20,1):
    sk=0
    sigym=[]
    sigym=sigxm[0:512*j]
    """
    for i in range(len(sigxm)):
        if i>len(sigxm)*1/j:
            continue
        else:
            sy=sigxm[sk]
            sigym.append(sy)
            sk += 1
    """
    sin_=[]
    stream2,CHUNK,p,fr=audio_init(RATE=int(7*Fr/5),fs=fs)        #small up 
    sin_wave = [np.clip(int(float(x)* 32767),-32767,32767)  for x in sigym[0:1024*j]] #32767.0   
    bw = struct.pack("h" * len(sin_wave), *sin_wave)
    for i in range(int(10*20/j)):
        output=stream2.write(bw)
        sin_ += sin_wave
        
    print(len(sigym))        
    x4=np.linspace(0,fs,4*(m*pitch-(m-1))/5)

    plt.plot(x4[:512*j], sigym[:512*j])
    plt.savefig('./aiueo/sig_0730/sentence_1/uwan2/x4vssigym_c_'+str(j)+'.jpg')
    plt.pause(1)
    plt.close()
    
    print(max(sin_),min(sin_))
    sk=j
    save_wav(p,fr,sin_, path+'x4vssigym_c_'+str(pitch)+'_'+str(m)+'_',sk)
    