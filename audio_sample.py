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
    if sk > 40:
        break

skl = sk
sin_wave = []
origin_wave = read_wav(path,sk=0)
for i in range(0,skl,1):
        b,x_ = readcsv(path,sk=i)
        sin_wave += x_
        bin_wave += b
        origin= read_wav(path,sk=i)
        origin_wave += origin

for i in range(len(sin_wave)):
    sin_wave[i]=sin_wave[i]/32762.

output=stream.write(bin_wave)    
output=stream1.write(bin_wave)
output=stream.write(origin_wave)
    
    
