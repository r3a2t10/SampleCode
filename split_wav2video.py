from os import listdir
from os.path import isfile, isdir, join
from pydub import AudioSegment
import subprocess

def get_length(name):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", name],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT)
    return float(result.stdout)

audio = AudioSegment.from_file("source/obama.wav")
#print(audio.duration_seconds)

mypath = "./source/instagram_video"
files = sorted(listdir(mypath))
time_temp = 0
last_time = 0

for f in files:
  fullpath = join(mypath, f)
  if isfile(fullpath) and fullpath.endswith(".mp4"):
    print(fullpath)
    wav_name = "./source/instagram_wav/" + fullpath.split('/')[-1].split('.')[0] + ".wav"
    last_time = time_temp
    time_temp = time_temp + get_length(fullpath)
    print(last_time,time_temp)
    if time_temp > audio.duration_seconds:
        print(time_temp,">",audio.duration_seconds)
        time_temp = time_temp - audio.duration_seconds
        t1 = last_time * 1000 #Works in milliseconds
        t2 = audio.duration_seconds * 1000
        t3 = 0 * 1000
        t4 = (time_temp - audio.duration_seconds) * 1000
        newAudio = AudioSegment.from_wav("source/obama.wav")
        newAudio = newAudio[t1:t2] + newAudio[t3:t4]
        print(newAudio)
        newAudio.export(wav_name, format="wav") #Exports to a wav file in the current path.

    else:
        t1 = last_time * 1000 #Works in milliseconds
        t2 = time_temp * 1000
        newAudio = AudioSegment.from_wav("source/obama.wav")
        newAudio = newAudio[t1:t2]
        print(newAudio)
        newAudio.export(wav_name, format="wav") #Exports to a wav file in the current path.