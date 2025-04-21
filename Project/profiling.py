from timeit import default_timer as timer
import statistics
from baseline import baseline 
from cPython.cpython_v import cpython_v
from cPython_v2.cpython_v2 import cpython_v2

# Baseline
time_mean1=[]
time_stdev1=[]

iteration = 5
resolution = [128,192,256,384,512]

for i in range(5):
    tmp=[]
    for _ in range(iteration):
        start=timer()
        res=baseline(resolution[i],False)
        end=timer()
        tmp.append(end-start)
        
    time_mean1.append(statistics.mean(tmp))
    time_stdev1.append(statistics.stdev(tmp))

print("Baseline:")
print(time_mean1)
print(time_stdev1)

# cpython_v
time_mean2=[]
time_stdev2=[]

iteration = 5
resolution = [128,192,256,384,512]

for i in range(5):
    tmp=[]
    for _ in range(iteration):
        start=timer()
        res=cpython_v(resolution[i],False)
        end=timer()
        tmp.append(end-start)
        
    time_mean2.append(statistics.mean(tmp))
    time_stdev2.append(statistics.stdev(tmp))

print("cPython_v1:")
print(time_mean2)
print(time_stdev2)

# cpython_v2
time_mean3=[]
time_stdev3=[]

iteration = 5
resolution = [128,192,256,384,512]

for i in range(5):
    tmp=[]
    for _ in range(iteration):
        start=timer()
        res=cpython_v2(resolution[i],False)
        end=timer()
        tmp.append(end-start)
        
    time_mean3.append(statistics.mean(tmp))
    time_stdev3.append(statistics.stdev(tmp))

print("cPython_v1:")
print(time_mean3)
print(time_stdev3)
