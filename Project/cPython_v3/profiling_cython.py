from timeit import default_timer as timer
import statistics
from cpython_v3 import cpython_v3


# cpython_v3
time_mean1=[]
time_stdev1=[]

iteration = 5
resolution = [128,192,256,384,512]

for i in range(5):
    tmp=[]
    for _ in range(iteration):
        start=timer()
        res=cpython_v3(resolution[i],False)
        end=timer()
        tmp.append(end-start)
        
    time_mean1.append(statistics.mean(tmp))
    time_stdev1.append(statistics.stdev(tmp))

print("cpython_v3:")
print(time_mean1)
print(time_stdev1)

