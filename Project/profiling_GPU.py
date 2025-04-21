from timeit import default_timer as timer
import statistics
from GPU_v import GPU_v 

# Baseline
time_mean1=[]
time_stdev1=[]

iteration = 5
resolution = [128,192,256,384,512]

for i in range(5):
    tmp=[]
    for _ in range(iteration):
        start=timer()
        res=GPU_v(resolution[i],False)
        end=timer()
        tmp.append(end-start)
        
    time_mean1.append(statistics.mean(tmp))
    time_stdev1.append(statistics.stdev(tmp))

print("Baseline:")
print(time_mean1)
print(time_stdev1)

