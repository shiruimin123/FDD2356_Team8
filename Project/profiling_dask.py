from timeit import default_timer as timer
import statistics
from dask_v import dask_v


# cpython_v3
time_mean1=[]


resolution = [128,192,256,384,512]

for i in range(5):
    start=timer()
    res=dask_v(resolution[i],False,resolution[i])
    end=timer()
    time_mean1.append(end-start)
        
print("dask_v:")
print(time_mean1)

