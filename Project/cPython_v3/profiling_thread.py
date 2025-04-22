from timeit import default_timer as timer
import statistics
from cpython_v3 import cpython_v3


# cpython_v3
time_mean1=[]


thread = [1, 8, 16, 36, 72]

for i in range(5):
    start=timer()
    res=cpython_v3(512,False,thread[i])
    end=timer()
    time_mean1.append(end-start)
        

print("cpython_v3:")
print(time_mean1)

