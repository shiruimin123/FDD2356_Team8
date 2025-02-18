import numpy as np
import pytest

def DGEMM_numpy(res,A,B):
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(B)[1]):
            for k in range(np.shape(A)[1]):
                res[i][j] += A[i][k] * B[k][j] 
    return res

@pytest.mark.parametrize('C, A, B, expected',[(np.zeros((100, 100)),np.ones((100, 100)),np.ones((100, 100)),1000000.0)])
def test_DGEMM_numpy(C,A, B,expected):
    res=DGEMM_numpy(C,A,B)
    assert np.sum(res) == expected