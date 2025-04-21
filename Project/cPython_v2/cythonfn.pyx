import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fabs

def extrapolateInSpaceToFace_cython(double[:, :] f, 
                                  double[:, :] f_dx, 
                                  double[:, :] f_dy, 
                                  double dx):
 
    cdef int i, j
    cdef int nx = f.shape[0]
    cdef int ny = f.shape[1]
    
    # Initialize output arrays
    cdef np.ndarray[np.double_t, ndim=2] f_XL = np.empty((nx, ny), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] f_XR = np.empty((nx, ny), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] f_YL = np.empty((nx, ny), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] f_YR = np.empty((nx, ny), dtype=np.double)
    
    # Calculate face values
    for i in range(nx):
        for j in range(ny):
            f_XR[i,j] = f[i,j] + f_dx[i,j] * dx/2
            f_XL[i,j] = f[i,j] - f_dx[i,j] * dx/2
            f_YR[i,j] = f[i,j] + f_dy[i,j] * dx/2
            f_YL[i,j] = f[i,j] - f_dy[i,j] * dx/2
    
    # Roll arrays for shifted values (equivalent to np.roll)
    # For f_XL (roll right along axis 0)
    cdef np.ndarray[np.double_t, ndim=2] f_XL_rolled = np.empty((nx, ny), dtype=np.double)
    for i in range(nx):
        for j in range(ny):
            if i == nx-1:
                f_XL_rolled[0,j] = f_XL[i,j]  # Wrap around
            else:
                f_XL_rolled[i+1,j] = f_XL[i,j]
    
    # For f_YL (roll right along axis 1)
    cdef np.ndarray[np.double_t, ndim=2] f_YL_rolled = np.empty((nx, ny), dtype=np.double)
    for i in range(nx):
        for j in range(ny):
            if j == ny-1:
                f_YL_rolled[i,0] = f_YL[i,j]  # Wrap around
            else:
                f_YL_rolled[i,j+1] = f_YL[i,j]
    
    return f_XL_rolled, f_XR, f_YL_rolled, f_YR

def getFlux(double[:, :] rho_L, double[:, :] rho_R,
                  double[:, :] vx_L, double[:, :] vx_R,
                  double[:, :] vy_L, double[:, :] vy_R,
                  double[:, :] P_L, double[:, :] P_R,
                  double gamma):
    """
    Cython-optimized version of getFlux using memoryviews
    """
    cdef int i, j
    cdef int nx = rho_L.shape[0]
    cdef int ny = rho_L.shape[1]
    
    # Initialize output arrays
    cdef np.ndarray[np.double_t, ndim=2] flux_Mass = np.empty((nx, ny), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] flux_Momx = np.empty((nx, ny), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] flux_Momy = np.empty((nx, ny), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] flux_Energy = np.empty((nx, ny), dtype=np.double)
    
    cdef double en_L, en_R, rho_star, momx_star, momy_star, en_star, P_star
    cdef double C_L, C_R, C
    
    for i in range(nx):
        for j in range(ny):
            # left and right energies
            en_L = P_L[i,j]/(gamma-1) + 0.5*rho_L[i,j] * (vx_L[i,j]**2 + vy_L[i,j]**2)
            en_R = P_R[i,j]/(gamma-1) + 0.5*rho_R[i,j] * (vx_R[i,j]**2 + vy_R[i,j]**2)

            # compute star (averaged) states
            rho_star  = 0.5*(rho_L[i,j] + rho_R[i,j])
            momx_star = 0.5*(rho_L[i,j] * vx_L[i,j] + rho_R[i,j] * vx_R[i,j])
            momy_star = 0.5*(rho_L[i,j] * vy_L[i,j] + rho_R[i,j] * vy_R[i,j])
            en_star   = 0.5*(en_L + en_R)
            
            P_star = (gamma-1)*(en_star-0.5*(momx_star**2+momy_star**2)/rho_star)
            
            # compute fluxes (local Lax-Friedrichs/Rusanov)
            flux_Mass[i,j] = momx_star
            flux_Momx[i,j] = momx_star**2/rho_star + P_star
            flux_Momy[i,j] = momx_star * momy_star/rho_star
            flux_Energy[i,j] = (en_star+P_star) * momx_star/rho_star
            
            # find wavespeeds
            C_L = sqrt(gamma*P_L[i,j]/rho_L[i,j]) + fabs(vx_L[i,j])
            C_R = sqrt(gamma*P_R[i,j]/rho_R[i,j]) + fabs(vx_R[i,j])
            C = C_L if C_L > C_R else C_R
            
            # add stabilizing diffusive term
            flux_Mass[i,j] -= C * 0.5 * (rho_L[i,j] - rho_R[i,j])
            flux_Momx[i,j] -= C * 0.5 * (rho_L[i,j] * vx_L[i,j] - rho_R[i,j] * vx_R[i,j])
            flux_Momy[i,j] -= C * 0.5 * (rho_L[i,j] * vy_L[i,j] - rho_R[i,j] * vy_R[i,j])
            flux_Energy[i,j] -= C * 0.5 * (en_L - en_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy