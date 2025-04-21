import torch
import matplotlib.pyplot as plt
import time

def getConserved(rho, vx, vy, P, gamma, vol):

    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol
    return Mass, Momx, Momy, Energy

def getPrimitive(Mass, Momx, Momy, Energy, gamma, vol):

    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
    return rho, vx, vy, P

def getGradient(f, dx):

    # torch.roll
    f_dx = (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2*dx)
    f_dy = (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2*dx)
    return f_dx, f_dy

def slopeLimit(f, dx, f_dx, f_dy):

    # torch.clamp instred of np.maximum,np.minimum
    f_dx = torch.clamp((f - torch.roll(f, 1, dims=0))/dx / (f_dx + 1e-8*(f_dx==0)), 0, 1) * f_dx
    f_dx = torch.clamp(-(f - torch.roll(f, -1, dims=0))/dx / (f_dx + 1e-8*(f_dx==0)), 0, 1) * f_dx
    f_dy = torch.clamp((f - torch.roll(f, 1, dims=1))/dx / (f_dy + 1e-8*(f_dy==0)), 0, 1) * f_dy
    f_dy = torch.clamp(-(f - torch.roll(f, -1, dims=1))/dx / (f_dy + 1e-8*(f_dy==0)), 0, 1) * f_dy
    return f_dx, f_dy

def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):

    f_XL = f - f_dx * dx/2
    f_XL = torch.roll(f_XL, -1, dims=0)
    f_XR = f + f_dx * dx/2
    
    f_YL = f - f_dy * dx/2
    f_YL = torch.roll(f_YL, -1, dims=1)
    f_YR = f + f_dy * dx/2
    
    return f_XL, f_XR, f_YL, f_YR

def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):

    F = F - dt * dx * flux_F_X
    F = F + dt * dx * torch.roll(flux_F_X, 1, dims=0)
    F = F - dt * dx * flux_F_Y
    F = F + dt * dx * torch.roll(flux_F_Y, 1, dims=1)
    return F

def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):

    en_L = P_L/(gamma-1) + 0.5*rho_L * (vx_L**2 + vy_L**2)
    en_R = P_R/(gamma-1) + 0.5*rho_R * (vx_R**2 + vy_R**2)

    rho_star = 0.5*(rho_L + rho_R)
    momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5*(en_L + en_R)
    
    P_star = (gamma-1)*(en_star - 0.5*(momx_star**2 + momy_star**2)/rho_star)
    
    flux_Mass = momx_star
    flux_Momx = momx_star**2/rho_star + P_star
    flux_Momy = momx_star * momy_star/rho_star
    flux_Energy = (en_star + P_star) * momx_star/rho_star
    
    C_L = torch.sqrt(gamma*P_L/rho_L) + torch.abs(vx_L)
    C_R = torch.sqrt(gamma*P_R/rho_R) + torch.abs(vx_R)
    C = torch.maximum(C_L, C_R)
    
    flux_Mass = flux_Mass - C * 0.5 * (rho_L - rho_R)
    flux_Momx = flux_Momx - C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy = flux_Momy - C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy = flux_Energy - C * 0.5 * (en_L - en_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def baseline_gpu(N=256, whether_plot=True, device='cuda'):
    
    # check GPU CUDA available
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    
    boxsize = 1.0
    gamma = 5/3
    courant_fac = 0.4
    t = 0
    tEnd = 2
    tOut = 0.02
    useSlopeLimiting = False
    plotRealTime = True
    
    
    dx = boxsize / N
    vol = dx**2
    xlin = torch.linspace(0.5*dx, boxsize-0.5*dx, N, device=device)
    Y, X = torch.meshgrid(xlin, xlin, indexing='ij')
    
    
    w0 = 0.1
    sigma = 0.05/torch.sqrt(torch.tensor(2.0, device=device))
    rho = 1.0 + (torch.abs(Y-0.5) < 0.25).float()
    vx = -0.5 + (torch.abs(Y-0.5) < 0.25).float()
    vy = w0 * torch.sin(4*torch.pi*X) * (torch.exp(-(Y-0.25)**2/(2*sigma**2)) + torch.exp(-(Y-0.75)**2/(2*sigma**2)))
    P = 2.5 * torch.ones_like(X)
    
 
    Mass, Momx, Momy, Energy = getConserved(rho, vx, vy, P, gamma, vol)
    


    fig = plt.figure(figsize=(4,4), dpi=80)
    
    outputCount = 1
    
    while t < tEnd:
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, gamma, vol)
        
        dt = courant_fac * torch.min(dx / (torch.sqrt(gamma*P/rho) + torch.sqrt(vx**2 + vy**2)))
        plotThisTurn = False
        if t + dt > outputCount*tOut:
            dt = outputCount*tOut - t
            plotThisTurn = True
        

        rho_dx, rho_dy = getGradient(rho, dx)
        vx_dx, vx_dy = getGradient(vx, dx)
        vy_dx, vy_dy = getGradient(vy, dx)
        P_dx, P_dy = getGradient(P, dx)
        

        if useSlopeLimiting:
            rho_dx, rho_dy = slopeLimit(rho, dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slopeLimit(vx, dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slopeLimit(vy, dx, vy_dx, vy_dy)
            P_dx, P_dy = slopeLimit(P, dx, P_dx, P_dy)
        

        rho_prime = rho - 0.5*dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_prime = vx - 0.5*dt * (vx * vx_dx + vy * vx_dy + (1/rho) * P_dx)
        vy_prime = vy - 0.5*dt * (vx * vy_dx + vy * vy_dy + (1/rho) * P_dy)
        P_prime = P - 0.5*dt * (gamma*P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)
        

        rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy, dx)
        vx_XL, vx_XR, vx_YL, vx_YR = extrapolateInSpaceToFace(vx_prime, vx_dx, vx_dy, dx)
        vy_XL, vy_XR, vy_YL, vy_YR = extrapolateInSpaceToFace(vy_prime, vy_dx, vy_dy, dx)
        P_XL, P_XR, P_YL, P_YR = extrapolateInSpaceToFace(P_prime, P_dx, P_dy, dx)

        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma)
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFlux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma)
        

        Mass = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        Momx = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        Momy = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
        Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
        

        t += dt
        if (plotRealTime and plotThisTurn) or (t >= tEnd):
            plt.cla()
            plt.imshow(rho.T)
            plt.clim(0.8, 2.2)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)	
            ax.set_aspect('equal')	
            plt.pause(0.001)
            outputCount += 1
			
    plt.savefig('finitevolume.png',dpi=240)
    plt.close()
	# Save figure
    if whether_plot == True:
        plt.show()
	    
    
    return 0

if __name__ == "__main__":
    start = time.time()
    baseline_gpu(128, False)
    end = time.time()
    print("The execution time: ",(end-start))