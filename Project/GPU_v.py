import torch
import matplotlib.pyplot as plt
import time

def getConserved( rho, vx, vy, P, gamma, vol ):
    """
    Calculate the conserved variable from the primitive
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	P        is matrix of cell pressures
	gamma    is ideal gas gamma
	vol      is cell volume
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	"""
    Mass   = rho * vol
    Momx   = rho * vx * vol
    Momy   = rho * vy * vol
    Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol
	
    return Mass, Momx, Momy, Energy


def getPrimitive( Mass, Momx, Momy, Energy, gamma, vol ):
	"""
    Calculate the primitive variable from the conservative
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	gamma    is ideal gas gamma
	vol      is cell volume
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	P        is matrix of cell pressures
	"""
	rho = Mass / vol
	vx  = Momx / rho / vol
	vy  = Momy / rho / vol
	P   = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
	
	return rho, vx, vy, P


def getGradient(f, dx):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = ( torch.roll(f,R,dims=0) - torch.roll(f,L,dims=0) ) / (2*dx)
	f_dy = ( torch.roll(f,R,dims=1) - torch.roll(f,L,dims=1) ) / (2*dx)
	
	return f_dx, f_dy


def slopeLimit(f, dx, f_dx, f_dy):
	"""
    Apply slope limiter to slopes
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = torch.maximum(0., torch.minimum(1., ( (f-torch.roll(f,L,dims=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dx = torch.maximum(0., torch.minimum(1., (-(f-torch.roll(f,R,dims=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dy = torch.maximum(0., torch.minimum(1., ( (f-torch.roll(f,L,dims=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	f_dy = torch.maximum(0., torch.minimum(1., (-(f-torch.roll(f,R,dims=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	
	return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	f_dx     is a matrix of the field x-derivatives
	f_dy     is a matrix of the field y-derivatives
	dx       is the cell size
	f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis 
	f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis 
	f_YL     is a matrix of spatial-extrapolated values on `left' face along y-axis 
	f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis 
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_XL = f - f_dx * dx/2
	f_XL = torch.roll(f_XL,R,dims=0)
	f_XR = f + f_dx * dx/2
	
	f_YL = f - f_dy * dx/2
	f_YL = torch.roll(f_YL,R,dims=1)
	f_YR = f + f_dy * dx/2
	
	return f_XL, f_XR, f_YL, f_YR
	

def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
	"""
    Apply fluxes to conserved variables
	F        is a matrix of the conserved variable field
	flux_F_X is a matrix of the x-dir fluxes
	flux_F_Y is a matrix of the y-dir fluxes
	dx       is the cell size
	dt       is the timestep
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	# update solution
	F += - dt * dx * flux_F_X
	F +=   dt * dx * torch.roll(flux_F_X,L,dims=0)
	F += - dt * dx * flux_F_Y
	F +=   dt * dx * torch.roll(flux_F_Y,L,dims=1)
	
	return F


#@profile
def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
	"""
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
	rho_L        is a matrix of left-state  density
	rho_R        is a matrix of right-state density
	vx_L         is a matrix of left-state  x-velocity
	vx_R         is a matrix of right-state x-velocity
	vy_L         is a matrix of left-state  y-velocity
	vy_R         is a matrix of right-state y-velocity
	P_L          is a matrix of left-state  pressure
	P_R          is a matrix of right-state pressure
	gamma        is the ideal gas gamma
	flux_Mass    is the matrix of mass fluxes
	flux_Momx    is the matrix of x-momentum fluxes
	flux_Momy    is the matrix of y-momentum fluxes
	flux_Energy  is the matrix of energy fluxes
	"""
	
	# left and right energies
	en_L = P_L/(gamma-1)+0.5*rho_L * (vx_L**2+vy_L**2)
	en_R = P_R/(gamma-1)+0.5*rho_R * (vx_R**2+vy_R**2)

	# compute star (averaged) states
	rho_star  = 0.5*(rho_L + rho_R)
	momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
	momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
	en_star   = 0.5*(en_L + en_R)
	
	P_star = (gamma-1)*(en_star-0.5*(momx_star**2+momy_star**2)/rho_star)
	
	# compute fluxes (local Lax-Friedrichs/Rusanov)
	flux_Mass   = momx_star
	flux_Momx   = momx_star**2/rho_star + P_star
	flux_Momy   = momx_star * momy_star/rho_star
	flux_Energy = (en_star+P_star) * momx_star/rho_star
	
	# find wavespeeds
	C_L = torch.sqrt(gamma*P_L/rho_L) + torch.abs(vx_L)
	C_R = torch.sqrt(gamma*P_R/rho_R) + torch.abs(vx_R)
	C = torch.maximum( C_L, C_R )
	
	# add stabilizing diffusive term

	
	flux_Mass   -= C * 0.5 * (rho_L - rho_R)
	flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
	flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
	flux_Energy -= C * 0.5 * ( en_L - en_R )

	return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def GPU_v(N=256, whether_plot=True, device='cuda'):
	""" Finite Volume simulation """
    
    # check GPU CUDA available
	if not torch.cuda.is_available():
		print("CUDA not available, falling back to CPU")
		device = 'cpu'
	
	# Simulation parameters
	# N                      = 256 # resolution
	boxsize                = 1.
	gamma                  = 5/3 # ideal gas gamma
	courant_fac            = 0.4
	t                      = 0
	tEnd                   = 2
	tOut                   = 0.02 # draw frequency
	useSlopeLimiting       = False
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Mesh
	dx = boxsize / N
	vol = dx**2
	xlin = torch.linspace(0.5*dx, boxsize-0.5*dx, N)
	Y, X = torch.meshgrid( xlin, xlin,indexing='xy' )
	
	# Generate Initial Conditions - opposite moving streams with perturbation
	w0 = 0.1
	sigma = 0.05/torch.sqrt(torch.tensor(2.0))
	rho = 1. + (torch.abs(Y-0.5) < 0.25)
	vx = -0.5 + (torch.abs(Y-0.5)<0.25)
	vy = w0*torch.sin(4*torch.pi*X) * ( torch.exp(-(Y-0.25)**2/(2 * sigma**2)) + torch.exp(-(Y-0.75)**2/(2*sigma**2)) )
	P = 2.5 * torch.ones(X.shape)

	# Get conserved variables
	Mass, Momx, Momy, Energy = getConserved( rho, vx, vy, P, gamma, vol )
	
	outputCount = 1

	fig = plt.figure(figsize=(4,4), dpi=80)
		

	# Simulation Main Loop
	while t < tEnd:
		# get Primitive variables
		rho, vx, vy, P = getPrimitive( Mass, Momx, Momy, Energy, gamma, vol )
		
		# get time step (CFL) = dx / max signal speed
		dt = courant_fac * torch.min( dx / (torch.sqrt( gamma*P/rho ) + torch.sqrt(vx**2+vy**2)) )
		plotThisTurn = False
		if t + dt > outputCount*tOut:
			dt = outputCount*tOut - t
			plotThisTurn = True
		
		# calculate gradients
		rho_dx, rho_dy = getGradient(rho, dx)
		vx_dx,  vx_dy  = getGradient(vx,  dx)
		vy_dx,  vy_dy  = getGradient(vy,  dx)
		P_dx,   P_dy   = getGradient(P,   dx)
		
		# slope limit gradients
		if useSlopeLimiting:
			rho_dx, rho_dy = slopeLimit(rho, dx, rho_dx, rho_dy)
			vx_dx,  vx_dy  = slopeLimit(vx , dx, vx_dx,  vx_dy )
			vy_dx,  vy_dy  = slopeLimit(vy , dx, vy_dx,  vy_dy )
			P_dx,   P_dy   = slopeLimit(P  , dx, P_dx,   P_dy  )
		
		# extrapolate half-step in time
		rho_prime = rho - 0.5*dt * ( vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
		vx_prime  = vx  - 0.5*dt * ( vx * vx_dx + vy * vx_dy + (1/rho) * P_dx )
		vy_prime  = vy  - 0.5*dt * ( vx * vy_dx + vy * vy_dy + (1/rho) * P_dy )
		P_prime   = P   - 0.5*dt * ( gamma*P * (vx_dx + vy_dy)  + vx * P_dx + vy * P_dy )
		
		# extrapolate in space to face centers
		rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy, dx)
		vx_XL,  vx_XR,  vx_YL,  vx_YR  = extrapolateInSpaceToFace(vx_prime,  vx_dx,  vx_dy,  dx)
		vy_XL,  vy_XR,  vy_YL,  vy_YR  = extrapolateInSpaceToFace(vy_prime,  vy_dx,  vy_dy,  dx)
		P_XL,   P_XR,   P_YL,   P_YR   = extrapolateInSpaceToFace(P_prime,   P_dx,   P_dy,   dx)
		
		# compute fluxes (local Lax-Friedrichs/Rusanov)
		flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma)
		flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFlux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma)
		
		# update solution
		Mass   = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
		Momx   = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
		Momy   = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
		Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
		
		# update time
		t += dt
		
		# plot in real time
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
    GPU_v(128, True)
    end = time.time()
    print("The execution time: ",(end-start))