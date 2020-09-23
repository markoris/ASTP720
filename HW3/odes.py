import numpy as np

def euler(func, times, initial_guess, dt=0.1):

	n_odes = initial_guess.shape[0] # number of ODEs is given by however many initial conditions we have
	t_evals = int((times[-1] - times[0])/dt) # evaluation points are determined by range of time divided by the timestep
    
	x, y = times[0], initial_guess

	for time in range(t_evals):

		yprime = func(x, y) # func is the derivative
		
		for ode in range(n_odes):
			y[ode] += dt*yprime[ode] # update y to be the time step multiplied by the derivative
			try:
				yvec = np.append(yvec, y[ode]) # store updated ys
			except NameError:
				yvec = y[ode]

		try:
			tvec = np.append(tvec, x) # store evaluation points
		except NameError:
			tvec = x
	
		x += dt # increment by time step
            
	return tvec, yvec

def heun(func, times, initial_guess, dt=0.1):

	n_odes = initial_guess.shape[0] # number of ODEs is given by however many initial conditions we have
	t_evals = int((times[-1] - times[0])/dt) # evaluation points are determined by range of time divided by the timestep
    
	x, y = times[0], initial_guess 

	for time in range(t_evals):

		yprime1 = func(x, y) # func is the derivative, here calculated at the point t_i, y_i

		k = yprime1*dt # iterate once to improve the guess

		ypred = y + k

		yprime2 = func(x+dt, ypred) # same func, but now evaluating at t_i+1, y_i+1
		
		for ode in range(n_odes):
			y[ode] += (dt/2)*yprime1[ode] + (dt/2)*yprime2[ode] # averaging slopes to get a better prediction than Euler's method
			try:
				yvec = np.append(yvec, y[ode]) # store updated ys
			except NameError:
				yvec = y[ode]
		
		try:
			tvec = np.append(tvec, x) # store evaluation points
		except NameError:
			tvec = x

		x += dt # increment by time step
            
	return tvec, yvec

def rk4(func, times, initial_guess, dt=0.1):

	n_odes = initial_guess.shape[0] # number of ODEs is given by however many initial conditions we have
	t_evals = int((times[-1] - times[0])/dt) # evaluation points are determined by range of time divided by the timestep
    
	x, y = times[0], initial_guess 

	for time in range(t_evals):

		k1 = func(x, y) # func is the derivative, here calculated at the point t_i, y_i

		yprime1 = y + k1*(dt/2)

		k2 = func(x+dt/2, yprime1) # re-evaluate func, iterating over y for better accuracy

		yprime2 = y + k2*(dt/2)

		k3 = func(x+dt/2, yprime2) # repeat with updated y

		yprime3 = y + k3*dt

		k4 = func(x+dt, yprime3) # final (fourth, hence fourth-order) evaluation at t_i+dt
		
		for ode in range(n_odes):

			y[ode] += dt/6*(k1[ode] + 2*k2[ode] + 2*k3[ode] + k4[ode]) # 1/6 normalization constant, middle terms * 2 since considered twice
			try:
				yvec = np.append(yvec, y[ode]) # store updated ys
			except NameError:
				yvec = y[ode]
		
		try:
			tvec = np.append(tvec, x) # store evaluation points
		except NameError:
			tvec = x
		
		x += dt # increment by time step
            
	return tvec, yvec
