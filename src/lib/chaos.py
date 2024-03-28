import numpy as np


class Globally_coupled_limit_cycle_oscillators():
    def __init__(self, c1, c2, K):
        self.c1 = c1
        self.c2 = c2
        self.K = K

    
    def f(self, t, w):
        w_mean = np.mean(w)
        return w - (1 + self.c2*1j)*np.abs(w)**2*w + self.K*(1 + self.c1*1j)*(w_mean - w)
    

    def gen_time_series(self, x0, tspan, dt):
        x = x0
        X = []
        for t in tspan:
            x = rk4(self.f, t, x, dt)
            X.append(x)

        self.time_serie = np.array(X)


    def _jac(self, X, Y):
        N = len(X)
        dXdX = np.diag(1 - 3*X**2 + 2*self.c2*X*Y - Y**2 - self.K) + self.K/N
        dXdY = np.diag(self.c2*X**2 - 2*X*Y + 3*self.c2*Y**2 + self.K*self.c1) - self.K*self.c1/N
        dYdX = np.diag(-3*self.c2*X**2 - 2*X*Y - self.c2*Y**2 - self.K*self.c1) + self.K*self.c1/N
        dYdY = np.diag(1 - X**2 - 2*self.c2*X*Y - 3*Y**2 - self.K) + self.K/N
        return np.block([[dXdX, dXdY], [dYdX, dYdY]])


    def shimada_nagashima(self, W, dt, qr_step=1):
        Q = np.eye(len(W[0])*2)
        r = 0
        X = W.real
        Y = W.imag
        
        for i, (x, y) in enumerate(zip(X, Y)):
            J = self._jac(x, y)
            Q += np.dot(J, Q)*dt
            # k1 = np.dot(J, Q)
            # k2 = np.dot(J, Q + 0.5*dt*k1)
            # k3 = np.dot(J, Q + 0.5*dt*k2)
            # k4 = np.dot(J, Q + dt*k3)
            # Q = Q + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            if i%qr_step == 0:
                Q, R = np.linalg.qr(Q)        
                r += np.log(np.abs(np.diag(R)))
        
        self.lyapunov_exponents = r/((i+1)*dt)


class Locally_coupled_limit_cycle_oscillators():
    def __init__(self, c1, c2, K):
        self.c1 = c1
        self.c2 = c2
        self.K = K

    
    def f(self, t, w):
        return w - (1 + self.c2*1j)*np.abs(w)**2*w + self.K*(1 + self.c1*1j)*(np.roll(w, 1) + np.roll(w, -1) - 2*w)
    

    def gen_time_series(self, x0, tspan, dt):
        x = x0
        X = []
        for t in tspan:
            x = rk4(self.f, t, x, dt)
            X.append(x)

        self.time_serie = np.array(X)


    def _jac(self, X, Y):
        N = len(X)
        dXdX = np.diag(1 - 3*X**2 + 2*self.c2*X*Y - Y**2 - 2*self.K) + self.K*(np.roll(np.eye(N), 1, axis=0) + np.roll(np.eye(N), -1, axis=0))
        dXdY = np.diag(self.c2*X**2 - 2*X*Y + 3*self.c2*Y**2 + 2*self.K*self.c1) - self.K*self.c1*(np.roll(np.eye(N), 1, axis=0) + np.roll(np.eye(N), -1, axis=0))
        dYdX = np.diag(-3*self.c2*X**2 - 2*X*Y - self.c2*Y**2 - 2*self.K*self.c1) + self.K*self.c1*(np.roll(np.eye(N), 1, axis=0) + np.roll(np.eye(N), -1, axis=0))
        dYdY = np.diag(1 - X**2 - 2*self.c2*X*Y - 3*Y**2 - 2*self.K) + self.K*(np.roll(np.eye(N), 1, axis=0) + np.roll(np.eye(N), -1, axis=0))
        return np.block([[dXdX, dXdY], [dYdX, dYdY]])


    def shimada_nagashima(self, W, dt, qr_step=1):
        Q = np.eye(len(W[0])*2)
        r = 0
        X = W.real
        Y = W.imag
        
        for i, (x, y) in enumerate(zip(X, Y)):
            J = self._jac(x, y)
            Q += np.dot(J, Q)*dt
            # k1 = np.dot(J, Q)
            # k2 = np.dot(J, Q + 0.5*dt*k1)
            # k3 = np.dot(J, Q + 0.5*dt*k2)
            # k4 = np.dot(J, Q + dt*k3)
            # Q = Q + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            if i%qr_step == 0:
                Q, R = np.linalg.qr(Q)        
                r += np.log(np.abs(np.diag(R)))
        
        self.lyapunov_exponents = r/((i+1)*dt)


class N_rossler():
    def __init__(self, N, a, b, d, eps):
        self.N = N
        self.a = a
        self.b = b
        self.d = d
        self.eps = eps


    def eq(self, t, X):
        dXdt = np.hstack([
        -X[1] + self.a*X[0],
        X[:-2] - X[2:],
        self.eps + self.b*X[-1]*(X[-2] - self.d)
        ])
        return dXdt
    

    def gen_time_series(self, x0, tspan, dt):
        x = x0
        X = []
        for t in tspan:
            x = rk4(self.eq, t, x, dt)
            X.append(x)

        self.time_serie = np.array(X)

    
    def _jac(self, X):
        J1 = np.zeros((1, self.N))
        J1[0, 0] = self.a
        J1[0, 1] = -1
        Ji = np.eye(N=self.N-2, M=self.N, k=0) - np.eye(N=self.N-2, M=self.N, k=2)
        JN = np.zeros((1, self.N))
        JN[0, -2] = self.b*X[-1]
        JN[0, -1] = self.b*(X[-2] - self.d)
        return np.vstack([J1, Ji, JN])
    

    def shimada_nagashima(self, X, dt, qr_step=1):
        Q = np.eye(len(X[0]))
        r = 0
        
        for i, x in enumerate(X):
            J = self._jac(x)
            # Q += np.dot(J, Q)*dt
            k1 = np.dot(J, Q)
            k2 = np.dot(J, Q + 0.5*dt*k1)
            k3 = np.dot(J, Q + 0.5*dt*k2)
            k4 = np.dot(J, Q + dt*k3)
            Q = Q + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            if i%qr_step == 0:
                Q, R = np.linalg.qr(Q)        
                r += np.log(np.abs(np.diag(R)))
        
        self.lyapunov_exponents = r/((i+1)*dt)


class Coupled_rossler():
    def __init__(self, a, f, c, omega1, omega2, eps):
        self.a = a
        self.f = f
        self.c = c
        self.omega1 = omega1
        self.omega2 = omega2
        self.eps = eps


    def eq(self, t, X):
        x1, y1, z1, x2, y2, z2 = X
        return [
            -self.omega1*y1 - z1 + self.eps*(x2 - x1),
            self.omega1*x1 + self.a*y1,
            self.f + z1*(x1 - self.c),
            -self.omega2*y2 - z2 + self.eps*(x1 - x2),
            self.omega2*x2 + self.a*y2,
            self.f + z2*(x2 - self.c)
        ]


    def gen_time_series(self, x0, tspan, dt):
        x = x0
        X = []
        for t in tspan:
            x = rk4(self.eq, t, x, dt)
            X.append(x)

        self.time_serie = np.array(X)

    
    def _jac(self, X):
        x1, y1, z1, x2, y2, z2 = X
        return np.array([
            [-self.eps, -self.omega1, -1, self.eps, 0, 0],
            [self.omega1, self.a, 0, 0, 0, 0],
            [z1, 0, x1 - self.c, 0, 0, 0],
            [self.eps, 0, 0, -self.eps, -self.omega2, -1],
            [0, 0, 0, self.omega2, self.a, 0],
            [0, 0, 0, z2, 0, x2 - self.c]
        ])
    

    def shimada_nagashima(self, X, dt, qr_step=1):
        Q = np.eye(len(X[0]))
        r = 0
        
        for i, x in enumerate(X):
            J = self._jac(x)
            # Q += np.dot(J, Q)*dt
            k1 = np.dot(J, Q)
            k2 = np.dot(J, Q + 0.5*dt*k1)
            k3 = np.dot(J, Q + 0.5*dt*k2)
            k4 = np.dot(J, Q + dt*k3)
            Q = Q + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            if i%qr_step == 0:
                Q, R = np.linalg.qr(Q)        
                r += np.log(np.abs(np.diag(R)))
        
        self.lyapunov_exponents = r/((i+1)*dt)


class Kuramoto_model():
    def __init__(self, N, omega, K):
        self.N = N
        self.omega = omega
        self.K = K

    
    def f(self, t, X):
        return self.omega + self.K*np.sum(np.sin(X.reshape(self.N, 1) - X), axis=0)/self.N
    

    def gen_time_series(self, x0, tspan, dt):
        x = x0
        X = []
        for t in tspan:
            x = rk4(self.f, t, x, dt)
            X.append(x)

        self.time_serie = np.array(X)


    def _jac(self, X):
        return self.K/self.N*(-np.diag(np.sum(np.cos(X - X.reshape(self.N, 1)), axis=1)) + np.cos(X - X.reshape(self.N, 1)))


    def shimada_nagashima(self, X, dt, qr_step=1):
        Q = np.eye(self.N)
        r = 0
        
        for i, x in enumerate(X):
            J = self._jac(x)
            # Q += np.dot(J, Q)*dt
            k1 = np.dot(J, Q)
            k2 = np.dot(J, Q + 0.5*dt*k1)
            k3 = np.dot(J, Q + 0.5*dt*k2)
            k4 = np.dot(J, Q + dt*k3)
            Q = Q + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            if i%qr_step == 0:
                Q, R = np.linalg.qr(Q)        
                r += np.log(np.abs(np.diag(R)))
        
        self.lyapunov_exponents = r/((i+1)*dt)


class Coupled_lorenz():
    def __init__(self, sigma, r1, r2, b, gamma):
        self.sigma = sigma
        self.r1 = r1
        self.r2 = r2
        self.b = b
        self.gamma = gamma


    def f(self, t, X):
        x1, y1, z1, x2, y2, z2 = X
        return [
            self.sigma*(y1 - x1) + self.gamma*(x2 - x1),
            self.r1*x1 - x1*z1 - y1,
            x1*y1 - z1*self.b,
            self.sigma*(y2 - x2) + self.gamma*(x1 - x2),
            self.r2*x2 - x2*z2 - y2,
            x2*y2 - z2*self.b,
        ]


    def gen_time_series(self, x0, tspan, dt):
        x = x0
        X = []
        for t in tspan:
            x = rk4(self.f, t, x, dt)
            X.append(x)

        self.time_serie = np.array(X)

    
    def _jac(self, X):
        x1, y1, z1, x2, y2, z2 = X
        return np.array([
            [-self.sigma - self.gamma, self.sigma, 0, self.gamma, 0, 0],
            [self.r1 - z1, -1, -x1, 0, 0, 0],
            [y1, x1, -self.b, 0, 0, 0],
            [self.gamma, 0, 0, -self.sigma - self.gamma, self.sigma, 0],
            [0, 0, 0, self.r2 - z2, -1, -x2],
            [0, 0, 0, y2, x2, -self.b]
        ])
    

    def shimada_nagashima(self, X, dt, qr_step=1):
        Q = np.eye(len(X[0]))
        r = 0
        
        for i, x in enumerate(X):
            J = self._jac(x)
            # Q += np.dot(J, Q)*dt
            k1 = np.dot(J, Q)
            k2 = np.dot(J, Q + 0.5*dt*k1)
            k3 = np.dot(J, Q + 0.5*dt*k2)
            k4 = np.dot(J, Q + dt*k3)
            Q = Q + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            if i%qr_step == 0:
                Q, R = np.linalg.qr(Q)        
                r += np.log(np.abs(np.diag(R)))
        
        self.lyapunov_exponents = r/((i+1)*dt)


def rk4(f, t, x, dt):
    k1 = np.array(f(t, x))
    k2 = np.array(f(t, x + 0.5 * dt * k1))
    k3 = np.array(f(t, x + 0.5 * dt * k2))
    k4 = np.array(f(t, x + dt * k3))
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def ky_dim(lyap):
    lyap = np.sort(lyap.flatten())[::-1]
    accum = np.cumsum(lyap)
    j = sum(accum > 0)
    return (j + accum[j-1]/abs(lyap[j]))
