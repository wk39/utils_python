import numpy as np

class KalmanFilter:
    ''' Kalman Filter (linear, no control input) '''

    def __init__(self, A, H, Q, R):

        self.A  = np.array(A)
        self.H  = np.array(H)
        self.Q  = np.array(Q)
        self.R  = np.array(R)
        #
        n = self.A.shape[0]
        #
        self.xm = np.zeros(n)
        self.xp = np.zeros(n)
        self.Pm = np.eye(n)
        self.Pp = np.eye(n)
        #
        self.K  = None
        self.y  = None
        self.S  = None

        return

    def predict(self, Q=None):

        # xm = A * xp + B * u
        self.xm = np.dot(self.A, self.xp)
        # Pm = F * Pp * F' + Q
        if Q:
            self.Pm = np.dot(np.dot(self.A, self.Pp), self.A.T) + Q
        else:
            self.Pm = np.dot(np.dot(self.A, self.Pp), self.A.T) + self.Q

        return

    def update(self, z, R=None):

        # y = z - H * xm (innovation)
        self.y  = z - np.dot(self.H, self.xm)
        # S = R + H * Pm * H' (innovation Covariance)
        if R:
            self.S  =      R + np.dot(np.dot(self.H, self.Pm), self.H.T)
        else:
            self.S  = self.R + np.dot(np.dot(self.H, self.Pm), self.H.T)

        # K = Pm * H' * inv(S)
        self.K  = np.dot(np.dot(self.Pm, self.H.T), np.linalg.inv(self.S))

        # xp = xm + K * y
        self.xp = self.xm + np.dot(self.K, self.y)
        # Pp = Pm - K * S * K'
        self.Pp = self.Pm - np.dot(np.dot(self.K, self.S), self.K.T)

        return




if __name__=='__main__':

    import matplotlib.pyplot as plt

    import sys


    # dataset
    dt = 0.1
    amp = 1.0
    t = np.arange(0,100,dt)
    x_true = np.sin(t/100*2*np.pi)*amp
    z = x_true + np.random.randn(*x_true.shape)*amp/20


    # kalman filter
    A = [[ 1, dt, 0.5*dt*dt], [0, 1, dt], [0,0,1]]
    H = [[1,0,0]]
    Q = np.diag([0.01, 0.01, 0.01])
    R = np.diag([999.90])
    
    kf = KalmanFilter(A,H,Q,R)
    kf.Pp = np.diag([10,10,10])

    x_est = np.zeros_like(x_true)
    v_est = np.zeros_like(x_true)
    a_est = np.zeros_like(x_true)
    p_est = np.zeros_like(x_true)
    for i in range(len(t)):
        kf.predict()
        kf.update(z[i])

        x_est[i] = kf.xp[0]
        v_est[i] = kf.xp[1]
        a_est[i] = kf.xp[2]
        p_est[i] = kf.Pp[0,0]


    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.plot(t, x_true)
    ax.plot(t, z, 'gray')
    ax.plot(t, x_est, 'magenta')
    ax.grid()

    ax = plt.subplot(2,1,2)
    # ax.plot(t, p_est)
    ax.plot(t, np.cos(t/100*2*np.pi)*amp*2*np.pi/100)
    ax.plot(t, v_est)
    ax.grid()


    plt.show()
    sys.exit()

    


