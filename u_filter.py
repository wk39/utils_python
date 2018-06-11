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

    def predict(self, Q=None, Bu=None):

        if Bu==None:
            Bu = np.zeros_like(self.xm)

        # xm = A * xp + B * u
        self.xm = np.dot(self.A, self.xp) + Bu
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



def ransac_spline(x_data, y_data, n_knots=10, b_periodic=False, b_graph=True, random_seed=None):

    """
    Robust B-Spline regression with scikit-learn

    https://gist.github.com/MMesch/35d7833a3daa4a9e8ca9c6953cbe21d4

    input :
        x_data
        y_data : numpy array
        n_knots

    return :
        

    """

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.interpolate as si
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression, RANSACRegressor,\
                                     TheilSenRegressor, HuberRegressor

    from sklearn.metrics import mean_squared_error


    class BSplineFeatures(TransformerMixin):
        def __init__(self, knots, degree=3, periodic=False):
            self.bsplines = get_bspline_basis(knots, degree, periodic=periodic)
            self.nsplines = len(self.bsplines)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            nsamples, nfeatures = X.shape
            features = np.zeros((nsamples, nfeatures * self.nsplines))
            for ispline, spline in enumerate(self.bsplines):
                istart = ispline * nfeatures
                iend = (ispline + 1) * nfeatures
                features[:, istart:iend] = si.splev(X, spline)
            return features

    def get_bspline_basis(knots, degree=3, periodic=False):
        """Get spline coefficients for each basis spline."""
        nknots = len(knots)
        y_dummy = np.zeros(nknots)

        knots, coeffs, degree = si.splrep(knots, y_dummy, k=degree,
                                          per=periodic)
        ncoeffs = len(coeffs)
        bsplines = []
        for ispline in range(nknots):
            coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
            bsplines.append((knots, coeffs, degree))
        return bsplines




    #np.random.seed(42)
    # X = np.arange(len(y_data)) # np.random.uniform(low=-30, high=30, size=400)
    X = x_data
    x_predict = X
    #y = np.sin(2 * np.pi * 0.1 * X)
    # X_test = np.random.uniform(low=-30, high=30, size=200)
    # y_test = np.sin(2 * np.pi * 0.1 * X_test)

    #y_errors_large = y.copy()
    #y_errors_large[::10] = 6
    y_errors_large = y_data.copy()

    # Make sure that X is 2D
    X = X[:, np.newaxis]
    # X_test = X_test[:, np.newaxis]

    # predict y
    knots = np.linspace(x_data[0], x_data[-1], n_knots) #np.linspace(-30, 30, 20)
    # bspline_features = BSplineFeatures(knots, degree=3, periodic=False)
    bspline_features = BSplineFeatures(knots, degree=3, periodic=b_periodic)
    ##estimators = [('Least-Square', 'g-', 'C0',
    ##               LinearRegression(fit_intercept=False)),
    ##              ('Theil-Sen', 'm-', 'C1', TheilSenRegressor(random_state=42)),
    ##              ('RANSAC', 'r-', 'C2', RANSACRegressor(random_state=42)),
    ##              ('HuberRegressor', 'c-', 'C3', HuberRegressor())]
    if random_seed:
        estimators = [#('Least-Square', 'g-', 'C0', LinearRegression(fit_intercept=False)),
                      # ('RANSAC', 'r-', 'C2', RANSACRegressor(residual_threshold=10))
                      # ('RANSAC', 'r-', 'C2', RANSACRegressor(random_state=42))
                      ('RANSAC', 'r-', 'C2', RANSACRegressor(random_state=random_seed))
                      ]
    else:
        estimators = [#('Least-Square', 'g-', 'C0', LinearRegression(fit_intercept=False)),
                      # ('RANSAC', 'r-', 'C2', RANSACRegressor(residual_threshold=10))
                      # ('RANSAC', 'r-', 'C2', RANSACRegressor(random_state=42))
                      ('RANSAC', 'r-', 'C2', RANSACRegressor())
                      ]

    # fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    # print('b_graph', b_graph)
    if b_graph:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Robust B-Spline Regression with SKLearn')
        ax.plot(X[:, 0], y_errors_large, 'bs', label='scattered data')
    
    for label, style, color, estimator in estimators:
        model = make_pipeline(bspline_features, estimator)
        model.fit(X, y_errors_large)
        # mse = mean_squared_error(model.predict(X_test), y_test)
        y_predicted = model.predict(x_predict[:, None])
        # ax.plot(x_predict, y_predicted, style, lw=2, markevery=8, ms=6,
                # color=color, label=label + ' E={:2.2g}'.format(mse))
        if b_graph:
            ax.plot(x_predict, y_predicted, style, lw=2, label=label)
    if b_graph:
        ax.legend()
        ax.grid()
    # ax.set(ylim=(-2, 8), xlabel='x_data [s]', ylabel='amplitude')
    # plt.show()

    return y_predicted



def moving_average(data, w):

    n = len(data)   # ex) 10
    dw = w//2       # ex) 2

    d = np.copy(data)

    for i in range(n):
        if i<dw:              # i < 2
            d[i] = np.mean(data[:dw*2+1])
        elif n-dw-1 < i:   # 7 < i
            d[i] = np.mean(data[n-(dw*2+1):n])
        else:
            d[i] = np.mean(data[i-dw:i+(2*dw+1)])

    return d


def low_pass_filter(data, gain):

    n = len(data)   # ex) 10

    d = np.copy(data)

    for i in range(1,n):
        d[i] = d[i-1] + gain*(data[i]-d[i-1])

    return d



def slope_limit(data, slope):

    n = len(data)   # ex) 10
    d = np.copy(data)

    for i in range(1,n):

        s = data[i]-d[i-1]

        if s>slope:
            d[i] = d[i-1] + slope
        elif s<-slope:
            d[i] = d[i-1] - slope
        else:
            d[i] = data[i]

    return d



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

    


