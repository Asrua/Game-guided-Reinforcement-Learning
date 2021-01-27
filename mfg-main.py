import numpy as np

S = 10  # the number of server
N = 100 #the umber of user
dt = 0.1 #step size
iter = 30000 #iterations
lambda_bar = 3  #user task

#s = np.random.uniform(0, 1, 10)
s = [90.0, 10.0, 8.0, 6.0, 80.0, 20.0, 10.0, 5.0, 10.0, 70.0]
s = np.random.normal(s, s * 0.1)

s_sum = np.sum(s)
for index in range(S):
    s[index] = s[index]/s_sum  #distribution of users



#mu = np.ones(S) * 100 # service rate
#mu = 310 * np.ones(10)
mu=[300,390,305,305,305,365,385,361,300,308]
t = np.zeros(S)
e = np.zeros(S)

def mf(s, S, lambda_bar, mu, N):
    F = np.zeros(10)
    for i in range(S):
        F[i] = (lambda_bar * mu[i])/pow((mu[i]-(s[i] * N * lambda_bar)), 2)

    A = np.zeros((S, S))
    A[0, 0] = - np.max(F[0] - F[1], 0)
    A[0, 1] = np.max(F[1] - F[0], 0)

    A[S-1, S-1] = - np.max(F[S-1] - F[S-2], 0)
    A[S-1, S-2] = np.max(F[S-2] - F[S-1], 0)

    for i in range(1, S-1):
        A[i, i] = -np.max(F[i] - F[i + 1], 0) - np.max(F[i] - F[i - 1], 0)
        A[i, i + 1] = np.max(F[i + 1] - F[i], 0)
        A[i, i - 1] = np.max(F[i - 1] - F[i], 0)

    return A






for k in range(iter):
    AA = mf(s, S, lambda_bar, mu, N)
    s = s + np.transpose(np.dot(AA, np.transpose(s))) * dt
    #print(s)
    #print(np.sum(s))
    for j in range(S):
        t[j] = (lambda_bar * s[j])/(mu[j] - s[j] * N * lambda_bar)
        e[j] = lambda_bar * (pow(mu[j], 2))

    T = np.sum(100*t)
    print(np.sum(100*t))







