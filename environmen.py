
import numpy as np



class CITY(object):

    def __init__(self):
        #self.kk = kk
        self.state_space = 10 #server number
        self.action_space = self.state_space - 1
        self.init_distribution = np.zeros(self.state_space)
        self.origin1 = np.zeros(self.state_space)
        self.user_num = 100
        self.mu = np.array([300,390,305,305,305,365,385,361,300,308])
        #self.mu = 310 * np.ones(self.state_space)
        self.lambda_bar = 3.0

    def reset(self):
        #s = np.random.uniform(0, 1, self.state_space)
        s = np.array([50.0, 10.0, 15.0, 10.0, 70.0, 15.0, 10.0, 10.0, 20.0, 80.0])
        s = s + 80 * np.random.rand(10)
        #s = np.random.normal(s, s * 0.1)
        #self.mu[1] = self.mu[1] - 30 * np.random.rand(1)
        #self.mu[2:5] = self.mu[2:5] + 20 * np.random.rand(3)
        #self.mu[9] = self.mu[9] - 20 * np.random.rand(1)

        s_sum = np.sum(s)

        for index in range(self.state_space):
            s[index] = s[index] / s_sum #initial distribution




        init_time = np.zeros(self.state_space)

        for j in range(self.state_space):
            init_time[j] = (self.lambda_bar * s[j]) / (self.mu[j] - s[j] * self.user_num * self.lambda_bar)

        init_time_sum = np.sum(init_time)
        self.origin1 = np.copy(s)

        return s, init_time_sum



    def step(self, a):

        s_buff = np.copy(self.origin1)
        trans_prob = np.zeros(self.action_space)
        s_ = np.zeros(self.state_space)
        s_change = np.zeros(self.state_space)

        for i in range(self.action_space):
            if a[i] >= 0:
                trans_prob[i] = s_buff[i] * a[i]   # + 1 to 2
                s_change[i] = s_change[i] + trans_prob[i]  # flow out
                s_change[i+1] = s_change[i+1] - trans_prob[i]  # flow in
        for i in range(self.action_space):
            if a[i] < 0:
                trans_prob[i] = - s_buff[i+1] * a[i]  # - 2 to 1
                s_change[i] = s_change[i] - trans_prob[i]  # flow in
                s_change[i+1] = s_change[i+1] + trans_prob[i]  # flow out

        for i in range(self.state_space):
            s_[i] = s_buff[i] - s_change[i]


        self.origin1 = np.copy(s_)

        # reward function

        r_time = np.zeros(self.state_space)
        r_energy = np.zeros(self.state_space)

        for j in range(self.state_space):
            if (self.mu[j] - s_[j] * self.user_num * self.lambda_bar) > 0:
                r_time[j] = (self.lambda_bar * s_[j]) / (self.mu[j] - s_[j] * self.user_num * self.lambda_bar)
                r_energy[j] = s_[j] * self.lambda_bar * (pow(self.mu[j], 2)) / (2 * pow(10, 7))
            else:
                r_time[j] = (self.lambda_bar * s_[j]) / 0.1
                r_energy[j] = s_[j] * self.lambda_bar * (pow(self.mu[j], 2)) / (2 * pow(10, 7))



        reward1 = 100 * np.sum(r_time)

        reward2 = 100 * np.sum(r_energy)

        reward = -reward1 -0.2 * reward2

        # print(reward)
        # print(s_)

        return s_, reward, reward1, reward2

