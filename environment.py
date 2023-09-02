import numpy as np

class MyEnviroment(object):
    def __init__(self, train_data=None):
        self.train_data = train_data
        self.current_index = 0
        self.action_space = 6
        self.env_targets =float(10)
    def reset(self):
        return self.train_data[0]
    '''
    action: 0-9 categori, -1 : start and no reward
    return: next_state(image), reward
    '''
    def step(self, action):
        if action==-1:
            _c_index = self.current_index
            self.current_index = self._sample_index()
            return (self.train_X[_c_index], 0)
        r = self.reward(action)
        self.current_index = self.current_index+1
        return self.train_X[self.current_index], r

    def reward(self, a,q):
        #ai = np.argmax(a.detach().cpu().numpy(),1)
        r=[]
        for i in range(len(a)):
            ri=(1 if q[i]==a[i] else -1)
            r.append(ri)
        return np.array(r)

    def reward2(self, a,q):
        r=[]
        l = float(self.getLen(q))
        w = self.env_targets / l
        ai = np.argmax(a.detach().cpu().numpy(),1)
        for i in range(len(a)):

            ri=(w if q[i]==ai[i] else 0)
            r.append(ri)
        return np.array(r)
    def getLen(self,q):
        l=0
        for i in q:
            if i!=-1:
                l+=1
        return l
