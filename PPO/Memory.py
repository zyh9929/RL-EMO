
class Memory:
    def __init__(self):
        self.datas = []
        self.logprobs = []
        self.rewards = []
        self.labels = []
        self.is_terminals = []
        self.speaker_mask=None
        self.mems=None
        self.window_mask=None
        self.all_losses=[]
        self.all_preds=[]
        self.all_labels=[]

    def clear_memory(self):
        # del语句作用在变量上，而不是数据对象上。删除的是变量，而不是数据。
        del self.datas[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.labels[:]
        del self.is_terminals[:]

    def clear_mask(self):
        self.speaker_mask = None
        self.mems = None
        self.window_mask = None