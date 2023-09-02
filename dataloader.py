import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy
from  transformers import BertTokenizer, TransfoXLTokenizer, XLNetTokenizer
import json
class IEMOCAPDataset(Dataset):

    def __init__(self, train=True,select_rate=1):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./ERC_features/IEMOCAP_features_raw.pkl', 'rb'), encoding='latin1')
        #self.testVid = pickle.load(open('./ERC_features/IEMOCAP_features2.pkl', 'rb'), encoding='latin1')


        #self.roberta1, self.roberta2, self.roberta3, self.roberta4 = pickle.load(open('./ERC_features/iemocap_features_roberta2.pkl', 'rb'), encoding='latin1')
        _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        _, _, _, _ = pickle.load(open('ERC_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        keys = [x for x in (self.trainVid if train else self.testVid)]

        self.keys=get_train_valid_sampler(keys,select_rate)

        self.len = len(self.keys)

        #textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

def one_hot(a,num):
    batch_size=len(a)
    a=torch.LongTensor(a)
    a=a.reshape(batch_size,1)
    one_hot = torch.zeros(batch_size, num).scatter_(1, a, 1)
    return one_hot

class DailyDialogDataset(Dataset):

    def __init__(self, type):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoSentence = pickle.load(open('./data/DailyDialog/{}.pkl'.format(type), 'rb'), encoding='latin1')
        #self.testVid = pickle.load(open('./ERC_features/IEMOCAP_features2.pkl', 'rb'), encoding='latin1')


        #self.roberta1, self.roberta2, self.roberta3, self.roberta4 = pickle.load(open('./ERC_features/iemocap_features_roberta2.pkl', 'rb'), encoding='latin1')
        #_, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        #_, _, _, _ = pickle.load(open('ERC_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.len = len(self.videoLabels)
        self.processSpeaker()

        #textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
    def processSpeaker(self):
        speaker_num=0
        for speakers in self.videoSpeakers:
            for sp in speakers:
                if(speaker_num<sp):
                    speaker_num=sp
        newVideoSpeakers=[]
        for speakers in self.videoSpeakers:
            speakerId=one_hot(speakers,speaker_num+1)
            newVideoSpeakers.append(speakerId)
        self.videoSpeakers=newVideoSpeakers




    def __getitem__(self, index):
        vid = index
        return torch.stack(self.videoText[vid]), \
               torch.stack(self.videoText[vid]), \
               torch.stack(self.videoText[vid]), \
               torch.stack(self.videoText[vid]), \
                torch.rand(1,2), \
                torch.rand(1, 2), \
                torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

class EmoryNLPDataset(Dataset):

    def __init__(self, type):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoSentence = pickle.load(open('./data/EmoryNLP/{}.pkl'.format(type), 'rb'), encoding='latin1')
        #self.testVid = pickle.load(open('./ERC_features/IEMOCAP_features2.pkl', 'rb'), encoding='latin1')


        #self.roberta1, self.roberta2, self.roberta3, self.roberta4 = pickle.load(open('./ERC_features/iemocap_features_roberta2.pkl', 'rb'), encoding='latin1')
        #_, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        #_, _, _, _ = pickle.load(open('ERC_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.len = len(self.videoLabels)
        self.processSpeaker()

        #textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
    def processSpeaker(self):
        speaker_num=0
        for speakers in self.videoSpeakers:
            for sp in speakers:
                if(speaker_num<sp):
                    speaker_num=sp
        newVideoSpeakers=[]
        for speakers in self.videoSpeakers:
            speakerId=one_hot(speakers,speaker_num+1)
            newVideoSpeakers.append(speakerId)
        self.videoSpeakers=newVideoSpeakers




    def __getitem__(self, index):
        vid = index
        return torch.stack(self.videoText[vid]), \
               torch.stack(self.videoText[vid]), \
               torch.stack(self.videoText[vid]), \
               torch.stack(self.videoText[vid]), \
                torch.rand(1,2), \
                torch.rand(1, 2), \
                torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid,_= pickle.load(open('ERC_features/MELD_features_raw1.pkl', 'rb'))

        #self.roberta1, self.roberta2, self.roberta3, self.roberta4, = pickle.load(open("./ERC_features/meld_feature_roberta2.pkl", 'rb'), encoding='latin1')
        _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        _, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open("./ERC_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid  

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

def get_train_valid_sampler(trainset, valid=0.3):
    size = len(trainset)
    split = int(valid*size)
    return trainset[:split]



def load_vocab(dataset_name,classes_num):
    import sys
    print(sys.path)
    speaker_vocab = pickle.load(open('/mnt/DialogXL-main/data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('/mnt/DialogXL-main/data/%s/label_vocab_%d.pkl' % (dataset_name,classes_num), 'rb'))
    person_vec_dir = '../data/%s/person_vect.pkl' % (dataset_name)
    # if os.path.exists(person_vec_dir):
    #     print('Load person vec from ' + person_vec_dir)
    #     person_vec = pickle.load(open(person_vec_dir, 'rb'))
    # else:
    #     print('Creating personality vectors')
    #     person_vec = np.random.randn(len(speaker_vocab['itos']), 100)
    #     print('Saving personality vectors to' + person_vec_dir)
    #     with open(person_vec_dir,'wb') as f:
    #         pickle.dump(person_vec, f, -1)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec

def read_datas(dataset_name, batch_size):
    # training set
    with open('/mnt/DialogXL-main/data/%s/train_data.json' % (dataset_name), encoding='utf-8') as f:
        train_raw = json.load(f)
    train_raw = sorted(train_raw,key = lambda x:len(x))
    new_train_raw = []
    for i in range(0, len(train_raw), batch_size):
        new_train_raw.append(train_raw[i:i+batch_size])

    with open('/mnt/DialogXL-main/data/%s/dev_data.json' % (dataset_name), encoding='utf-8') as f:
        dev_raw = json.load(f)
    dev_raw = sorted(dev_raw,key = lambda x:len(x))
    new_dev_raw = []
    for i in range(0, len(dev_raw), batch_size):
        new_dev_raw.append(dev_raw[i:i+batch_size])

    with open('/mnt/DialogXL-main/data/%s/test_data.json' % (dataset_name), encoding='utf-8') as f:
        test_raw = json.load(f)
    test_raw = sorted(test_raw,key = lambda x:len(x))
    new_test_raw = []
    for i in range(0, len(test_raw), batch_size):
        new_test_raw.append(test_raw[i:i+batch_size])

    return new_train_raw, new_dev_raw, new_test_raw