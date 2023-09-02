import pickle


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def processIEMOCAP(path):

    videoIDs, videoSpeakers, videoLabels, videoText, \
    videoAudio, videoVisual, videoSentence, trainVid, \
    testVid = pickle.load(open(path, 'rb'), encoding='latin1')

    multi_data=load_pickle("ERC_features/iemocap_data.pkl的副本")

    roberta1, roberta2, roberta3, roberta4= pickle.load(open('ERC_features/iemocap_features_roberta2.pkl', 'rb'), encoding='latin1')

    new_roberta1={}
    new_roberta2 = {}
    new_roberta3 = {}
    new_roberta4 = {}

    audioMap = {}
    for k, v in multi_data['audio'][0].items():
        audioMap[k] = v
    for k, v in multi_data['audio'][1].items():
        audioMap[k] = v
    for k, v in multi_data['audio'][2].items():
        audioMap[k] = v

    videoMap = {}
    for k, v in multi_data['video'][0].items():
        videoMap[k] = v
    for k, v in multi_data['video'][1].items():
        videoMap[k] = v
    for k, v in multi_data['video'][2].items():
        videoMap[k] = v

    videoSpeakers2={}
    videoLabels2={}
    videoTexts2={}
    videoVisual2={}
    videoSentences2={}
    videoIds2={}
    videoAudio2={}

    for vid in videoLabels:
        print(vid)
        videoIds=[]
        sentences=[]
        visuals=[]
        audios=[]
        texts=[]
        speakers=[]
        labels=[]
        labelList=videoLabels[vid]
        index=0
        robert1s=[]
        robert2s = []
        robert3s = []
        robert4s = []
        for label in labelList:
            if label==0:
                cid=videoIDs[vid][index]
                if(cid in videoMap.keys()):
                    visuals.append(videoMap[cid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[cid].flatten()[::32])
                labels.append(2)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                videoIds.append(cid)
                index+=1
            elif label==1:
                cid=videoIDs[vid][index]
                if(cid in videoMap.keys()):
                    visuals.append(videoMap[cid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[cid].flatten()[::32])
                labels.append(3)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                videoIds.append(cid)
                index+=1
            elif label==2:
                cid=videoIDs[vid][index]
                if (cid in videoMap.keys()):
                    visuals.append(videoMap[cid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[cid].flatten()[::32])
                labels.append(0)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                videoIds.append(cid)
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                index+=1
            elif label==3:
                cid=videoIDs[vid][index]
                if (cid in videoMap.keys()):
                    visuals.append(videoMap[cid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[cid].flatten()[::32])
                labels.append(4)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                videoIds.append(cid)
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                index+=1
            elif label==4:
                cid=videoIDs[vid][index]
                if (cid in videoMap.keys()):
                    visuals.append(videoMap[cid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[cid].flatten()[::32])
                labels.append(1)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                videoIds.append(cid)
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                index+=1
        videoIds2[vid] = videoIds
        videoTexts2[vid] = texts
        videoAudio2[vid] = audios
        videoLabels2[vid] = labels
        videoVisual2[vid] = visuals
        videoSentences2[vid] = sentences
        videoSpeakers2[vid] = speakers
        new_roberta1[vid] = robert1s
        new_roberta2[vid] = robert2s
        new_roberta3[vid] = robert3s
        new_roberta4[vid] = robert4s

    data=[videoIds2, videoSpeakers2, videoLabels2, videoTexts2,videoAudio2, videoVisual2, videoSentences2, trainVid,testVid]

    to_pickle(data, "ERC_features/IEMOCAP_features2.pkl")

    roberta_data = [new_roberta1, new_roberta2, new_roberta3, new_roberta4]

    to_pickle(roberta_data, "ERC_features/iemocap_features_roberta2.pkl")

def processMELD(path):
    videoIDs,videoSpeakers, videoLabels, videoText, \
    videoAudio, videoVisual,videoSentence, trainVid, \
    testVid, _ = pickle.load(open(path, 'rb'))

    videoSpeakers2 = {}
    videoLabels2 = {}
    videoTexts2 = {}
    videoVisual2 = {}
    videoSentences2 = {}
    videoIds2 = {}
    videoAudio2 = {}

    multi_data=load_pickle("ERC_features/meld_data.pkl的副本")
    _, _, _, roberta1,roberta2, roberta3, roberta4, \
    _, _, _, _ \
        = pickle.load(open("ERC_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')
    audioMap = {}
    new_roberta1={}
    new_roberta2 = {}
    new_roberta3 = {}
    new_roberta4 = {}

    for k, v in multi_data['audio'][0].items():
        audioMap[k] = v
    for k, v in multi_data['audio'][1].items():
        lst=k.split("_")
        newK=str(int(lst[0])+1038)+"_"+lst[1]
        audioMap[newK] = v
    for k, v in multi_data['audio'][2].items():
        lst = k.split("_")
        newK=str(int(lst[0])+1152)+"_"+lst[1]
        audioMap[newK] = v

    videoMap = {}
    for k, v in multi_data['video'][0].items():
        videoMap[k] = v
    print(k)
    for k, v in multi_data['video'][1].items():
        lst = k.split("_")
        newK = str(int(lst[0]) + 1038) + "_" + lst[1]
        videoMap[newK] = v
    print(newK)
    for k, v in multi_data['video'][2].items():
        lst = k.split("_")
        newK = str(int(lst[0]) + 1152) + "_" + lst[1]
        videoMap[newK] = v
    print(newK)

    for vid in videoLabels:
        print(vid)
        videoIds = []
        sentences = []
        visuals = []
        audios = []
        texts = []
        speakers = []
        labels = []
        labelList = videoLabels[vid]
        index = 0
        robert1s=[]
        robert2s = []
        robert3s = []
        robert4s = []
        for label in labelList:
            if label == 0:
                uid = str(vid) + "_" + str(videoIDs[vid][index])
                if (uid in videoMap.keys()):
                    visuals.append(videoMap[uid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[uid].flatten()[::32])
                labels.append(0)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                videoIds.append(videoIDs[vid][index])
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                index+=1
            elif label == 1:
                uid = str(vid) + "_" + str(videoIDs[vid][index])
                if (uid in videoMap.keys()):
                    visuals.append(videoMap[uid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[uid].flatten()[::32])
                labels.append(1)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                videoIds.append(videoIDs[vid][index])
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                index+=1
            elif label == 3:
                uid = str(vid) + "_" + str(videoIDs[vid][index])
                if (uid in videoMap.keys()):
                    visuals.append(videoMap[uid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[uid].flatten()[::32])
                labels.append(3)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                videoIds.append(videoIDs[vid][index])
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                index+=1
            elif label == 4:
                uid = str(vid) + "_" + str(videoIDs[vid][index])
                if (uid in videoMap.keys()):
                    visuals.append(videoMap[uid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[uid].flatten()[::32])
                labels.append(2)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                videoIds.append(videoIDs[vid][index])
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                index+=1
            elif label == 6:
                uid = str(vid) + "_" + str(videoIDs[vid][index])
                if (uid in videoMap.keys()):
                    visuals.append(videoMap[uid].flatten()[::4])
                else:
                    continue
                audios.append(audioMap[uid].flatten()[::32])
                labels.append(4)
                texts.append(videoSentence[vid][index])
                sentences.append(videoSentence[vid][index])
                speakers.append(videoSpeakers[vid][index])
                videoIds.append(videoIDs[vid][index])
                robert1s.append(roberta1[vid][index])
                robert2s.append(roberta2[vid][index])
                robert3s.append(roberta3[vid][index])
                robert4s.append(roberta4[vid][index])
                index+=1
        if len(videoIds)!=0:
            videoIds2[vid] = videoIds
            videoTexts2[vid] = texts
            videoAudio2[vid] = audios
            videoLabels2[vid] = labels
            videoVisual2[vid] = visuals
            videoSentences2[vid] = sentences
            videoSpeakers2[vid] = speakers
            new_roberta1[vid]=robert1s
            new_roberta2[vid] = robert2s
            new_roberta3[vid] = robert3s
            new_roberta4[vid] = robert4s
        else:
            if vid in trainVid:
                trainVid.remove(vid)
            if vid in testVid:
                testVid.remove(vid)

    data=[videoIds2, videoSpeakers2, videoLabels2, videoTexts2,videoAudio2, videoVisual2, videoSentences2, trainVid,testVid]

    roberta_data=[new_roberta1,new_roberta2,new_roberta3,new_roberta4]

    to_pickle(data, "ERC_features/MELD_features2.pkl")
    to_pickle(roberta_data, "ERC_features/meld_feature_roberta2.pkl")



if __name__ == '__main__':
    processIEMOCAP('./ERC_features/IEMOCAP_features_raw.pkl')
    processMELD('./ERC_features/MELD_features_raw1.pkl')