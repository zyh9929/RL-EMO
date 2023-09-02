from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support

def compute_reward(preds,labels):
    rewards=[]
    for i in range(len(preds)):
        if(preds[i]==labels[i]):
            rewards.append(1)
        else:
            rewards.append(-0.5)

    return rewards


def sum(rewards):
    sum=0
    for i in range(rewards):
        sum+=i
    return sum