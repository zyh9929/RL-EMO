from MMGCN_train import *
from environment import MyEnviroment
from model import *
import torch
import torch.nn as nn
from collections import deque
import time
from ppo import getNewModel,eval
import tqdm

def one_hot(a):
    batch_size=len(a)
    a=torch.LongTensor(a)
    a=a.reshape(batch_size,1)
    one_hot = torch.zeros(batch_size, args.n_classes).scatter_(1, a, 1)
    return one_hot

def train_or_evaluate(model, data, epoch,loss_function, train=True,args=None):
    losses, preds, labels = [], [], []
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in
                                                                         data[:-1]] if args.cuda else data[:-1]


    if args.multi_modal:
        if args.mm_fusion_mthd == 'concat':
            if modals == 'avl':
                textf = torch.cat([acouf, visuf, textf1, textf2, textf3, textf4], dim=-1)
            elif modals == 'av':
                textf = torch.cat([acouf, visuf], dim=-1)
            elif modals == 'vl':
                textf = torch.cat([visuf, textf1, textf2, textf3, textf4], dim=-1)
            elif modals == 'al':
                textf = torch.cat([acouf, textf1, textf2, textf3, textf4], dim=-1)
            else:
                raise NotImplementedError
        elif args.mm_fusion_mthd == 'gated':
            textf = textf1
    else:
        if modals == 'a':
            textf = acouf
        elif modals == 'v':
            textf = visuf
        elif modals == 'l':
            textf = textf1
        else:
            raise NotImplementedError

    lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
    if args.multi_modal and args.mm_fusion_mthd == 'gated':
        log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths, acouf, visuf)
    elif args.multi_modal and args.mm_fusion_mthd == 'concat_subsequently':
        log_prob, e_i, e_n, e_t, e_l = model([textf1, textf2, textf3, textf4], qmask, umask, lengths, acouf, visuf,
                                             epoch)
    elif args.multi_modal and args.mm_fusion_mthd == 'concat_DHT':
        log_prob, e_i, e_n, e_t, e_l = model([textf1, textf2, textf3, textf4], qmask, umask, lengths, acouf, visuf,
                                             epoch)
    else:
        log_prob, e_i, e_n, e_t, e_l = model([textf1, textf2, textf3, textf4], qmask, umask, lengths)
    label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
    loss = loss_function(log_prob, label)
    preds.append(torch.argmax(log_prob, 1).cpu().numpy())
    losses.append(loss.item())
    if train:
        loss.backward()
        optimizer.step()

    return label.cpu().numpy(), preds, log_prob,umask,loss





class DQNModel(nn.Module):
    def __init__(self,erc_xlnet):
        super().__init__()
        self.erc_xlnet=erc_xlnet
        self.loss_func =  nn.MSELoss() # 交叉墒损失

    def forward(self, s, a,p_pred,epoch,train=True):
        #emission,_=self.erc_xlnet.evalaute(s)
        if train:
            self.erc_xlnet.train()
        labels, preds,emission,_,loss1=train_or_evaluate(self.erc_xlnet,s,epoch,loss_function,train=False,args=args)
        a=one_hot(a)
        a=torch.Tensor(a).to(device)
        p_pred=torch.Tensor(p_pred).to(device)
        p_list=[]
        for i in range(emission.shape[0]):
            ei=emission[i]
            ai=a[i]
            pi=torch.dot(ei,ai)
            p_list.append(pi)

        p_=torch.stack(p_list)

        loss2=self.loss_func(p_,p_pred)

        loss=loss1+loss2
        return loss

def get_q_values(state,epoch):
    labels, preds,log_prob,_,_= train_or_evaluate(actor_q_model,state,epoch,loss_function,train=False,args=args)
    qvalues=log_prob.detach().cpu().numpy()
    return np.argmax(qvalues,1), np.max(qvalues,1),labels


def discount_cumsum(q,r,alpha2,gamma):
    discount_cumsum = np.zeros_like(q)
    discount_cumsum[-1] = q[-1]
    for t in reversed(range(q.shape[0]-1)):
        #discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        discount_cumsum[t] = (1 - alpha2) * q[t] + alpha2 * (r[t].reshape(-1, 1) + gamma * q[t+1])
    return discount_cumsum


def remember(state,action,action_q,reward):
    memory.append([state,action,action_q,reward])

# def replay():
#     if len(memory) < replay_size:
#         return
#     #从记忆中i.i.d采样
#     samples = sample_ram(replay_size)
#     #展开所有样本的相关数据
#     #这里next_states没用 因为和上一个state无关。
#     states, actions, old_q, rewards, next_states = zip(*samples)
#     states, actions, old_q, rewards = np.array(states),np.array(actions).reshape(-1,1),\
#                                     np.array(old_q).reshape(-1,1),np.array(rewards).reshape(-1,1)
#
#     actions_one_hot = np_utils.to_categorical(actions,num_actions)
#     #print(states.shape,actions.shape,old_q.shape,rewards.shape,actions_one_hot.shape)
#     #从actor获取下一个状态的q估计值
#     inputs_ = [next_states,np.ones((replay_size,num_actions))]
#     qvalues = actor_q_model.predict(inputs_)
#
#     q = np.max(qvalues,axis=1,keepdims=True)
#     q = 0
#     q_estimate = (1-alpha)*old_q +  alpha *(rewards.reshape(-1,1) + gamma * q) # 这里的q就是预期希望的q
#     history = critic_model.fit([states,actions_one_hot],q_estimate,epochs=1,verbose=0) #使用state和action进行训练，其实也就是图片输入和分类结果，还有q：分类概率
#     return np.mean(history.history['loss'])

def train_step(r,s,q,a,epoch):
    #q_=critic_model(s,a)
    critic_model.train()
    loss=critic_model(s,a,q,epoch)
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    critic_model.eval()

    return loss




def main(args):
    env=MyEnviroment(train_data=None)

    replay_size = 64
    #epoches = 2000
    pre_train_num = 256
 # every state is i.i.d
    #alpha2 = args.alpha2
    forward = 512

    gamma = args.gamma  # every state is i.i.d
    alpha2 = args.alpha2

    if args.dataset_name == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                   batch_size=batch_size,
                                                                   num_workers=2,
                                                                   select_rate=args.second_rate)
    elif args.dataset_name == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2,
                                                                      select_rate=args.second_rate)
    elif args.dataset_name == 'EmoryNLP':
        train_loader, valid_loader, test_loader = get_EmoryNLP_loaders(valid=0.0,
                                                                       batch_size=batch_size,
                                                                       num_workers=2,
                                                                       select_rate=args.second_rate)
    elif args.dataset_name == 'DailyDialog':
        train_loader, valid_loader, test_loader = get_DailyDialog_loaders(valid=0.0,
                                                                          batch_size=batch_size,
                                                                          num_workers=2,
                                                                          select_rate=args.second_rate)



    memory.clear()
    total_rewards = 0
    reward_rec = []
    every_copy_step = 128
    #pbar = tqdm(range(1, epoches + 1))

    torch.backends.cudnn.enabled = False
    best_fscore=0
    best_label=[]
    best_pred=[]
    for epoch in range(epoches):
        total_rewards = 0
        epo_start = time.time()
        # 获取data
        print("epoch",epoch,len(train_loader))
        index=0
        for batch_data in train_loader:
            #batch_data = [x.to(device) for x in batch_data]
            #print(index)
            index+=1
            # 对每个状态使用epsilon_greedy选择
            s=batch_data
            a, q ,labels= get_q_values(s,epoch)  # 决策模型可能会决策出q值
            # play 环境每次给出奖励
            r = env.reward(a,labels)

            q_estimate=discount_cumsum(q,r,alpha2,gamma) #给出的奖励
            # 加入到经验记忆中 一个transition
            remember(s, a, q, q_estimate)  # 决策模型的q值存入,state是图片，action是标签,  q应该是带梯度的
            # 从记忆中采样回放，保证iid。实际上这个任务中这一步不是必须的。

            loss = train_step(r,s,q_estimate,a,epoch)  # 训练之前的数据

            actor_model.load_state_dict(critic_model.state_dict())
            total_rewards += r.sum()

            #print("loss",loss)

        reward_rec.append(total_rewards)
        fscore,pred,label=eval(actor_q_model, loss_function,test_loader,args.dataset_name,args)
        if fscore>best_fscore:
            best_fscore=fscore
            best_pred=pred
            best_label=label
            print("best fscore",best_fscore)
    best_mask=None

    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))




if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='GRU', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='GCN3', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False,
                        help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=True,
                        help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False,
                        help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='none',
                        help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False,
                        help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--dataset_name', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=True, help='whether to use modal embedding')

    parser.add_argument('--norm', default='LN2', help='NORM type')

    parser.add_argument('--edge_ratio', type=float, default=0.01, help='edge_ratio')

    parser.add_argument('--num_convs', type=int, default=3, help='num_convs in EH')

    parser.add_argument('--opn', default='corr', help='option')

    parser.add_argument('--second_rate', type=float, default=0.3)

    parser.add_argument('--gamma', type=float, default=0)

    parser.add_argument('--alpha2', type=float, default=0.5)

    parser.add_argument('--second_train', type=bool, default=False)

    parser.add_argument('--pretrained_model', type=str, default='IEMOCAP',
                        help='model name')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print()
    print()
    print()
    print()
    print("================================================================================================")
    print("LR:{},L2:{},ALPHA:{},GAMMA:{}".format(args.lr,args.l2,args.alpha2,args.gamma))
    print()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals

    lr = args.lr

    memory = deque(maxlen=512)
    replay_size = 64
    epoches = 50
    pre_train_num = 256
    forward = 512
    epislon_total = 2018
    num_actions=7

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    actor_q_model=getNewModel(args)
    actor_model=DQNModel(actor_q_model)
    critic_q_model=getNewModel(args)
    critic_model=DQNModel(critic_q_model)
    actor_q_model.to(device)
    actor_model.to(device)
    critic_q_model.to(device)
    critic_model.to(device)

    optimizer = optim.Adam(critic_model.parameters(), lr=args.lr, weight_decay=args.l2)

    loss_function = nn.NLLLoss()

    if args.dataset_name == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])

    if args.dataset_name != 'IEMOCAP':
        loss_function = nn.NLLLoss()
    else:
        if args.class_weight:
            if args.graph_model:
                #loss_function = FocalLoss()
                loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    args.n_classes = 7 if args.dataset_name == 'MELD' else 6 if args.dataset_name == 'IEMOCAP' else 7 if  args.dataset_name == 'EmoryNLP' else 7 if args.dataset_name == 'DailyDialog' else 1

    args.Dataset=args.dataset_name



    main(args)