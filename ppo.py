"""
Created on  Mar 1 2021
@author: wangmeng
"""

import sys
import datetime
sys.path.append("")
# print(sys.path)

import torch.nn.functional as F

import torch
import random
import torch.nn as nn
import numpy as np, argparse
import math
from transformers import AdamW
import torch.optim as optim


from PPO.Reward import compute_reward

from PPO.Memory import *

from model import *

from MMGCN_train import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    def __iPPOnit__(self,args, lr, betas, gamma, K_epochs, rate):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.rate = rate
        self.K_epochs = K_epochs
        # 两个模型

        self.policy = getNewModel(args)
        self.policy_old = getNewModel(args)
        modelPath = 'save/' + args.dataset_name_A + "_" + args.base_model + '.pth'
        self.policy.load_state_dict(torch.load(modelPath))


        #self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr, weight_decay=args.l2)

        self.first_step=True

        if cuda:
            self.policy_old=self.policy_old.cuda()
            self.policy = self.policy.cuda()

        #self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.KLDivLoss = nn.KLDivLoss(reduction="batchmean")
        self.loss_function = nn.NLLLoss()


    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def  train_or_evaluate(self,data,epoch,new=True,train=True):
        losses, preds, labels = [], [], []
        if new:
            model=self.policy
        else:
            model=self.policy_old

        if train:
            model.train()
            self.optimizer.zero_grad()
        else:
            model.eval()


        textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in
                                                                             data[:-1]] if cuda else data[:-1]
        if args.multi_modal:
            if args.mm_fusion_mthd=='concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf],dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf1,textf2,textf3,textf4],dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd=='gated':
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
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = self.loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            self.optimizer.step()

        return labels, preds,log_prob


    def evaluate_by_mem(self,datas,memory):
        all_probs=[]
        all_rewards=[]
        all_preds=[]
        all_labels=[]
        all_logits=[]
        all_raw_loss=[]
        for data in datas:
            labels, preds,probs,logits,raw_loss = self.train_or_evaluate(data, memory, new=True,train=True)
            rewards = compute_reward(preds, labels)
            #print("preds2",preds)
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_rewards.extend(rewards)
            all_logits.extend(logits)
            all_raw_loss.append(raw_loss)
        return all_probs,all_rewards,all_labels,all_logits,all_preds,all_raw_loss



    def update(self,example):

        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
            # Evaluating old actions and values : #给老的动作和状态打分 新打分
            #logprobs, state_values, dist_entropy = self.policy.evaluate()

            # 新老模型训练结果
            labels, preds,new_logits  = self.train_or_evaluate(example,i, new=True,train=False)
            labels, preds,old_logits = self.train_or_evaluate(example,i, new=False,train=False)

            self.policy_old.load_state_dict(self.policy.state_dict())
            if len(old_logits)==0:
                continue

            old_logits=old_logits.to(device).detach()
            new_logits=new_logits.to(device)
            test1=F.log_softmax(old_logits,dim=1)
            test2=F.softmax(new_logits,dim=1)

            loss = self.KLDivLoss(test1,test2)*self.rate
            #print("loss2",loss)
            #print("loss2",loss)
            # self.optimizer.zero_grad()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
            # loss.backward(retain_graph=True)
            # self.optimizer.step()

            # # Finding the ratio (pi_theta / pi_theta__old):
            # ratios = torch.exp(new_logits - old_logprobs.detach())
            # new_rewards = torch.tensor(new_rewards, dtype=torch.float32).to(device)
            # logits=torch.stack(logits)
            labels=torch.tensor(labels).to(device)
            # print("------------------------")
            # print("logits",logits)
            # print("preds",preds)
            # print("labels",labels)


            # Finding Surrogate Loss:
            # advantages = rewards - new_rewards
            #surr1 = ratios * advantages
            #surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            #raw_loss=torch.stack(raw_loss)


                #loss = 0.5 * self.MseLoss(new_rewards, rewards)  +self.loss_function(logits,labels)
                #print("sur_loss",surr1.mean())
                #print("loss_function", self.loss_function(logits,labels))

            #loss = self.loss_function(logits, labels)


            #take gradient step


def eval(model,loss_function,test_loader,dataset_name,args):
    test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model,
                                                                                                       loss_function,
                                                                                                       test_loader, 0,
                                                                                                       args.cuda,
                                                                                                       modals=args.modals,
                                                                                                    args=args)
    print('dataset:', dataset_name)
    print('test: accuracy', test_acc)
    print('test: fscore', test_fscore)


    return test_fscore,test_label,test_pred


idx1=0
idx2=0
len1=0
len2=0

def get_datalist(dataloader):
    datalist=[]
    for data in dataloader:
        datalist.append(data)
    return datalist

example_idx=-1

def get_example_data(datalist):
    global example_idx
    length=len(datalist)
    example_idx+=1
    if example_idx==length:
        example_idx=0

    return datalist[example_idx]




def main(args):
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v3"
    render = False
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 2# update policy every n timesteps
    # constant std for action distribution (Multivariate Normal)
    K_epochs = 1 # update policy for K epochs
    eps_clip = 1  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 1e-5  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    rate=1

    random_seed = 123
    #############################################

    # creating environment
    #env = gym.make(env_name)
    #state_dim = env.observation_space.shape[0]
    #action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        #env.seed(random_seed)
        np.random.seed(random_seed)



    if args.dataset_name_A == 'MELD':
        train_loader_A, valid_loader_A, test_loader_A = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=2,
                                                                   select_rate=args.second_rate)
    if args.dataset_name_A == 'IEMOCAP':
        train_loader_A, valid_loader_A, test_loader_A = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2,
                                                                      select_rate=args.second_rate)

    if args.dataset_name_B == 'MELD':
        train_loader_B, valid_loader_B, test_loader_B = get_MELD_loaders(valid=0.0,
                                                                         batch_size=batch_size,
                                                                         num_workers=2,
                                                                         select_rate=args.second_rate)
    if args.dataset_name_B == 'IEMOCAP':
        train_loader_B, valid_loader_B, test_loader_B = get_IEMOCAP_loaders(valid=0.0,
                                                                            batch_size=batch_size,
                                                                            num_workers=2,
                                                                            select_rate=args.second_rate)

    memory = Memory()
    memory2=Memory()
    ppo = (args,lr, betas, gamma, K_epochs, rate)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    best_score=0
    datalist_A=get_datalist(train_loader_A)
    # training loop
    for i_episode in range(1, max_episodes + 1):
        memory.clear_mask()
        #memory2.clear_mask()
        print("epoch:{},len:{}".format(i_episode,len(train_loader_B)))
        index=0
        memory.all_losses = []
        memory.all_preds = []
        memory.all_labels = []

        test_fscore=eval(ppo.policy, ppo.loss_function,  test_loader_B,args.dataset_name_B,args)
        test_fscore2 = eval(ppo.policy, ppo.loss_function, test_loader_A,args.dataset_name_A,args)

        for data in train_loader_B:

            time_step += 1
            # 训练B数据
            labels, preds,log_prob=ppo.train_or_evaluate(data,i_episode,new=True,train=True)
            # 训练A数据
            example = get_example_data(datalist_A)
            labels, preds,log_prob = ppo.train_or_evaluate(example, i_episode, new=True, train=True)

            if len(preds)==0:
                continue

            if time_step % update_timestep == 0:
                ppo.update(example)
                memory.clear_memory()
                time_step = 0




        if test_fscore>best_score:
            print("best model save",test_fscore)
            torch.save(ppo.policy.state_dict(),'{}+{}_{}.pth'.format(args.dataset_name_B ,args.dataset_name_A ,"ppo"))
            best_score=test_fscore


        # stop training if avg_reward > solved_reward 如果奖励够了
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 100 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


def getNewModel(args):

    # D_audio = 314
    # D_visual = 512
    # D_text = 1024
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = 100 if args.dataset_name == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = 512 if args.dataset_name == 'IEMOCAP' else feat2dim['denseface']
    D_text = 1024  # feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    print("------------")
    print(args.multi_modal)
    print(args.modals)
    args.multi_modal=True
    if args.multi_modal:
        if args.mm_fusion_mthd == 'concat':
            if args.modals == 'avl':
                D_m = D_audio + D_visual + D_text
            elif args.modals == 'av':
                D_m = D_audio + D_visual
            elif args.modals == 'al':
                D_m = D_audio + D_text
            elif args.modals == 'vl':
                D_m = D_visual + D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if args.modals == 'a':
            D_m = D_audio
        elif args.modals == 'v':
            D_m = D_visual
        elif args.modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 512  # 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.dataset_name == 'MELD' else 2
    #n_classes = 5
    n_classes = 7 if args.dataset_name == 'MELD' else 6 if args.dataset_name == 'IEMOCAP' else 7 if  args.dataset_name == 'EmoryNLP' else 7 if args.dataset_name == 'DailyDialog' else 1

    if args.graph_model:
        seed_everything()

        model = Model(args.base_model,
                      D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                      n_speakers=n_speakers,
                      max_seq_len=200,
                      window_past=args.windowp,
                      window_future=args.windowf,
                      n_classes=n_classes,
                      listener_state=args.active_listener,
                      context_attention=args.attention,
                      dropout=args.dropout,
                      nodal_attention=args.nodal_attention,
                      no_cuda=args.no_cuda,
                      graph_type=args.graph_type,
                      use_topic=args.use_topic,
                      alpha=args.alpha,
                      multiheads=args.multiheads,
                      graph_construct=args.graph_construct,
                      use_GCN=args.use_gcn,
                      use_residue=args.use_residue,
                      D_m_v=D_visual,
                      D_m_a=D_audio,
                      modals=args.modals,
                      att_type=args.mm_fusion_mthd,
                      av_using_lstm=args.av_using_lstm,
                      Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                      dataset=args.dataset_name,
                      use_speaker=args.use_speaker,
                      use_modal=args.use_modal,
                      norm=args.norm,
                      edge_ratio=args.edge_ratio,
                      num_convs=args.num_convs,
                      opn=args.opn)

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if args.cuda:
        model.cuda()

    return model




if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='GRU', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='hyper', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False, help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat_DHT', help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--dataset_name_A', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--dataset_name_B', default='MELD', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='LN2', help='NORM type')

    parser.add_argument('--edge_ratio', type=float, default=0.01, help='edge_ratio')

    parser.add_argument('--num_convs', type=int, default=3, help='num_convs in EH')

    parser.add_argument('--opn', default='corr', help='option')

    parser.add_argument('--second_rate', type=float, default=0.3)

    parser.add_argument('--second_train', type=bool, default=False)

    parser.add_argument('--pretrained_model', type=str, default='IEMOCAP',
                        help='model name')

    args = parser.parse_args()
    today = datetime.datetime.now()
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


    # print('building model..')
    # if args.basemodel == 'transfo_xl':
    #     model = ERC_transfo_xl(args, n_classes)
    # elif args.basemodel in ['xlnet', 'xlnet_dialog']:
    #     model = ERC_xlnet(args, n_classes, use_cls=True)
    main(args)