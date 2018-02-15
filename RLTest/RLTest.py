import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import gym



use_gpu = torch.cuda.is_available()

if use_gpu == True:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor



def forward(x, w1,w2,b1,b2):
    nntanh = nn.Tanh()
    return nntanh(x.mm(w1)+b1).mm(w2)+b2

def actionOneHot(action):
    if action == 0:
        return np.array([1,0])
    if action == 1:
        return np.array([0,1])
    
    return np.array([0,0])

def actionMask(y, mask):
    return (y * mask).sum(1)

def chooseAction(bestAction, greedyFrac):

    if np.random.rand() < greedyFrac:
        action = bestAction
    else:
        if np.random.rand() < 0.5:
            action = 0
        else:
            action = 1

    return action

def greedyFrac(maxFrac, timeToMax, time):
    return min((time / timeToMax)*maxFrac, maxFrac)

def buildX(obs, N = None):
    if N == None:
        n = 1
    else:
        n = N
    return Variable(torch.from_numpy(obs.reshape((n,4))).type(dtype),requires_grad=False)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    episodes = 2000
    episodeLength = 1000
    fixedWUpdate = 1000
    discount = 0.99
    maxGreedyFrac = 0.95
    timeToMax = 5000
    renderEpisode = 100
    maxMemorySize = 10000


    D_in, D_out, H  = 4, 2, 16
    N = 50


    w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
    w1Fixed = Variable(w1.data, requires_grad=False)


    w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
    w2Fixed = Variable(w2.data, requires_grad=False)

    b1 = Variable(torch.randn(H).type(dtype), requires_grad=True)
    b1Fixed = Variable(b1.data, requires_grad=False)


    b2 = Variable(torch.randn(D_out).type(dtype), requires_grad=True)
    b2Fixed = Variable(b2.data, requires_grad=False)

    step_Size = 1e-5
    totalFrameCounter = 0
    rewards = np.zeros(episodes)


    replayMemory = []
    for e in range(episodes):
        obs = env.reset()
        r = 0.0
        done = False
        info = {}

        x = buildX(obs)
        bestCost, bestAction = torch.max(forward(x, w1Fixed, w2Fixed,b1Fixed, b2Fixed), 1)
    
        episodeReward = 0
        episodeAvgLoss = 0
        endTime = episodeLength - 1

        for t in range(episodeLength):

            if e % renderEpisode == 0:
                env.render()

            episodeReward += r


            y = Variable(torch.from_numpy(np.array([float(r)])).type(dtype),requires_grad=False)

            if done == False:
                action = chooseAction(bestAction.data.cpu().numpy()[0], greedyFrac(maxGreedyFrac, timeToMax, totalFrameCounter))
                newObs, newR, newDone, newInfo = env.step(action)
                newX = buildX(newObs)
                newBestCost, newBestAction = torch.max(forward(newX, w1, w2,b1, b2),1)

            else:
                newObs = np.array([0,0,0,0])
                newR = 0
                newDone = True
                newInfo = {}


                   
            
            if len(replayMemory) < N:
    
                x = buildX(obs)
                y_pred = forward(x, w1, w2,b1,b2)[0,action]

                if done == False:
                    newX = buildX(newObs)
                    newBestCostTarget, newBestActionTarget = torch.max(forward(newX, w1Fixed, w2Fixed,b1Fixed, b2Fixed),1)
                    #newBestCost, newBestAction = torch.max(forward(newX, w1, w2,b1, b2),1)

                    y += discount * newBestCostTarget

                loss = (y - y_pred).pow(2)
            
            else:

                trainingMemories = np.random.choice(replayMemory, N-1, replace=False)
                obsList = [trainingMemories[n]['obs'] for n in range(N-1)]
                obsList.append(obs)
                x = buildX(np.array(obsList), N)

                actionList = [actionOneHot(trainingMemories[n]['action']) for n in range(N-1)]
                actionList.append(actionOneHot(action))
                vActionList = Variable(torch.from_numpy(np.array(actionList)).type(dtype), requires_grad=False)


                y_pred = actionMask(forward(x, w1, w2, b1, b2),vActionList)
                

                #yList = [trainingMemories[n]['y'] for n in range(N-1)]
                #yList.append(y)
                #vYList = torch.cat(tuple(yList),0)

                rList = [trainingMemories[n]['r'] for n in range(N-1)]
                rList.append(r)
                vRList = Variable(torch.from_numpy(np.array(rList)).type(dtype), requires_grad=False)
                
                newObsList = [trainingMemories[n]['newObs'] for n in range(N-1)]
                newObsList.append(newObs)
                newX = buildX(np.array(newObsList),N)
                #newBestCostList, newBestActionList = torch.max(forward(newX, w1, w2, b1, b2), 1)

                newBestCostTargetList, newBestActionTargetList = torch.max(forward(newX, w1Fixed, w2Fixed, b1Fixed, b2Fixed), 1)
                #newBestCost = newBestCostList[-1]
                #newBestAction = newBestActionList[-1]
                
                doneList = [1 if trainingMemories[n]['done'] == False else 0 for n in range(N-1)]
                if done == False:
                    doneList.append(1)
                else:
                    doneList.append(0)
                vDoneList = Variable(torch.from_numpy(np.array(doneList)).type(dtype), requires_grad=False)

                yList = vRList + discount * newBestCostTargetList * vDoneList



                loss = (yList - y_pred).pow(2).sum()
            

            episodeAvgLoss += loss.data[0]
            loss.backward()



            w1.data -= step_Size * w1.grad.data
            w2.data -= step_Size * w2.grad.data

            b1.data -= step_Size * b1.grad.data
            b2.data -= step_Size * b2.grad.data
            
            w1.grad.data.zero_()
            w2.grad.data.zero_()

            b1.grad.data.zero_()
            b2.grad.data.zero_()
            
            state = {'obs' : obs, 'action' : action, 'r' : r, 'done' : done, 'newObs' : newObs }
            if totalFrameCounter < maxMemorySize:
                replayMemory.append(state)
            else:
                memPos = totalFrameCounter % maxMemorySize
                replayMemory[memPos] = state
            

            totalFrameCounter += 1

            if totalFrameCounter % fixedWUpdate == 0:
                w1Fixed.data = w1.data
                w2Fixed.data = w2.data
                b1Fixed.data = b1.data
                b2Fixed.data = b2.data


            if done == False:
                obs, r, done, info, bestCost, bestAction = newObs, newR, newDone, newInfo, newBestCost, newBestAction
            else:
                endTime = t
                break

        rewards[e] = episodeReward
        if e % renderEpisode == 0:
            episodeAvgLoss = episodeAvgLoss / endTime
            print("Episode {} total reward: {}, avg loss: {:.4f}".format(e, episodeReward, episodeAvgLoss))


    plt.plot(rewards)
    plt.show()

