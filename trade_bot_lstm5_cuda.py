##############################################################################################################
#
# Futures trading environment
# Env: Long and short positions. Only one position at a time, includes MACD and RSI indicator history values
# NN: 2 layer LSTM network, MSE error
# Ties reward to the total portfolio value at each time step
###############################################################################################################
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import copy

USE_CUDA        = True
LR              = 0.001
NUM_EPISODES    = 100
HIDDEN_SIZE     = 100
NUM_FEATURES    = 90 * 3
INPUT_SIZE      = 90 * 3 + 2 # includes price, MACD and RSI histories. Current trade profits, total profits.
OUTPUT_SIZE     = 3

torch.autograd.set_detect_anomaly(True)

np.random.seed(9)

class Environment:
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

        self.trades = []
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.total_value = 0
        self.history = [0 for _ in range(self.history_t)]
        self.macd_history = [0 for _ in range(self.history_t)]
        self.rsi_history = [0 for _ in range(self.history_t)]
        self.trades = []
        self.valuation  = []
        
        return [self.position_value, self.total_value] + self.history + self.macd_history + self.rsi_history
    
    def step(self, action):
        reward = 0
        
        # action = 0: stay, 1: buy, 2: sell
        if action == 1 and len(self.positions) == 0:
            self.positions.append(self.data.iloc[self.t, :]['Close'])
            
            # Log trade
            trade = {}
            trade["date"] = self.data.index[self.t]
            trade["trade"] = "buy"
            trade["entry_price"] = self.data.iloc[self.t, :]['Close']
            trade["units"] = 1
            self.trades.append(trade)
                
        elif action == 2: # sell
            if len(self.positions) == 0:
                # If no long positions are held, sell short
                self.positions.append(-1 * self.data.iloc[self.t, :]['Close'])
            
                # Log trade
                trade = {}
                trade["date"] = self.data.index[self.t]
                trade["trade"] = "sell short"
                trade["entry_price"] = -1 * self.data.iloc[self.t, :]['Close']
                trade["units"] = 1
                self.trades.append(trade)
            else:
                # Close out existing positions
                profits = 0
                units = 0
                for p in self.positions:
                    if p < 0: 
                        profits += (-p - self.data.iloc[self.t, :]['Close'])    
                    else:
                        profits += (self.data.iloc[self.t, :]['Close'] - p)
                    units += 1
                
                reward += profits
                
                self.profits += profits
                self.positions = []

                # Log trade
                trade = {}
                trade["date"] = self.data.index[self.t]
                trade["trade"] = "close open position"
                trade["units"] = units
                trade["exit_price"] = self.data.iloc[self.t, :]['Close']
                trade["profit"] = profits
                
                self.trades.append(trade)
        
        # Update valuation
        self.total_value = self.profits + self.position_value
        
        val = {}
        val["Date"] = self.data.index[self.t]
        val["Realized"] = self.profits
        val["Unrealized"] = self.position_value
        val["Total Valuation"] = self.total_value

        self.valuation.append(val)
        
        # set next time
        self.t += 1
        
        self.position_value = 0
        for p in self.positions:
            if p < 0:
                self.position_value += (-p - self.data.iloc[self.t, :]['Close'])
            else:
                self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close'])

        if self.t > 20:
            self.macd_history.pop(0)
            self.rsi_history.pop(0)
            self.macd_history.append(self.data.iloc[self.t, :]['MACD'])
            self.rsi_history.append(self.data.iloc[self.t, :]['RSI'])
        
        for p in self.positions:
            if p < 0 and (self.data.iloc[self.t, :]['Close'] > 1.05 * abs(p)):
                stop_loss = True
                profits = 0
                units = 0
            
                if p < 0: 
                    profits += (-p - self.data.iloc[self.t, :]['Close'])    
                else:
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                units += 1
                reward += profits
                self.profits += profits
            
                # Log trade
                trade = {}
                trade["date"] = self.data.index[self.t]
                trade["trade"] = "Stop loss"
                trade["units"] = units
                trade["exit_price"] = self.data.iloc[self.t, :]['Close']
                trade["profit"] = profits

                self.trades.append(trade)

                # Reset metrics
                self.positions = []
                self.position_value = 0
                
            elif p > 0 and (self.data.iloc[self.t, :]['Close'] < 0.95 * abs(p)):
                stop_loss = True
                
                profits = 0
                units = 0
            
                if p < 0: 
                    profits += (-p - self.data.iloc[self.t, :]['Close'])    
                else:
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                units += 1
                reward += profits
                self.profits += profits
            
                # Log trade
                trade = {}
                trade["date"] = self.data.index[self.t]
                trade["trade"] = "Stop loss"
                trade["units"] = units
                trade["exit_price"] = self.data.iloc[self.t, :]['Close']
                trade["profit"] = profits

                self.trades.append(trade)

                # Reset metrics
                self.positions = []
                self.position_value = 0
                
        # Check if end of market data file, and end
        if (self.t == len(self.data)-1):
            self.done = True
        
        return [self.position_value, self.total_value] + self.history + self.macd_history + self.rsi_history, reward, self.done # obs, reward, done

class Q_Network(nn.Module):
    def __init__(self,obs_len,hidden_size,actions_n):
        super(Q_Network,self).__init__()
        
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE).cuda()
        # self.batch_norm = nn.BatchNorm1d(HIDDEN_SIZE).cuda()
        self.lstm = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE).cuda()
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE).cuda()

        self.fc_pi = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE).cuda()
        self.fc_v = nn.Linear(HIDDEN_SIZE, 1).cuda()

        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        
    def forward(self,x, hidden):
        x = F.relu(self.fc1(x))     
        x = x.view(-1, 1, HIDDEN_SIZE)
        x, lstm_hidden = self.lstm(x, hidden)
        x = F.relu(self.fc2(x))

        pi = self.fc_pi(x)
        pi = F.softmax(pi, dim=2)
        v = self.fc_v(x)
        return pi, v, lstm_hidden
        
class TradeBot():
    def __init__(self, data_fp, mode, out_fn):
        self.mode = mode
        self.trades = []
        self.out_fn = out_fn
        
        #########################################
        # Load and pre-process data
        #########################################
        data = pd.read_csv(data_fp)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        data.fillna(method='ffill', inplace=True)

        # Add Exponential Moving
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] =data['EMA_12'] - data['EMA_26']

        # Add RSI: https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
        close = data['Close']
        delta = close.diff()
        delta = delta[1:] 
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = up.ewm(span = 20).mean()
        roll_down1 = down.abs().ewm(span = 20).mean()
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        data['RSI'] = RSI1
        
        date_split = '2019-07-01'
        self.train_data = data[:date_split]
        self.test_data = data[date_split:]

        if mode == "train":
            self.train()
        elif mode == "test":
            self.test()

    def train(self):
        env = Environment(self.train_data)
        
        Q = Q_Network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        Q_ast = copy.deepcopy(Q)

        if USE_CUDA:
            Q = Q.cuda()
        
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(list(Q.parameters()), lr=LR)
        
        epoch_num = NUM_EPISODES
        step_max = len(env.data) - 1
        memory_size = 200
        batch_size = 50
        gamma = 0.97

        obs, reward, done = env.step(5)

        memory = []
        total_step = 0
        total_rewards = []
        total_losses = []
        epsilon = 1.0
        epsilon_decrease = 1e-3
        epsilon_min = 0.1
        start_reduce_epsilon = 200
        train_freq = 10
        update_q_freq = 20
        gamma = 0.97
        
        episode_rewards = []
        episode_losses = []
        avg_rewards = []
        avg_losses = []
        
        for epoch in range(epoch_num):
            start_time = time.time()
            
            h_out = (torch.zeros([1, 1, HIDDEN_SIZE], dtype=torch.float32).cuda(),
                     torch.zeros([1, 1, HIDDEN_SIZE],  dtype=torch.float32).cuda())

            pobs = env.reset()
            step = 0
            done = False
            total_reward = 0
            total_loss = 0

            while not done and step < step_max:
                h_in = h_out
                h_in_copy = h_out
                
                # select action
                pact = np.random.randint(3)
                if np.random.rand() > epsilon:
                    pact, v, _ = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)).cuda(), h_in)
                    
                    pact = np.argmax(pact.data.cpu())
                    pact = pact.numpy()

                # act
                obs, reward, done = env.step(pact)

                # add memory
                memory.append((pobs, pact, reward, obs, done))
                
                if len(memory) > memory_size:
                    memory.pop(0)

                # train or update q
                if len(memory) == memory_size:
                    if total_step % train_freq == 0:
                        shuffled_memory = np.random.permutation(memory)
                        memory_idx = range(len(shuffled_memory))
                        for i in memory_idx[::batch_size]:
                            batch = np.array(shuffled_memory[i:i+batch_size])
                            b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                            b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                            b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                            b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                            b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                            q, v, h_in_next = Q(torch.from_numpy(b_pobs).cuda(), h_in)
                            q_, v_, h_in_copy = Q_ast(torch.from_numpy(b_obs).cuda(), h_in_copy)
                            
                            q_ = torch.squeeze(q_, dim=1)
                            
                            maxq = np.max(q_.data.cpu().numpy(),axis=1)
                            
                            target = copy.deepcopy(q.data.cpu().numpy())
                            target = np.squeeze(target, axis=1)
                            
                            for j in range(batch_size):
                                target[j, b_pact[j]] = b_reward[j] + gamma * maxq[j] * (not b_done[j])
                            
                            Q.zero_grad()
                            
                            target = torch.from_numpy(target).cuda()
                            
                            q = torch.squeeze(q, dim=1)
                            
                            # loss = loss_function(q, target)
                            loss = F.smooth_l1_loss(q, target)
                            
                            total_loss += loss.data.item()
                            loss.backward(retain_graph=True)
                            optimizer.step()
                            
                    if total_step % update_q_freq == 0:
                        Q_ast = copy.deepcopy(Q)
                        
                    # epsilon
                    if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                        epsilon -= epsilon_decrease

                    # next step
                    total_reward += reward
                    pobs = obs
                    step += 1
                    total_step += 1

                total_rewards.append(total_reward)
                total_losses.append(total_loss)
            
            ########################################
            # Display episode metrics
            ########################################
            episode_rewards.append(total_reward)
            episode_losses.append(total_loss)
            
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)

            avg_loss = np.mean(episode_losses[-100:])
            avg_losses.append(avg_loss)
            
            end_time = time.time()

            total_time = end_time - start_time
            
            print("Epoch: ", epoch, "Episode reward: ", total_reward, "Episode loss: ", total_loss, "Running avg loss: ", avg_loss, " Running avg reward: ", avg_reward, " Steps: ", step, " Time: ", total_time)

        ########################################
        # Save model and metrics
        ########################################
        torch.save(Q.state_dict(), 'lstm5_model_params')

        out_pd = pd.DataFrame(avg_rewards, columns=["Rolling Avg Reward"])
        out_pd["Rolling Avg Loss"] = avg_losses
        out_pd["Episode Reward"] = episode_rewards
        out_pd["Episode Loss"] = episode_losses
        
        out_pd.to_excel('./output/' + self.out_fn + '_output.xlsx')

        trades_pd = pd.DataFrame(env.trades)
        trades_pd.to_excel("./output/" + self.out_fn + "_train_trades.xlsx")
        

    def test(self):
        h_in = (torch.zeros([1, 1, HIDDEN_SIZE], dtype=torch.float32).cuda(),
                 torch.zeros([1, 1, HIDDEN_SIZE],  dtype=torch.float32).cuda())
        
        test_env = Environment(self.test_data)
        
        # Load saved model
        Q = Q_Network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        Q.load_state_dict(torch.load('lstm5_model_params'))
        
        pobs = test_env.reset()
        test_acts = []
        test_rewards = []

        for _ in range(len(test_env.data)-1):
            # pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
            Q.eval()
            pact, v, _ = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)).cuda(), h_in)
            
            pact = np.argmax(pact.data.cpu())
            test_acts.append(pact.item())
            
            obs, reward, done = test_env.step(pact.numpy())
            test_rewards.append(reward)

            pobs = obs
        
        test_profits = test_env.profits
        
        trades_pd = pd.DataFrame(test_env.trades)
        trades_pd.to_excel("./output/" + self.out_fn + "_test_trades.xlsx")
        
        valuation = pd.DataFrame(test_env.valuation)
        valuation.to_excel("./output/" + self.out_fn + "_valuation.xlsx")
        
        print("Test profits: ", test_profits)
        
if __name__ == "__main__":
    data_fp = sys.argv[1]
    out_fn = sys.argv[2]
    mode = sys.argv[3]
    
    trade_bot = TradeBot(data_fp, mode, out_fn)