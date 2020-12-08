# DeepTrader: Deep Reinforcement Learning TradeBot

## Uses a DQN Network to train an agent to buy/short/hold positions in the futures markets.

Versions of the agent:
- trade_bot_cuda.py: Long Only, Fully Connected Layer networks
- trade_bot_fut_cuda.py: Long/Short, Fully Connected Layer networks
- trade_bot_lstm_5_cuda.py: Long/Short, LSTM with Stop Loss and Full Trade Profit Reward
- trade_bot_lstm_6_cuda.py: Long/Short, LSTM 

All data input files are located in the ./data folder. 
Command arguments:
1. Input data file path
2. Output data file path
3. Mode: [train] to train the agent and save the model params. [test] to test the agent on the previously saved model params.


To run:
```
python .\trade_bot_lstm5_cuda.py .\data\RTY=F.csv rty_lstm5 train
python .\trade_bot_lstm5_cuda.py .\data\RTY=F.csv rty_lstm5 test
```