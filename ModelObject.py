from TicketData import TicketData
from time import sleep

tickers = ['AMD', 'FB', 'SPY', 'TSLA']

for ticker in tickers:
    sleep(5)
    print(ticker)
    objeto = TicketData(ticker, (2021,12,1), (2021,12,22), "1wk")
    objeto.standarizing_sampling_period()
    objeto.request_data()
    print(objeto.data_frame)
    print("Done")