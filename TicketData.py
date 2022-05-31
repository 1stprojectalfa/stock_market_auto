import time
import datetime
import pandas as pd

class TicketData():

    def __init__(self, name, initial_date, final_date, sampling_frequency):
        """
        date = (year, month, day)
        sampling frequency = 1mo or 1w or 1d
        """
        self.name = name
        self.initial_year, self.initial_month, self.initial_day = initial_date
        self.final_year, self.final_month, self.final_day = final_date
        self.sampling_frequency = sampling_frequency #1d, 1wk, 1mo
        
    def standarizing_sampling_period(self):
        """
        Standarizing dates from (YYYY, MM, DD) to seconds from initial Unix time
        """
        self.initial_date_standard = int(time.mktime(datetime.datetime(self.initial_year, self.initial_month, self.initial_day, 23, 59).timetuple()))
        self.final_date_standard = int(time.mktime(datetime.datetime(self.final_year, self.final_month, self.final_day, 23, 59).timetuple()))

    def request_data(self):
        """
        Sending request to Yahoo Finance Website
        """
        request_string_1 = f'https://query1.finance.yahoo.com/v7/finance/download/{self.name}?period1={self.initial_date_standard}'
        request_string_2 = f'&period2={self.final_date_standard}&interval={self.sampling_frequency}&events=history&includeAdjustedClose=true'
        request_string = request_string_1 + request_string_2
        df = pd.read_csv(request_string)
        print(df)

tickers = ['AMD', 'FB', 'SPY', 'TSLA']

for ticker in tickers:
    time.sleep(5)
    print(ticker)
    objeto = TicketData(ticker, (2021,12,1), (2021,12,22), "1wk")
    objeto.standarizing_sampling_period()
    objeto.request_data()
    print("Done")