import pandas as pd
import json
import os
from sodapy import Socrata

key, username, password = os.environ["API_KEY"], os.environ["API_USERNAME"], os.environ["API_PASSWORD"]

class SoPy:
    def __init__(self, 
                key=key,
                username=username,
                password=password):

        self.key=key 
        self.username=username
        self.password=password
        self.client = Socrata("data.cityofnewyork.us", 
                    self.key,
                    self.username,
                    self.password
                    )
    def fetch(self, endpoint, **kwargs):
        results = self.client.get(endpoint, **kwargs)
        # https://data.cityofnewyork.us/resource/nu7n-tubp.json
        return results

    def results_df(self, result):
        results_df = pd.DataFrame.from_records(result)
        return results_df

if __name__ == "__main__":
    sp = SoPy()
    res = sp.fetch("nu7n-tubp")
    res_df = sp.results_df(res)
    print(res_df)