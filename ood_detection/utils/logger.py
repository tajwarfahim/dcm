import numpy as np
import pandas as pd


class Logger:
    def __init__(self, log_location):
        self.log_location = log_location
        self.logs = {}

    def _add_to_log(self, field_name, field_value):
        if field_name not in self.logs:
            self.logs[field_name] = []

        self.logs[field_name].append(field_value)

    def log(self, log_dict):
        for field_name in log_dict:
            self._add_to_log(field_name=field_name, field_value=log_dict[field_name])

    def flush(self):
        df = pd.DataFrame(data=self.logs)
        df.to_csv(self.log_location)
