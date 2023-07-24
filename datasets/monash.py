"""
Taken from https://github.com/rakshitha123/TSForecasting/
"""
from datetime import datetime
from types import SimpleNamespace
from numpy import distutils
import torch
import os
from torch.utils.data import Dataset
import distutils
import pandas as pd
import numpy as np
import bisect
from torch.utils.data import WeightedRandomSampler

# Mapping from
# {dataset_name: (minimum time lag, filename, forecast horizon, integer outputs, val mode)}
datasets = {
    "nn5_daily": (9, "nn5_daily_dataset_without_missing_values.tsf", None, False, "strict"),
    "nn5_weekly": (65, "nn5_weekly_dataset.tsf", 8, False, "strict"),
    "tourism_yearly": (2, "tourism_yearly_dataset.tsf", None, False, "overlap"),
    "tourism_quarterly": (5, "tourism_quarterly_dataset.tsf", None, False, "strict"),
    "tourism_monthly": (15, "tourism_monthly_dataset.tsf", None, False, "strict"),
    "m1_yearly": (2, "m1_yearly_dataset.tsf", None, False, "strict"),
    "m1_quarterly": (5, "m1_quarterly_dataset.tsf", None, False, "strict"),
    "m1_monthly": (15, "m1_monthly_dataset.tsf", None, False, "strict"),
    "m3_yearly": (2, "m3_yearly_dataset.tsf", None, False, "strict"),
    "m3_quarterly": (5, "m3_quarterly_dataset.tsf", None, False, "strict"),
    "m3_monthly": (15, "m3_monthly_dataset.tsf", None, False, "strict"),
    "m3_other": (2, "m3_other_dataset.tsf", None, False, "strict"),
    "m4_quarterly": (5, "m4_quarterly_dataset.tsf", None, False, "strict"),
    "m4_monthly": (15, "m4_monthly_dataset.tsf", None, False, "strict"),
    "m4_weekly": (65, "m4_weekly_dataset.tsf", None, False, "strict"),
    "m4_daily": (9, "m4_daily_dataset.tsf", None, False, "strict"),
    "m4_hourly": (210, "m4_hourly_dataset.tsf", None, False, "strict"),
    "car_parts": (15, "car_parts_dataset_without_missing_values.tsf", 12, True, "strict"),
    "hospital": (15, "hospital_dataset.tsf", 12, True, "strict"),
    "fred_md": (15, "fred_md_dataset.tsf", 12, False, "strict"),
    "traffic_weekly": (65, "traffic_weekly_dataset.tsf", 8, False, "strict"),
    "traffic_hourly": (30, "traffic_hourly_dataset.tsf", 168, False, "strict"),
    "electricity_weekly": (65, "electricity_weekly_dataset.tsf", 8, True, "strict"),
    "electricity_hourly": (30, "electricity_hourly_dataset.tsf", 168, True, "strict"),
    "solar_weekly": (6, "solar_weekly_dataset.tsf", 5, False, "strict"),
    "kaggle_web_traffic_weekly": (10, "kaggle_web_traffic_weekly_dataset.tsf", 8, True, "strict"),
    "dominick": (10, "dominick_dataset.tsf", 8, False, "strict"),
    "us_births": (9, "us_births_dataset.tsf", 30, True, "strict"),
    "saugeen_river_flow": (9, "saugeenday_dataset.tsf", 30, False, "strict"),
    "sunspot": (9, "sunspot_dataset_without_missing_values.tsf", 30, True, "strict"),
    "covid_deaths": (9, "covid_deaths_dataset.tsf", 30, True, "strict"),
    "weather": (9, "weather_dataset.tsf", 30, False, "strict"),
    "solar_10_minutes": (50, "solar_10_minutes_dataset.tsf", 1008, False, "strict"),
    "kdd_cup": (210, "kdd_cup_2018_dataset_without_missing_values.tsf", 168, False, "strict"),
    "melbourne_pedestrian_counts": (210, "pedestrian_counts_dataset.tsf", 24, True, "strict"),
    "bitcoin": (9, "bitcoin_dataset_without_missing_values.tsf", 30, False, "strict"),
    "vehicle_trips": (9, "vehicle_trips_dataset_without_missing_values.tsf", 30, True, "strict"),
    "aus_elecdemand": (420, "australian_electricity_demand_dataset.tsf", 336, False, "strict"),
    "rideshare": (168, "rideshare_dataset_without_missing_values.tsf", 168, False, "overlap"), # lag used to be 210, was too long to create single train example
    "temperature_rain": (9, "temperature_rain_dataset_without_missing_values.tsf", 30, False, "strict"),
}

BASE_DIR = "TSForecasting"
VALUE_COL_NAME = "series_value"
TIME_COL_NAME = "start_timestamp"


# Seasonality values corresponding with the frequencies: minutely, 10_minutes, half_hourly, hourly, daily, weekly, monthly, quarterly and yearly
# Consider multiple seasonalities for frequencies less than daily
SEASONALITY_MAP = {
   "minutely": [1440, 10080, 525960],
   "10_minutes": [144, 1008, 52596],
   "half_hourly": [48, 336, 17532],
   "hourly": [24, 168, 8766],
   "daily": 7,
   "weekly": 365.25/7,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

# Frequencies used by GluonTS framework
FREQUENCY_MAP = {
   "minutely": "1min",
   "10_minutes": "10min",
   "half_hourly": "30min",
   "hourly": "1H",
   "daily": "1D",
   "weekly": "1W",
   "monthly": "1M",
   "quarterly": "1Q",
   "yearly": "1Y"
}

# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with = 'NaN', value_column_name = "series_value"):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"): # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (len(line_content) != 3):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(distutils.util.strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(distutils.util.strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception("Missing attribute section. Attribute section must come before data.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception("Missing attribute section. Attribute section must come before data.")
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if(len(series) == 0):
                            raise Exception("A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol")

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if (numeric_series.count(replace_missing_vals_with) == len(numeric_series)):
                            raise Exception("All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series.")

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], '%Y-%m-%d %H-%M-%S')
                            else:
                                raise Exception("Invalid attribute type.") # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if(att_val == None):
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length


def load_data(data_dir, dataset_name):

    min_lag, input_file_name, external_forecast_horizon, integer_conversion, val_mode = datasets[dataset_name]
    print(f"{data_dir}/{input_file_name}")
    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(f"{data_dir}/{input_file_name}")

    # train_series_full_list = []
    # test_series_full_list = []
    train_series_list = []
    test_series_list = []
    full_series_list = []
    train_start_times = []

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1

    if isinstance(seasonality, list):
        seasonality = min(seasonality) # Use to calculate MASE

    # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
    if forecast_horizon is None:
        forecast_horizon = external_forecast_horizon

    for index, row in df.iterrows():
        if TIME_COL_NAME in df.columns:
            train_start_time = row[TIME_COL_NAME]
        else:
            train_start_time = datetime.strptime('1900-01-01 00-00-00', '%Y-%m-%d %H-%M-%S') # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False

        series_data = row[VALUE_COL_NAME]

        # Creating training and test series. Test series will be only used during evaluation
        train_series_data = series_data[:len(series_data) - forecast_horizon]
        test_series_data = series_data[(len(series_data) - forecast_horizon) : len(series_data)]

        train_series_list.append(train_series_data)
        test_series_list.append(test_series_data)
        full_series_list.append(series_data)

        # We use full length training series to train the model as we do not tune hyperparameters
        # train_series_full_list.append((train_series_data, pd.Timestamp(train_start_time, freq=freq)))
        # test_series_full_list.append((series_data, pd.Timestamp(train_start_time, freq=freq)))

        train_start_times.append(pd.Timestamp(train_start_time, freq=freq))

    return SimpleNamespace(
        # df=df,
        # train_series_full_list=train_series_full_list,
        # test_series_full_list=test_series_full_list,
        train_series_list=train_series_list,
        test_series_list=test_series_list,
        full_series_list=full_series_list,
        train_start_times=train_start_times,
        forecast_horizon=forecast_horizon,
        freq=freq,
        seasonality=seasonality,
        min_lag=min_lag,
        contain_missing_values=contain_missing_values,
        contain_equal_length=contain_equal_length,
        integer_conversion=integer_conversion,
        val_mode=val_mode
    )


class _MonashDataset(Dataset):

    TIME_FEATURES = {
        'day',
        'hour',
        'minute',
        'second',
        'month',
        'year',
        # 'dayofweek',
        # 'dayofyear',
        'quarter',
        'week',
        # 'is_month_start',
        # 'is_month_end',
        # 'is_quarter_start',
        # 'is_quarter_end',
        # 'is_year_start',
        # 'is_year_end',
        # 'is_leap_year',
    }

    def split_time_series(self, ts, ts_times, k: int, val_frac: float, nval):
        assert k >= 0, "Skip length `k` must be atleast 1. Denotes the length of the gap between the beginning of the last train example and the first val example."
        # Calculate the effective number of (min_lag, forecast horizon) examples that are present in every time series
        total_examples_no_validation = len(ts) - self.forecast_horizon - self.min_lag + 1
        # Validation causes us to skip over some examples so that there's less overlap between train and validation
        # The parameter k controls how many examples we skip
        # e.g. k = 1, means we skip no examples at all and the last train and first validation examples maximally overlap at all but one point
        # TODO: check for k = F
        total_examples_with_validation = total_examples_no_validation - (k - 1)

        # Keep some fraction for validation
        if nval is None:
            nval = int(val_frac * total_examples_with_validation)
        else:
            # if nval is specified, make sure its not more than half the examples
            nval = min(nval, int(0.5*total_examples_with_validation))

        if nval == 0 and k > 0:
            return None

        if k == 0:
            if self.split == 'train':
                return SimpleNamespace(
                    total_examples_no_validation=total_examples_no_validation,
                    total_examples_with_validation=-1,
                    nval=0,
                    data=ts[:-self.forecast_horizon], 
                    target=ts[self.min_lag:], 
                    data_times=ts_times[:-self.forecast_horizon], 
                    target_times=ts_times[self.min_lag:],
                )
            elif self.split == 'val':
                return SimpleNamespace(
                    total_examples_no_validation=total_examples_no_validation,
                    total_examples_with_validation=-1,
                    nval=0,
                    data=[],
                    target=[],
                    data_times=[],
                    target_times=[],
                )
        
        if self.split == 'train':
            data_endpoint = -(self.forecast_horizon + k + (nval - 1))

            target_startpoint = self.min_lag
            target_endpoint = data_endpoint + self.forecast_horizon

            data = ts[:data_endpoint]
            target = ts[target_startpoint:target_endpoint]

            data_times = ts_times[:data_endpoint]
            target_times = ts_times[target_startpoint:target_endpoint]
        elif self.split == 'val':
            data_startpoint = -(self.forecast_horizon + self.min_lag + (nval - 1))
            data_endpoint = -(self.forecast_horizon)

            target_startpoint = data_startpoint + self.min_lag

            data = ts[data_startpoint:data_endpoint]
            target = ts[target_startpoint:]

            data_times = ts_times[data_startpoint:data_endpoint]
            target_times = ts_times[target_startpoint:]
        
        return SimpleNamespace(
            total_examples_no_validation=total_examples_no_validation,
            total_examples_with_validation=total_examples_with_validation,
            nval=nval,
            data=data, 
            target=target, 
            data_times=data_times, 
            target_times=target_times,
        )


    def __init__(
        self, 
        dataset_name, 
        data_dir, 
        split, 
        lag=None, 
        lag_scale=None, 
        val_frac=None, 
        nval=None,
        val_k=None,
        data=None, 
        standardization=None,
        weighted_sampler=False, 
        save_processed=False,
        log_transform=False,
        autoregressive=False,
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.split = split
        self.autoregressive = autoregressive

        if data is None:
            data = load_data(data_dir, dataset_name)

        # self.df = data.df
        # self.train_series_full_list = data.train_series_full_list
        # self.test_series_full_list = data.test_series_full_list
        self.train_series_list = data.train_series_list
        self.test_series_list = data.test_series_list
        self.full_series_list = data.full_series_list
        self.train_start_times = data.train_start_times
        self.forecast_horizon = data.forecast_horizon
        self.freq = data.freq
        self.seasonality = data.seasonality
        self.min_lag = data.min_lag
        assert lag is None or lag_scale is None, "Only one of lag or lag_scale can be specified."
        self.lag = self.min_lag if lag is None else lag
        self.lag = int(self.lag * lag_scale) if lag_scale is not None else self.lag
        # assert self.lag >= self.min_lag, "Lag must be atleast min_lag."
        print(f"Lag is {self.lag}, min lag is {self.min_lag}.")
        self.contain_missing_values = data.contain_missing_values
        self.contain_equal_length = data.contain_equal_length
        self.integer_conversion = data.integer_conversion
        self.val_mode = data.val_mode # "strict" or "overlap", strict makes sure y_train and y_val do not overlap
        self.log_transform = log_transform
        
        # Timestamps for all time series
        self.train_series_times = [pd.date_range(start=start_time, periods=len(series), freq=self.freq) for start_time, series in zip(self.train_start_times, self.train_series_list)]
        self.full_series_times = [pd.date_range(start=start_time, periods=len(series), freq=self.freq) for start_time, series in zip(self.train_start_times, self.full_series_list)]
        
        if val_frac is None: val_frac = 0.0
        
        if save_processed: 
            processed_filename = f'processed_{self.split}_valfrac_{int(val_frac * 100)}.npy' if self.split != 'test' else f'processed_test.npy'
        
        if save_processed and os.path.exists(os.path.join(data_dir, processed_filename)):
            self.data, self.data_times, self.target, self.target_times, self.ts_indices = np.load(os.path.join(data_dir, processed_filename), allow_pickle=True)
        else:
            if self.split != 'test':
                # which time-series are kept in the dataset: only relevant for validation, where some TS might not be long enough to pull out validation samples 
                self.ts_indices = []
                self.data, self.data_times, self.target, self.target_times = [], [], [], []
                for i, (ts, ts_times) in enumerate(zip(self.train_series_list, self.train_series_times)):
                    if val_k:
                        ts_split_info = self.split_time_series(ts, ts_times, val_k, val_frac, nval)
                        if ts_split_info is None:
                            continue
                    else:
                        for k in range(self.forecast_horizon, -1, -1):
                            ts_split_info = self.split_time_series(ts, ts_times, k, val_frac, nval)
                            if ts_split_info is None:
                                continue
                            if ts_split_info.nval > 0:
                                break
                    
                    if len(ts_split_info.data) > 0:
                        # Append the time series
                        self.data.append(ts_split_info.data)
                        self.target.append(ts_split_info.target)
                        self.data_times.append(ts_split_info.data_times)
                        self.target_times.append(ts_split_info.target_times)
                        self.ts_indices.append(i)

            elif self.split == 'test':
                # Just keep the end of the series for testing, in line with the Monash github repo
                self.data = [ts[-self.forecast_horizon - self.min_lag:-self.forecast_horizon] for ts in self.full_series_list]
                self.data_times = [ts[-self.forecast_horizon - self.min_lag:-self.forecast_horizon] for ts in self.full_series_times]
                self.target = [ts[-self.forecast_horizon:] for ts in self.full_series_list]
                self.target_times = [ts[-self.forecast_horizon:] for ts in self.full_series_times]
                self.ts_indices = np.arange(len(self.data))

            if save_processed:
                # Save all the data to disk
                np.save(os.path.join(data_dir, processed_filename), (self.data, self.data_times, self.target, self.target_times, self.ts_indices))
        
        # Number of examples in the split
        self.n_examples = [max(0, len(ts) - self.forecast_horizon + 1) for ts in self.target]
        self._cume = np.cumsum([0] + self.n_examples[:-1])

        # Log-transform data if required
        if self.log_transform:
            self.data = [np.log(ts) for ts in self.data]
            self.target = [np.log(ts) for ts in self.target]

        if self.split == 'train':
            assert len(self.ts_indices) == len(self.train_series_list), "Every time series must be in the train data."

            if standardization is None:
                # Calculate mean, std for normalization
                self.mean = [np.mean(ts) for ts in self.data]
                self.std = [np.std(ts) for ts in self.data]
            else:
               self.mean, self.std = [standardization[0]] * len(self.data), [standardization[1]] * len(self.data)

            if weighted_sampler:
                # idx in the __getitem__ of the train dataloader
                dl_idxs = np.arange(sum(self.n_examples))
                # extract the time-series idx (named loc) from the dataloader idx
                dl_locs = [bisect.bisect(self._cume, idx) - 1 for idx in dl_idxs]
                # each idx will have a weight inversely proportional to its corresponding time-series length
                self.sampler_weights = [1/self.n_examples[loc] for loc in dl_locs]

        elif self.split == 'val':
            self.mean, self.std = standardization
            self.mean = [self.mean[i] for i in self.ts_indices]
            self.std = [self.std[i] for i in self.ts_indices]
        elif self.split == 'test':
            self.mean, self.std = standardization

    def __len__(self):
        return sum(self.n_examples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError('Index out of range')
        # Bisect to find the time series to pull out
        loc = bisect.bisect(self._cume, idx) - 1

        context = self.data[loc][max(0, idx - self._cume[loc] - (self.lag - self.min_lag)) : idx - self._cume[loc] + self.min_lag]
        context_time = self.data_times[loc][max(0, idx - self._cume[loc] - (self.lag - self.min_lag)) : idx - self._cume[loc] + self.min_lag]

        target = self.target[loc][idx - self._cume[loc]: idx - self._cume[loc] + self.forecast_horizon]
        target_time = self.target_times[loc][idx - self._cume[loc]: idx - self._cume[loc] + self.forecast_horizon]

        # Normalize
        context = (context - self.mean[loc]) / self.std[loc]
        target = (target - self.mean[loc]) / self.std[loc]

        # To tensors
        context, target = torch.tensor(context, dtype=torch.float).unsqueeze(1), torch.tensor(target, dtype=torch.float).unsqueeze(1)

        # Prepend context to the beginning of the target
        # target = torch.cat([context, target], dim=0)
        actual_lag = context.shape[0]

        if not self.autoregressive:
            # Add zeros to the end of the context with length equal to the forecast horizon
            context = torch.cat([torch.zeros(self.lag - actual_lag, 1), context, torch.zeros(self.forecast_horizon, 1)], dim=0)
        time = context_time.union(target_time)
        
        return context, target, self._timestamp_to_features(time, pad=(self.lag - actual_lag)), loc
       
    def _timestamp_to_features(self, timestamp, pad=0):
        if pad == 0:
            return {k: torch.tensor(np.array(getattr(timestamp, k)), dtype=torch.long) + 1 for k in self.TIME_FEATURES}
        else:
            return {k: torch.cat([torch.zeros(pad, dtype=torch.long) - 1, torch.tensor(np.array(getattr(timestamp, k)), dtype=torch.long) + 1]) for k in self.TIME_FEATURES}



class Monash(Dataset):
    _name_ = "monash"

    init_defaults = {
        'dataset_name': 'aus_elecdemand',
        'val_frac': 0.2,
        'nval': None,
        'val_k': None,
        'lag': None,
        'lag_scale': None,
        'standardize': True,
        'weighted_sampler': True,
        'save_processed': False,
        'd_output': 1,
        'autoregressive': False,
        'log_transform': False,
    }

    def init(self):
        self.data_dir = self.data_dir


    @property
    def d_input(self):
        return 1
    
    @property
    def l_output(self):
        if self.autoregressive: return None
        return self.dataset_train.forecast_horizon

    @property
    def n_ts(self):
        return len(self.dataset_train.data)

    @property
    def L(self):
        if self.autoregressive: return self.dataset_train.min_lag
        return self.dataset_train.forecast_horizon + self.dataset_train.min_lag

    def setup(self):
        data = load_data(self.data_dir, self.dataset_name)

        self.dataset_train = _MonashDataset(
            self.dataset_name, 
            self.data_dir, 
            'train', 
            val_frac=self.val_frac, 
            nval = self.nval,
            val_k = self.val_k,
            data=data,
            lag=self.lag,
            lag_scale=self.lag_scale,
            save_processed=self.save_processed,
            standardization=None if self.standardize else (np.zeros(1), np.ones(1)),
            weighted_sampler=self.weighted_sampler,
            log_transform=self.log_transform,
            autoregressive=self.autoregressive
        )

        self.dataset_test = _MonashDataset(
            self.dataset_name,
            self.data_dir,
            'test',
            val_frac=self.val_frac,
            nval = self.nval,
            val_k = self.val_k,
            data=data,
            lag=self.lag,
            lag_scale=self.lag_scale,
            save_processed=self.save_processed,
            log_transform=self.log_transform,
            standardization=(self.dataset_train.mean, self.dataset_train.std),
            autoregressive=self.autoregressive
        )

        if self.val_frac > 0. or self.nval is not None:
            self.dataset_val = _MonashDataset(
                self.dataset_name,
                self.data_dir,
                'val',
                val_frac=self.val_frac,
                nval = self.nval,
                val_k = self.val_k,
                data=data,
                lag=self.lag,
                lag_scale=self.lag_scale,
                save_processed=self.save_processed,
                log_transform=self.log_transform,
                standardization=(self.dataset_train.mean, self.dataset_train.std),
                autoregressive=self.autoregressive
            )
        else:
            self.dataset_val = self.dataset_test


    @staticmethod
    def collate_fn(batch, resolution, **kwargs):
        x, y, *z = zip(*batch)
        x = torch.stack(x, dim=0)[:, ::resolution]
        y = torch.stack(y, dim=0)[:, ::resolution]
        time, ids = z
        time = {k: torch.stack([e[k] for e in time], dim=0)[:, ::resolution] for k in time[0].keys()}
        ids = torch.tensor(ids)
        ## z = [torch.stack(e, dim=0)[:, ::resolution] for e in z]
        return x, y, time, ids

    def train_dataloader(self, **kwargs):
        if self.weighted_sampler:
            sampler = WeightedRandomSampler(self.dataset_train.sampler_weights, len(self.dataset_train.sampler_weights))
            kwargs['sampler'] = sampler
            #kwargs['shuffle'] = False # because we define custom sampler
        return super().train_dataloader(**kwargs)

    def val_dataloader(self, **kwargs):
        # Shuffle the val dataloader so we get random forecast horizons!
        kwargs['shuffle'] = True
        kwargs['drop_last'] = False
        return super().val_dataloader(**kwargs)

    def test_dataloader(self, **kwargs):
        kwargs['drop_last'] = False
        return super().test_dataloader(**kwargs)

if __name__ == '__main__':
  dataset_name = "nn5_weekly"
  print(f'Processing: {dataset_name}')
  data_dir = f'/atlas/u/winniexu/monash/{dataset_name}/'
  data = load_data(data_dir, dataset_name)
  k = 1
  breakpoint()
  train_ds, test_ds = data.train_series_list[k].to_numpy(), data.test_series_list[k].to_numpy()
  print("train_ds: ", train_ds.shape, "test ds: ", test_ds.shape)
