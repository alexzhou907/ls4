import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from scipy import special
from .utils import split_and_subsample_batch, normalize_masked_data

class Climate(Dataset):
    """
    Dataset class for ODE type of data. With 2 values.
    Can be fed with either a csv file containg the dataframe or directly with a panda dataframe.
    One can further provide samples idx that will be used (for training / validation split purposes.)
    """

    def __init__(self, csv_file=None, cov_file=None, label_file=None, panda_df=None, cov_df=None, label_df=None,
                 root_dir="./", t_mult=1.0, idx=None, jitter_time=0, validation=False, val_options=None):
        """
        Args:
            csv_file   CSV file to load the dataset from
            panda_df   alternatively use pandas df instead of CSV file
            root_dir   directory of the CSV file
            t_mult     multiplier for time values (1.0 default)
            jitter_time  jitter size (0 means no jitter), to add randomly to Time.
                         Jitter is added before multiplying with t_mult
            validation boolean. True if this dataset is for validation purposes
            val_options  dictionnary with validation dataset options.
                                    T_val : Time after which observations are considered as test samples
                                    max_val_samples : maximum number of test observations per trajectory.
        """
        self.validation = validation

        if panda_df is not None:
            assert (csv_file is None), "Only one feeding option should be provided, not both"
            self.df = panda_df
            self.cov_df = cov_df
            self.label_df = label_df
        else:
            assert (csv_file is not None), "At least one feeding option required !"
            self.df = pd.read_csv(root_dir + "/" + csv_file)
            assert self.df.columns[0] == "ID"
            if label_file is None:
                self.label_df = None
            else:
                self.label_df = pd.read_csv(root_dir + "/" + label_file)
                assert self.label_df.columns[0] == "ID"
                assert self.label_df.columns[1] == "label"
            if cov_file is None:
                self.cov_df = None
            else:
                self.cov_df = pd.read_csv(root_dir + "/" + cov_file)
                assert self.cov_df.columns[0] == "ID"

        # Create Dummy covariates and labels if they are not fed.
        if self.cov_df is None:
            num_unique = np.zeros(self.df["ID"].nunique())
            self.cov_df = pd.DataFrame({"ID": self.df["ID"].unique(), "Cov": num_unique})
        if self.label_df is None:
            num_unique = np.zeros(self.df["ID"].nunique())
            self.label_df = pd.DataFrame({"ID": self.df["ID"].unique(), "label": num_unique})

        # If validation : consider only the data with a least one observation before T_val and one observation after:
        self.store_last = False
        if self.validation:
            df_beforeIdx = self.df.loc[self.df["Time"] <= val_options["T_val"], "ID"].unique()
            if val_options.get("T_val_from"):  # Validation samples only after some time.
                df_afterIdx = self.df.loc[self.df["Time"] >= val_options["T_val_from"], "ID"].unique()
                self.store_last = True  # Dataset get will return a flag for the collate to compute the last sample before T_val
            else:
                df_afterIdx = self.df.loc[self.df["Time"] > val_options["T_val"], "ID"].unique()

            valid_idx = np.intersect1d(df_beforeIdx, df_afterIdx)
            self.df = self.df.loc[self.df["ID"].isin(valid_idx)]
            self.label_df = self.label_df.loc[self.label_df["ID"].isin(valid_idx)]
            self.cov_df = self.cov_df.loc[self.cov_df["ID"].isin(valid_idx)]

        if idx is not None:
            self.df = self.df.loc[self.df["ID"].isin(idx)].copy()
            map_dict = dict(zip(self.df["ID"].unique(), np.arange(self.df["ID"].nunique())))
            self.df["ID"] = self.df["ID"].map(map_dict)  # Reset the ID index.

            self.cov_df = self.cov_df.loc[self.cov_df["ID"].isin(idx)].copy()
            self.cov_df["ID"] = self.cov_df["ID"].map(map_dict)  # Reset the ID index.

            self.label_df = self.label_df.loc[self.label_df["ID"].isin(idx)].copy()
            self.label_df["ID"] = self.label_df["ID"].map(map_dict)  # Reset the ID index.

        assert self.cov_df.shape[0] == self.df["ID"].nunique()

        self.variable_num = sum([c.startswith("Value") for c in self.df.columns])  # number of variables in the dataset
        self.cov_dim = self.cov_df.shape[1] - 1

        self.cov_df = self.cov_df.astype(np.float32)
        self.cov_df.set_index("ID", inplace=True)

        self.label_df.set_index("ID", inplace=True)

        self.df.Time = self.df.Time * t_mult

        # TO DO : make jitter compatible with several variables
        if jitter_time != 0:
            self.df = add_jitter(self.df, jitter_time=jitter_time)
            self.df.Value_1 = self.df.Value_1.astype(np.float32)
            self.df.Value_2 = self.df.Value_2.astype(np.float32)
            self.df.Mask_1 = self.df.Mask_1.astype(np.float32)
            self.df.Mask_2 = self.df.Mask_2.astype(np.float32)

        else:
            self.df = self.df.astype(np.float32)

        if self.validation:
            assert val_options is not None, "Validation set options should be fed"
            self.df_before = self.df.loc[self.df["Time"] <= val_options["T_val"]].copy()
            if val_options.get("T_val_from"):  # Validation samples only after some time.
                self.df_after = self.df.loc[self.df["Time"] >= val_options["T_val_from"]].sort_values("Time").copy()
            else:
                self.df_after = self.df.loc[self.df["Time"] > val_options["T_val"]].sort_values("Time").copy()

            if val_options.get("T_closest") is not None:
                df_after_temp = self.df_after.copy()
                df_after_temp["Time_from_target"] = (df_after_temp["Time"] - val_options["T_closest"]).abs()
                df_after_temp.sort_values(by=["Time_from_target", "Value_0"], inplace=True, ascending=True)
                df_after_temp.drop_duplicates(subset=["ID"], keep="first", inplace=True)
                self.df_after = df_after_temp.drop(columns=["Time_from_target"])
            else:
                self.df_after = self.df_after.groupby("ID").head(val_options["max_val_samples"]).copy()

            self.df = self.df_before  # We remove observations after T_val

            self.df_after.ID = self.df_after.ID.astype(np.int)
            self.df_after.sort_values("Time", inplace=True)
        else:
            self.df_after = None

        self.length = self.df["ID"].nunique()
        self.df.ID = self.df.ID.astype(np.int)
        self.df.set_index("ID", inplace=True)

        self.df.sort_values("Time", inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        subset = self.df.loc[idx]
        if len(subset.shape) == 1:  # Don't ask me anything about this.
            subset = self.df.loc[[idx]]

        covs = self.cov_df.loc[idx].values
        tag = self.label_df.loc[idx].astype(np.float32).values
        if self.validation:
            val_samples = self.df_after.loc[self.df_after["ID"] == idx]
        else:
            val_samples = None
        ## returning also idx to allow empty samples
        return {"idx": idx, "y": tag, "path": subset, "cov": covs, "val_samples": val_samples,
                "store_last": self.store_last}


def add_jitter(df, jitter_time=1e-3):
    """Modifies Double OU dataset, so that observations with both dimensions
       are split. One is randomly shifted earlier by amount 'jitter_time'.
    """
    if df.columns.shape[0] != 6:
        raise ValueError("Only df with 6 columns: supports 2 value and 2 mask columns.")

    both = (df["Mask_1"] == 1.0) & (df["Mask_2"] == 1.0)
    df_single = df[both == False]
    df_both = df[both]
    df_both1 = df_both.copy()
    df_both2 = df_both.copy()

    df_both1["Mask_2"] = 0.0
    df_both2["Mask_1"] = 0.0
    jitter = np.random.randint(2, size=df_both1.shape[0])
    df_both1["Time"] -= jitter_time * jitter
    df_both2["Time"] -= jitter_time * (1 - jitter)

    df_jit = pd.concat([df_single, df_both1, df_both2])
    ## make sure time is not negative:
    df_jit.Time.clip_lower(0.0, inplace=True)
    return df_jit

def normalize_data_climate(dataset):

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0]

    value_cols = [c.startswith("Value") for c in dataset[0]['path'].columns]
    mask_cols = [c.startswith("Mask") for c in dataset[0]['path'].columns]

    for b, data in enumerate(dataset):
        x = data['path']
        vals = x.iloc[:, value_cols].values
        mask = x.iloc[:, mask_cols].values
        n_features = vals.shape[-1]

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = torch.tensor(vals[:, i][mask[:, i] == 1])
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max

def custom_collate_fn_climate(batch, args, data_type="train", data_min=None, data_max=None):
    D = sum([c.startswith("Value") for c in batch[0]["path"]])
    combined_tt, inverse_indices = torch.unique(torch.cat([torch.tensor(ex['path']["Time"].values) for ex in batch]),
                                                sorted=True,
                                                return_inverse=True)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D])
    combined_mask = torch.zeros([len(batch), len(combined_tt), D])

    value_cols = [c.startswith("Value") for c in batch[0]['path'].columns]
    mask_cols = [c.startswith("Mask") for c in batch[0]['path'].columns]

    for i, b in enumerate(batch):
        x = b['path']
        vals = x.iloc[:, value_cols].values
        mask = x.iloc[:, mask_cols].values
        tt = x['Time']
        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[i, indices] = torch.tensor(vals)
        combined_mask[i, indices] = torch.tensor(mask)

    labels = torch.tensor([b["y"] for b in batch])

    # combined_vals, _, _ = normalize_masked_data(combined_vals, combined_mask,
    #                                             att_min=data_min, att_max=data_max)
    data_range = (data_max - data_min)
    data_range[data_range == 0.] = 1.
    if (data_range != 0.).all():
        combined_vals = (combined_vals - data_min) / data_range
    else:
        raise Exception("Zero!")
    if torch.isnan(combined_vals).any():
        raise Exception("nans!")
    combined_vals[combined_mask == 0] = 0


    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": labels}



    data_dict = split_and_subsample_batch(data_dict, args, data_type=data_type)

    ## get batch data
    batch_dict = {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None,
            "labels": None
            }

    # remove the time points where there are no observations in this batch
    non_missing_tp = torch.sum(data_dict["observed_data"], (0, 2)) != 0.
    batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
    batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]

    # print("observed data")
    # print(batch_dict["observed_data"].size())

    if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
        batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]

    batch_dict["data_to_predict"] = data_dict["data_to_predict"]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

    non_missing_tp = torch.sum(data_dict["data_to_predict"], (0, 2)) != 0.
    batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]

    # print("data_to_predict")
    # print(batch_dict["data_to_predict"].size())

    if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
        batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][:, non_missing_tp]

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        batch_dict["labels"] = data_dict["labels"]

    batch_dict["mode"] = data_dict["mode"]


    return batch_dict


def seq_collate_fn(batch):
    """
    Returns several tensors. Tensor of lengths should not be sent to CUDA.
    """
    idx2batch = pd.Series(np.arange(len(batch)), index=[b["idx"] for b in batch])
    df = pd.concat(b["path"] for b in batch)
    value_cols = [c.startswith("Value") for c in df.columns]
    mask_cols = [c.startswith("Mask") for c in df.columns]

    ## converting mask to int
    df.iloc[:, mask_cols] = df.iloc[:, mask_cols].astype(np.bool)
    df["num_obs"] = -df.iloc[:, mask_cols].sum(1)
    df.sort_values(by=["Time", "num_obs"], inplace=True)

    ## num_obs is not a value/mask column
    value_cols.append(False)
    mask_cols.append(False)

    cov = torch.Tensor([b["cov"] for b in batch])

    batch_ids = idx2batch[df.index.values].values

    ## calculating number of events at every time
    times, counts = np.unique(df.Time.values, return_counts=True)
    time_ptr = np.concatenate([[0], np.cumsum(counts)])
    assert df.shape[0] == time_ptr[-1]

    ## tensors for the data in the batch
    X = df.iloc[:, value_cols].values
    M = df.iloc[:, mask_cols].values

    ## selecting only observed X and splitting
    lengths = (-df.num_obs.values).tolist()
    Xsplit = torch.split(torch.from_numpy(X[M]), lengths)
    Fsplit = torch.split(torch.from_numpy(np.where(M)[1]), lengths)

    Xpadded = torch.nn.utils.rnn.pad_sequence(Xsplit, batch_first=True)
    Fpadded = torch.nn.utils.rnn.pad_sequence(Fsplit, batch_first=True)

    if batch[0]['val_samples'] is not None:
        df_after = pd.concat(b["val_samples"] for b in batch)
        df_after.sort_values(by=["ID", "Time"], inplace=True)
        value_cols_val = [c.startswith("Value") for c in df_after.columns]
        mask_cols_val = [c.startswith("Mask") for c in df_after.columns]
        X_val = torch.tensor(df_after.iloc[:, value_cols_val].values)
        M_val = torch.tensor(df_after.iloc[:, mask_cols_val].values)
        times_val = df_after["Time"].values
        index_val = idx2batch[df_after["ID"].values].values
    else:
        X_val = None
        M_val = None
        times_val = None
        index_val = None

    res = {}
    res["times"] = times
    res["time_ptr"] = time_ptr
    res["Xpadded"] = Xpadded
    res["Fpadded"] = Fpadded
    res["X"] = torch.from_numpy(X)
    res["M"] = torch.from_numpy(M.astype(np.float32))
    res["lengths"] = torch.LongTensor(lengths)
    res["obs_idx"] = torch.tensor(batch_ids)
    res["y"] = torch.tensor([b["y"] for b in batch])
    res["cov"] = cov

    res["X_val"] = X_val
    res["M_val"] = M_val
    res["times_val"] = times_val
    res["index_val"] = index_val

    return res


def extract_from_path(t_vec, p_vec, eval_times, path_idx_eval):
    '''
    Takes :
    t_vec : numpy vector of absolute times length [T]. Should be ordered.
    p_vec : numpy array of means and logvars of a trajectory at times t_vec. [T x batch_size x (2xfeatures)]
    eval_times : numpy vector of absolute times at which we want to retrieve p_vec. [L]
    path_idx_eval : index of trajectory that we want to retrieve. Should be same length of eval_times. [L]
    Returns :
    Array of dimensions [L,(2xfeatures)] of means and logvar of the required eval times and trajectories
    '''
    # Remove the evaluation after the updates. Only takes the prediction before the Bayesian update.
    t_vec, unique_index = np.unique(t_vec, return_index=True)
    p_vec = p_vec[unique_index, :, :]

    present_mask = np.isin(eval_times, t_vec)
    eval_times[~present_mask] = map_to_closest(eval_times[~present_mask], t_vec)

    mapping = dict(zip(t_vec, np.arange(t_vec.shape[0])))

    time_idx = np.vectorize(mapping.get)(eval_times)

    return (p_vec[time_idx, path_idx_eval, :])


def map_to_closest(input, reference):
    output = np.zeros_like(input)
    for idx, element in enumerate(input):
        closest_idx = (np.abs(reference - element)).argmin()
        output[idx] = reference[closest_idx]
    return (output)


def adjust_learning_rate(optimizer, epoch, init_lr):
    if epoch > 20:
        for param_group in optimizer.param_groups:
            param_group["lr"] = init_lr / 3


def compute_corr(X_true, X_hat, Mask):
    means_true = X_true.sum(0) / Mask.sum(0)
    means_hat = X_hat.sum(0) / Mask.sum(0)
    corr_num = ((X_true - means_true) * (X_hat - means_hat) * Mask).sum(0)
    corr_denum1 = ((X_true - means_true).pow(2) * Mask).sum(0).sqrt()
    corr_denum2 = ((X_hat - means_hat).pow(2) * Mask).sum(0).sqrt()
    return corr_num / (corr_denum1 * corr_denum2)


def sort_array_on_other(x1, x2):
    """
    This function returns the permutation y needed to transform x2 in x1 s.t. x2[y]=x1
    """

    temp_dict = dict(zip(x1, np.arange(len(x1))))
    index = np.vectorize(temp_dict.get)(x2)
    perm = np.argsort(index)

    assert ((x2[perm] == x1).all())

    return (perm)


def log_lik_gaussian(x, mu, logvar):
    return np.log(np.sqrt(2 * np.pi)) + (logvar / 2) + ((x - mu).pow(2) / (2 * logvar.exp()))


def tail_fun_gaussian(x, mu, logvar):
    """
    Returns the probability that the given distribution is HIGHER than x.
    """
    return 0.5 - 0.5 * special.erf((x - mu) / ((0.5 * logvar).exp() * np.sqrt(2)))