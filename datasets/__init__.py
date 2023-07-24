import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
from .physionet import PhysioNet, split_and_subsample_batch, get_data_min_max, variable_time_collate_fn

from .climate import Climate, custom_collate_fn_climate, normalize_data_climate
from .monash import load_data as Monash
from .monash import datasets as monash_names

def parse_datasets(args, batch_size, device, main_args=None):

    dataset_name = args.dataset
    

    ##################################################################
    # Monash dataset
    if dataset_name in monash_names:
        monash_ds = Monash(args.path, dataset_name) # (bs, T, dim)
        
        train_data = [d.to_numpy()[:,None] for d in monash_ds.full_series_list]
        if 'per_seq' in args.preproc:
            
            train_mean = torch.tensor(np.stack([d.mean(0, keepdims=True) for d in train_data], axis=0), dtype=torch.float)
            train_std = torch.tensor(np.stack([d.std(0, keepdims=True) for d in train_data], axis=0), dtype=torch.float)
            train_max = torch.tensor(np.stack([d.max(0, keepdims=True) for d in train_data], axis=0), dtype=torch.float)
            train_min = torch.tensor(np.stack([d.min(0, keepdims=True) for d in train_data], axis=0), dtype=torch.float)
        else:
            train_mean = torch.tensor(np.concatenate(train_data, axis=0).mean(0), dtype=torch.float)
            train_std = torch.tensor(np.concatenate(train_data, axis=0).std(0), dtype=torch.float)
            train_max = torch.tensor(np.concatenate(train_data, axis=0).max(0), dtype=torch.float)
            train_min = torch.tensor(np.concatenate(train_data, axis=0).min(0), dtype=torch.float)
        
        train_range = (train_max - train_min)
        train_range[train_range == 0] = 1
        train_std[train_std==0] = 1
        
        train_data = [torch.tensor(d, dtype=torch.float) for d in train_data]
        train_mask = [torch.ones_like(d, dtype=torch.bool) for d in train_data]
        
        train_data = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True)
        train_mask = torch.nn.utils.rnn.pad_sequence(train_mask, batch_first=True)
        
        
        if args.preproc == 'squash':
            train_data[train_mask] = ((train_data - train_min) / train_range)[train_mask].float()
            decode = lambda x: x * (train_max - train_min).to(x) + train_min.to(x)
        elif args.preproc == 'squash_shift':
            train_data[train_mask] = ((train_data - train_min) /train_range)[train_mask].float() * 2 - 1
            decode = lambda x:( x *0.5 + 0.5) * (train_max - train_min).to(x) + train_min.to(x)
        elif args.preproc == 'normalize':
            train_data[train_mask] = ((train_data - train_mean) / train_std)[train_mask].float()
            decode = lambda x: x * train_std.to(x) + train_mean.to(x)
        elif args.preproc == 'squash_shift_per_seq':
            train_data[train_mask] =  ((train_data - train_min) / train_range)[train_mask].float() * 2 - 1
            decode = lambda x: x
        elif args.preproc == 'squash_per_seq':
            train_data[train_mask] = ((train_data - train_min) / train_range)[train_mask].float()
            decode = lambda x: x
        elif args.preproc == 'normalize_per_seq':
            train_data[train_mask] = ((train_data - train_mean) / train_std)[train_mask].float()
            decode = lambda x: x 
        else:
            raise NotImplementedError
        
        full_len = train_data.shape[0]
        randperm = torch.randperm(full_len)

        train_data_ = TensorDataset(train_data[randperm[:int(full_len * 0.8)]], train_mask[randperm[:int(full_len * 0.8)]])
        test_data_ = TensorDataset(train_data[randperm[int(full_len * 0.8):]], train_mask[randperm[int(full_len * 0.8):]])
        
        train_dataloader = DataLoader(train_data_, batch_size=batch_size, shuffle=True )
        test_dataloader = DataLoader(test_data_, batch_size=batch_size, shuffle=False)
        
        data_objects = {"dataset_obj": train_data,
                        "train_dataloader": train_dataloader,#inf_generator(train_dataloader),
                        "test_dataloader": test_dataloader,# inf_generator(test_dataloader),
                        'decode_func': decode,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader),
                        }  # optional
        return data_objects
    # TODO: make this the correct dataset loader

    ##################################################################
    # Physionet dataset

    if dataset_name == "physionet":
        train_dataset_obj = PhysioNet(args.path, train=True,
                                      quantization=args.quantization,
                                      download=True, n_samples=min(10000, args.n),
                                      device=device)
        # Use custom collate_fn to combine samples with arbitrary time observations.
        # Returns the dataset along with mask and time steps
        test_dataset_obj = PhysioNet(args.path, train=False,
                                     quantization=args.quantization,
                                     download=True, n_samples=min(10000, args.n),
                                     device=device)

        # Combine and shuffle samples from physionet Train and physionet Test
        total_dataset = train_dataset_obj[:len(train_dataset_obj)]

        if not args.classify:
            # Concatenate samples from original Train and Test sets
            # Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
            total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                                 random_state=42, shuffle=True)

        record_id, tt, vals, mask, labels = train_data[0]

        n_samples = len(total_dataset)
        input_dim = vals.size(-1)

        batch_size = min(min(len(train_dataset_obj), batch_size), args.n)
        data_min, data_max = get_data_min_max(total_dataset, device=device)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: variable_time_collate_fn(batch, args, device,
                                                                                        data_type="train",
                                                                                        data_min=data_min,
                                                                                        data_max=data_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: variable_time_collate_fn(batch, args, device,
                                                                                       data_type="test",
                                                                                       data_min=data_min,
                                                                                       data_max=data_max))

        attr_names = train_dataset_obj.params
        data_objects = {"dataset_obj": train_dataset_obj,
                        "train_dataloader": train_dataloader,#inf_generator(train_dataloader),
                        "test_dataloader": test_dataloader,# inf_generator(test_dataloader),
                        "input_dim": input_dim,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader),
                        "attr": attr_names,  # optional
                        "classif_per_tp": False,  # optional
                        "n_labels": 1}  # optional
        return data_objects


    ##################################################################
    # climate dataset

    if dataset_name == "climate":
        csv_file_path = args.csv_file_path
        csv_file_tags = getattr(args, 'csv_file_tags', None)
        csv_file_cov = getattr(args, 'csv_file_cov', None)

        validation = True
        val_options = {"T_val": args.T_val, "max_val_samples": args.max_val_samples}

        train_idx = np.load(os.path.join(args.dir, "train_idx.npy"),
                            allow_pickle=True)
        val_idx = np.load(os.path.join(args.dir, "val_idx.npy"),
                          allow_pickle=True)
        test_idx = np.load(os.path.join(args.dir, "test_idx.npy"),
                           allow_pickle=True)

        train_data = Climate(csv_file=csv_file_path, label_file=csv_file_tags, cov_file=csv_file_cov,
                               idx=train_idx, root_dir=args.base)
        val_data = Climate(csv_file=csv_file_path, label_file=csv_file_tags,
                             cov_file=csv_file_cov, idx=val_idx, validation=validation,
                             val_options=val_options, root_dir=args.base)
        test_data = Climate(csv_file=csv_file_path, label_file=csv_file_tags,
                              cov_file=csv_file_cov, idx=test_idx, validation=validation,
                              val_options=val_options, root_dir=args.base)
        train_data = train_data + val_data

        data_min, data_max = normalize_data_climate(train_data)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: custom_collate_fn_climate(batch, args,
                                                                                       data_type="train",
                                                                                       data_min=data_min,
                                                                                       data_max=data_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: custom_collate_fn_climate(batch, args,
                                                                                       data_type="test",
                                                                                       data_min=data_min,
                                                                                       data_max=data_max))


        data_objects = {"dataset_obj": train_data,
                        "train_dataloader": train_dataloader,#inf_generator(train_dataloader),
                        "test_dataloader": test_dataloader,# inf_generator(test_dataloader),
                        "input_dim": 5,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader),
                        "classif_per_tp": False,}  # optional
        return data_objects

