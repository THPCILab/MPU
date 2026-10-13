"""Use the exact previously projected seed-44 EMNIST inputs."""
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def install(exp, cache_path):
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    def get_task_loaders(task_set, train_samples, test_samples, batch_size, seed,
                         download=False, label_mode="local",
                         input_preprocess="learned", input_rank=64):
        expected = payload["settings"]
        actual = dict(task_set=task_set,train_samples=train_samples,test_samples=test_samples,
                      seed=seed,label_mode=label_mode,input_preprocess=input_preprocess,input_rank=input_rank)
        if actual != expected or batch_size != 512:
            raise RuntimeError("Requested data protocol differs from the audited feature cache")
        train = {tid:DataLoader(TensorDataset(*tensors),batch_size=batch_size,shuffle=True)
                 for tid,tensors in payload["train"].items()}
        test = {tid:DataLoader(TensorDataset(*tensors),batch_size=batch_size,shuffle=False)
                for tid,tensors in payload["test"].items()}
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return payload["specs"],train,test,payload["profile"]
    exp.get_task_loaders = get_task_loaders
