from pathlib import Path
from collections import OrderedDict
import json
from torch.utils.data import Dataset, DataLoader
import torch

class ShardedWeatherBenchDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        input_steps = 4,
        lead_steps = 1,
        flatten_time = False,
        return_time = False,
        cache_size = 2,
        lat_channels=False
    ):
        """
        root:
            Directory containing metadata.json and split shard folders.

        split:
            "train", "val", or "test"

        input_steps:
            Number of input frames. Example:
                4 means [t-3, t-2, t-1, t]

        lead_steps:
            Forecast lead offset.
            For 6-hour data, lead_steps=1 means target starts at t+6h.

        flatten_time:
            If True, x shape becomes:
                input_steps * variable, lat, lon
            If False, x shape remains:
                input_steps, variable, lat, lon

        cache_size:
            Number of shard files to keep in memory per worker/process.
        """
        self.root = Path(root)
        self.split = split
        self.input_steps = input_steps
        self.lead_steps = lead_steps
        self.flatten_time = flatten_time
        self.return_time = return_time
        self.cache_size = cache_size
        self.lat_channels = lat_channels

        with open(self.root / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.shards = [
            s for s in self.metadata["shards"]
            if s["split"] == split
        ]

        if len(self.shards) == 0:
            raise ValueError(f"No shards found for split={split}")
        
        self.indices = []
        for shard_id, shard in enumerate(self.shards):
            n = shard["n_time"]

            # t is the final input time.
            # Need:
            #   t - input_steps + 1 >= 0
            #   t + lead_steps - 1 < n
            first_t = input_steps - 1
            last_t_exclusive = n - lead_steps

            for t in range(first_t, last_t_exclusive):
                self.indices.append((shard_id, t))

        self._cache = OrderedDict()

        if self.lat_channels:
            tmp_load = self._load_shard(0)
            T, V, H, W = tmp_load["data"].shape
            lat = torch.deg2rad(torch.as_tensor(tmp_load["latitude"]))
            sin_lat = torch.sin(lat).view(1, H, 1).expand(1, H, W)
            cos_lat = torch.cos(lat).view(1, H, 1).expand(1, H, W)
            self.lat_feature = torch.cat([sin_lat, cos_lat], dim=0)

    def __len__(self):
        return len(self.indices)
    

    def _load_shard(self, shard_id):
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        
        shard = self.shards[shard_id]
        path = self.root / shard['path']

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[shard_id] = payload

        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return payload

    def __getitem__(self, idx):
        shard_id, t = self.indices[idx]
        payload = self._load_shard(shard_id)

        data = payload["data"]

        # (time, variable, latitude, longitude)
        x = data[t-self.input_steps + 1: t+1]
        y = data[t+self.lead_steps]

        if self.flatten_time:
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
            if self.lat_channels:
                x = torch.cat([x, self.lat_feature], dim=0)
                
        if self.return_time:
            target_time = payload["times"][t+self.lead_steps]
            return x, y, target_time
        
        return x, y
    

class WeatherBenchMapDataset(Dataset):
    def __init__(self, state_array, input_steps=4, lead_steps=1, time_start=0, time_stop=None, flatten_time = False):
        self.state_array = state_array
        self.input_steps = input_steps
        self.lead_steps = lead_steps
        self.flatten_time = flatten_time

        if time_stop is None:
            time_stop = state_array.sizes['time']

        first_index = time_start + input_steps - 1
        last_index = time_stop - lead_steps

        self.indices = list(range(first_index, last_index))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.state_array.isel(time=slice(t-self.input_steps+1, t+1)).compute()
        y = self.state_array.isel(time=t+self.lead_steps).compute()

        # x (input steps, variables, lat, lon)
        # y (variables, lat, lon)
        x = torch.as_tensor(x.data, dtype=torch.float32)
        y = torch.as_tensor(y.data, dtype=torch.float32)

        if self.flatten_time:
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
    

def build_loaders(config):
    data_cfg = config["data"]

    train_dataset = ShardedWeatherBenchDataset(
        root=data_cfg["dataset_path"],
        split="train",
        input_steps=data_cfg["input_steps"],
        lead_steps=data_cfg["lead_steps"],
        flatten_time=data_cfg["flatten_time"],
        lat_channels=data_cfg["lat_channels"]
    )
    val_dataset = ShardedWeatherBenchDataset(
        root=data_cfg["dataset_path"],
        split="val",
        input_steps=data_cfg["input_steps"],
        lead_steps=data_cfg["lead_steps"],
        flatten_time=data_cfg["flatten_time"],
        lat_channels=data_cfg["lat_channels"]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"]
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"]
    )

    return train_loader, val_loader
    
