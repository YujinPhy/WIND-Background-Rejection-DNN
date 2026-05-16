import h5py
import torch
from torch.utils.data import Dataset

class WIND_Dataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        
        with h5py.File(self.h5_path, 'r') as f:
            self.num_events = f['input'].shape[0]
            
            self.energy_thr = f.attrs.get('energy_thr', 'Unknown')
            self.min_unique_pmt = f.attrs.get('min_unique_pmt', 'Unknown')
            
            self.channels = [f['input'].attrs.get(f'ch{i}') for i in range(f['input'].shape[1])]

        print(f"Load H5 Dataset: '{self.h5_path}'")
        print(f" - Events: {self.num_events}")
        print(f" - Channels: {self.channels}")
        print(f" - Attributes: [Energy Thr: {self.energy_thr}, Min PMT: {self.min_unique_pmt}]")

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = f['input'][idx]   # (C, H, W)
            label = f['label'][idx] # int
            
        return torch.from_numpy(img).float(), torch.tensor(label).long()