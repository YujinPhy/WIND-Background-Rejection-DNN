import torch
from torch.utils.data import Dataset
import uproot
import awkward as ak
from WIND_bkg_rejection.datasets.utils import *

class WIND_Dataset(Dataset):
    def __init__(self, root_path, label, energy_thr, vtx_cut_mm, min_unique_pmt):
        super().__init__()
        self.root_path = root_path
        self.label = label
        self.energy_thr = energy_thr
        self.vtx_cut_mm = vtx_cut_mm
        self.min_unique_pmt = min_unique_pmt
        
        self.branches = ["mcPEx", "mcPEy", "mcPEz", "mcPECharge", "mcPEHitTime"]
        temp_branches = ["mcid", "mcke", "mcnhits", "mcx", "mcy", "mcz"]
        
        with uproot.open(f"{self.root_path}:output") as tree:
            full_branches = self.branches + temp_branches
            full_data = tree.arrays(full_branches, library="ak")

        print(f"Load from '{self.root_path}'")
        print(f" - # of event in the root file      : {len(full_data)}")

        if self.label == 1: # ES
            self.data = es_event_selection(
                root_path=self.root_path,
                awk_data=full_data,
                energy_thr=self.energy_thr,
                vtx_cut_mm=self.vtx_cut_mm,
                min_unique_pmts=self.min_unique_pmt
            )
        else: # BKG
            self.data = bkg_event_selection(
                root_path=self.root_path,
                awk_data=full_data,
                energy_thr=self.energy_thr,
                vtx_cut_mm=self.vtx_cut_mm,
                min_unique_pmts=self.min_unique_pmt
            )

        print(f" - # of event after event selection : {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        event = self.data[idx]
        
        x_pos = ak.to_numpy(event["mcPEx"])
        y_pos = ak.to_numpy(event["mcPEy"])
        z_pos = ak.to_numpy(event["mcPEz"])
        charge = ak.to_numpy(event["mcPECharge"])
        hit_time = ak.to_numpy(event["mcPEHitTime"])
        
        chw = get_channels(x=x_pos,
                           y=y_pos,
                           z=z_pos,
                           pe_times=hit_time,
                           charges=charge,
                           width=142, side_height=45, cap_res=23)
        
        # geo_cor = get_geometry_channels(
        #     width=142,
        #     side_height=45,
        #     cap_res=23
        # )

        # full_chw = np.concatenate([chw, geo_cor], axis=0)
        full_chw = chw
        return torch.from_numpy(full_chw).float(), torch.tensor(self.label).long()