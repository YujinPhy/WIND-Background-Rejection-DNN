import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import pytorch_lightning as pl
from WIND_bkg_rejection.datasets.DataSet import WIND_Dataset

class WIND_DataModule(pl.LightningDataModule):
    def __init__(self, es_root, n16_root, energy_thr, vtx_cut_mm, min_unique_pmt,
                 test_ratio, val_ratio, seed, batch_size, shuffle, pin_memory, num_workers):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        es_full = WIND_Dataset(self.hparams.es_root, 1, self.hparams.energy_thr, self.hparams.vtx_cut_mm, self.hparams.min_unique_pmt)
        n16_full = WIND_Dataset(self.hparams.n16_root, 0, self.hparams.energy_thr, self.hparams.vtx_cut_mm, self.hparams.min_unique_pmt)

        def split_helper(dataset):
            total_count = len(dataset)
            test_size = int(total_count * self.hparams.test_ratio)
            train_val_size = total_count - test_size
            val_size = int(train_val_size * self.hparams.val_ratio)
            train_size = train_val_size - val_size

            train_val_set, test_set = random_split(dataset,
                                                   [train_val_size, test_size],
                                                   generator=torch.Generator().manual_seed(self.hparams.seed))
            
            
            train_set, val_set = random_split(train_val_set,
                                              [train_size, val_size],
                                              generator=torch.Generator().manual_seed(self.hparams.seed))
            return train_set, val_set, test_set

        es_train, es_val, es_test = split_helper(es_full)
        n16_train, n16_val, n16_test = split_helper(n16_full)

        # 3. 데이터셋 병합
        self.train_set = ConcatDataset([es_train, n16_train])
        self.val_set = ConcatDataset([es_val, n16_val])
        self.test_set = ConcatDataset([es_test, n16_test])

        print("\nDataset Statistics with Ratios ---")
        es_total = len(es_full)
        n16_total = len(n16_full)
        grand_total = es_total + n16_total
        
        print(f"ES Signal (After Selection): {es_total} ({es_total/grand_total*100:.1f}%)")
        print(f"16N Bkg    (After Selection): {n16_total} ({n16_total/grand_total*100:.1f}%)")
        print(f"Grand Total: {grand_total}")
        print("-" * 40)
            
        def print_split_info(name, es_set, n16_set):
            total = len(es_set) + len(n16_set)
            es_p = (len(es_set) / total) * 100
            n16_p = (len(n16_set) / total) * 100
            print(f"{name:9s}: Total {total:6d} | ES: {len(es_set):5d} ({es_p:4.1f}%) | 16N: {len(n16_set):5d} ({n16_p:4.1f}%)")

        print_split_info("Train Set", es_train, n16_train)
        print_split_info("Val Set", es_val, n16_val)
        print_split_info("Test Set", es_test, n16_test)
        print("========================================\n")
        
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          shuffle=self.hparams.shuffle,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=(self.hparams.num_workers > 0))
    
    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=(self.hparams.num_workers > 0))

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=(self.hparams.num_workers > 0))