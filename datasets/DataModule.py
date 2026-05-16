import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import pytorch_lightning as pl
from datasets.DataSet import WIND_Dataset

class WIND_DataModule(pl.LightningDataModule):
    def __init__(self, es_h5, n16_h5, test_ratio, val_ratio, seed, 
                 batch_size, shuffle, pin_memory, num_workers):
        super().__init__()
        # h5 파일 경로들을 하이퍼파라미터로 저장합니다.
        self.save_hyperparameters()

    def setup(self, stage=None):
        # 1. 이제 물리 컷 인자 없이 H5 경로만 전달합니다.
        # 내부 구조에서 설명했듯이 Dataset 클래스가 H5를 읽어올 것입니다.
        es_full = WIND_Dataset(self.hparams.es_h5)
        n16_full = WIND_Dataset(self.hparams.n16_h5)

        def split_helper(dataset):
            total_count = len(dataset)
            test_size = int(total_count * self.hparams.test_ratio)
            train_val_size = total_count - test_size
            val_size = int(train_val_size * self.hparams.val_ratio)
            train_size = train_val_size - val_size

            train_val_set, test_set = random_split(
                dataset, [train_val_size, test_size],
                generator=torch.Generator().manual_seed(self.hparams.seed)
            )
            
            train_set, val_set = random_split(
                train_val_set, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.hparams.seed)
            )
            return train_set, val_set, test_set

        # 2. ES와 16N 각각 Split 수행
        es_train, es_val, es_test = split_helper(es_full)
        n16_train, n16_val, n16_test = split_helper(n16_full)

        # 3. 데이터셋 병합 (ES: Label 1, 16N: Label 0은 이미 H5 안에 저장되어 있음)
        self.train_set = ConcatDataset([es_train, n16_train])
        self.val_set = ConcatDataset([es_val, n16_val])
        self.test_set = ConcatDataset([es_test, n16_test])

        # 통계 출력 부분 (디버깅용)
        print("\n" + "="*40)
        print("H5 Data Loading Statistics")
        es_total, n16_total = len(es_full), len(n16_full)
        grand_total = es_total + n16_total
        print(f" - ES Signal (H5): {es_total} ({es_total/grand_total*100:.1f}%)")
        print(f" - 16N Bkg    (H5): {n16_total} ({n16_total/grand_total*100:.1f}%)")
        print(f" - Grand Total   : {grand_total}")
        print("-" * 40)
            
        def print_split_info(name, es_set, n16_set):
            total = len(es_set) + len(n16_set)
            es_p = (len(es_set) / total) * 100 if total > 0 else 0
            n16_p = (len(n16_set) / total) * 100 if total > 0 else 0
            print(f"{name:9s}: Total {total:6d} | ES: {len(es_set):5d} ({es_p:4.1f}%) | 16N: {len(n16_set):5d} ({n16_p:4.1f}%)")

        print_split_info("Train Set", es_train, n16_train)
        print_split_info("Val Set", es_val, n16_val)
        print_split_info("Test Set", es_test, n16_test)
        print("="*40 + "\n")
        
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