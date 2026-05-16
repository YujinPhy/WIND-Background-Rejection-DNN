import torch
import torch_directml
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.DataModule import WIND_DataModule
from models.ResNet18 import resnet18
from models.cnn_reference import cnn_reference
from models.Sparse_ResNet18 import sparse_resnet18

from utils import save_batch_visualization

# ==== I/O ====
# ES_ROOT = "/home/yujin/projects/wind/WIND_bkg_rejection/ROOT_FILEs/WIND_66_4in_40p_ES_10k_internal_PMT.ntuple.root"
# N16_ROOT = "/home/yujin/projects/wind/WIND_bkg_rejection/ROOT_FILEs/WIND_66_4in_40p_16N_10k_internal.ntuple.root"

ES_H5 = "/home/yujin/projects/wind/WIND_bkg_rejection/ROOT_FILEs/WIND_ES.h5"
N16_H5 = "/home/yujin/projects/wind/WIND_bkg_rejection/ROOT_FILEs/WIND_16N.h5"

CHECKPOINT_PATH = "/home/yujin/projects/wind/WIND_bkg_rejection/CNN/logs"
CHECKPOINT_NAME = "test"

# ==== Physical Paramters ====
ENERGY_CUT = 0 # MeV
VERTEX_CUT = 0 # mm
MIN_UNIQUE_PMTS = 4
# MIN_UNIQUE_PMTS = 25


# ====  Hardware Parameters ====
IS_GPU = True

# ==== Training Parameters ====
TEST_RATIO = 0.2 
VAL_RATIO = 0.2 

SEED = 42
SHUFFLE = True

BATCH_SIZE = 32
NUM_WORKERS = 16
MAX_APOCH = 50
LR = 1e-5


if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)

    print(" ==== Hardware Detection ====")
    if IS_GPU:
        device = torch_directml.device()
        pin_memory = False
        print(f" - Intel GPU (DirectM1L): {device} ")
    else:
        device = torch.device("cpu")
        pin_memory = True
        print(" - CPU  ")

    print(" ==== Load Data Module ====")
    dm = WIND_DataModule(es_root=ES_ROOT,
                         n16_root=N16_ROOT,
                         energy_thr=ENERGY_CUT,
                         vtx_cut_mm=VERTEX_CUT,
                         min_unique_pmt=MIN_UNIQUE_PMTS,
                         test_ratio=TEST_RATIO,
                         val_ratio=VAL_RATIO,
                         seed=SEED,
                         batch_size=BATCH_SIZE,
                         shuffle=SHUFFLE,
                         pin_memory=pin_memory,
                         num_workers=NUM_WORKERS)

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    print("==== Batch Information ====")
    print(f"Train batches: {len(train_loader)}")

    batch = next(iter(train_loader))
    save_batch_visualization(batch, ["Charge", "First Hit Time"], "/home/yujin/projects/wind/WIND_bkg_rejection/CNN", "iamge_unique_4")

    model = resnet18(lr=LR, is_gpu=IS_GPU)
    # model = sparse_resnet18(lr=LR, is_gpu=IS_GPU)
    # model = cnn_reference(lr=LR, is_gpu=True)
    model.to(device) 

    # ==== 체크포인트 설정 ====
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",    
                                          mode="min",           
                                          save_top_k=1,
                                          filename="best-{epoch:02d}-{val_loss:.4f}")
    csv_logger = CSVLogger(save_dir=CHECKPOINT_PATH, name=CHECKPOINT_NAME)
    
    # ==== 5. Trainer 설정 ====
    trainer = pl.Trainer(max_epochs=MAX_APOCH,
                         accelerator="cpu", # Lightning 엔진은 CPU로 설정 (DML은 내부적으로 처리)
                         strategy="single_device",
                         devices=1,
                         logger=csv_logger,
                         callbacks=[checkpoint_callback],
                         log_every_n_steps=10,
                         enable_progress_bar=True)

    # ==== 6. 훈련 및 테스트 시작 ====

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)4