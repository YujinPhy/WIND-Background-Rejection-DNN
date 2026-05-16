import os
import pandas as pd
import matplotlib.pyplot as plt

def loss_analysis(log_dir, output_path, png_title):
    csv_path = os.path.join(log_dir, "metrics.csv")
    bak_path = os.path.join(log_dir, "metrics.csv.bak")
    
    # 1. Load and Merge files
    dfs = []
    
    # Check for backup file
    if os.path.exists(bak_path):
        dfs.append(pd.read_csv(bak_path))
        print(f" [INFO] Loaded backup log: {bak_path}")
    
    # Check for current metric file
    if os.path.exists(csv_path):
        dfs.append(pd.read_csv(csv_path))
        print(f" [INFO] Loaded current log: {csv_path}")
    
    if not dfs:
        print(" [ERROR] No log files found for analysis.")
        return

    # Integrate datasets
    df_full = pd.concat(dfs, axis=0, ignore_index=True)

    # 2. Data Cleaning (Handling Duplicate Epochs)
    # Using last() ensures that the most recent training data (from resume) is preserved
    epoch_train = df_full.dropna(subset=['train_loss']).groupby('epoch')['train_loss'].mean().reset_index()
    epoch_val = df_full.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].last().reset_index()

    # 3. Identify Best Epoch (Based on Validation Loss)
    best_idx = epoch_val['val_loss'].idxmin()
    best_epoch = epoch_val.loc[best_idx, 'epoch']
    best_loss = epoch_val.loc[best_idx, 'val_loss']

    # 4. Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(epoch_train['epoch'], epoch_train['train_loss'], 
             label='Training Loss (Total)', color='royalblue', lw=1.5, alpha=0.7)
    
    plt.plot(epoch_val['epoch'], epoch_val['val_loss'], 
             label='Validation Loss (Total)', color='darkorange', lw=2, marker='s', markersize=4)

    # Guide line for Best Epoch
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.8,
                label=f'Best: Ep {int(best_epoch)} (Loss: {best_loss:.4f})')
    plt.scatter(best_epoch, best_loss, color='red', s=60, zorder=5)

    plt.title(f'Continuous Learning Curve: {png_title}', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 5. Export results
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_path = os.path.join(output_path, f"{png_title}.png")
    plt.savefig(save_path, dpi=300)
    print(f" [SUCCESS] Analysis complete. Graph saved to: {save_path}")

def save_batch_visualization(batch, channel_names,base_path, dir_name):
    inputs, targets = batch
    inputs_cpu = inputs.cpu().numpy()
    targets_cpu = targets.cpu().numpy()

    # 저장 경로 설정
    final_save_path = os.path.join(base_path, dir_name)
    dir_es = os.path.join(final_save_path, "ES_Signal")
    dir_16n = os.path.join(final_save_path, "16N_BKG")
    os.makedirs(dir_es, exist_ok=True)
    os.makedirs(dir_16n, exist_ok=True)

    # 실제 데이터의 채널 수 확인
    num_channels = inputs_cpu.shape[1] # 예: 2 또는 6

    if num_channels != len(channel_names):
        raise ValueError("num_channels is mismatch to number of channel_names")
    
    print(f"==== Batch Visualization ====")
    print(f"Inputs Shape  : {inputs.shape}")
    print(f"Detected Channels: {num_channels}")

    for sample_idx in range(len(inputs_cpu)):
        img = inputs_cpu[sample_idx]
        label_num = targets_cpu[sample_idx]
        
        # 레이블에 따른 저장 폴더 결정
        if label_num == 1:
            label_name, current_dir = "ES", dir_es
        else:
            label_name, current_dir = "16N", dir_16n
        
        # 실제 채널 수(num_channels)만큼 서브플롯 생성
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels))
        if num_channels == 1:
            axes = [axes] # 채널이 1개일 때를 위한 예외 처리
            
        fig.suptitle(f"[{label_name}] Sample {sample_idx}", fontsize=20)

        for i in range(num_channels):
            # 채널 이름이 정의된 리스트보다 많을 경우 대비
            title = channel_names[i] if i < len(channel_names) else f"Ch {i}"
            
            # Hit(0), Charge(1)은 viridis, 시간 계열은 magma 컬러맵 사용
            cmap = 'viridis' if i < 2 else 'magma'
            
            im = axes[i].imshow(img[i], aspect='auto', origin='lower', cmap=cmap)
            axes[i].set_title(f"Ch {i}: {title}")
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 파일 저장
        file_name = f"sample_{sample_idx:03d}.png"
        plt.savefig(os.path.join(current_dir, file_name))
        plt.close(fig)

    print(f"Successfully saved {len(inputs_cpu)} images to {final_save_path}")

base_path = "/home/yujin/projects/wind/BKG_rejection/CNN/logs/test/version_0"
csv_path = os.path.join(base_path, "metrics.csv")
output_path = base_path
loss_analysis(csv_path=csv_path, output_path=base_path, png_title="loss_curve.png" )