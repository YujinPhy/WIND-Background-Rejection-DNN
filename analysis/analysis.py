import os
import pandas as pd
import matplotlib.pyplot as plt

def loss_analysis(csv_path, output_path, png_title):
    # 1. 데이터 로드
    df = pd.read_csv(csv_path)
    
    # 2. 에포크별 데이터 처리
    # Training Loss: 같은 에포크 내의 여러 step 값들을 평균 내어 대표값 생성
    epoch_train = df.groupby('epoch')['train_loss'].mean().reset_index()
    # Validation Loss: 에포크당 하나씩 있는 값을 추출
    epoch_val = df.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].first().reset_index()

    # 3. 최적의 Epoch 찾기 (Validation Loss 기준)
    best_idx = epoch_val['val_loss'].idxmin()
    best_epoch = epoch_val.loc[best_idx, 'epoch']
    best_loss = epoch_val.loc[best_idx, 'val_loss']

    # 4. 그래프 그리기
    plt.figure(figsize=(12, 6))

    # Training Loss Plot (평균값 사용)
    plt.plot(epoch_train['epoch'], epoch_train['train_loss'], 
             label='Training Loss (Avg)', color='royalblue', lw=2, marker='o', markersize=4)
    
    # Validation Loss Plot
    plt.plot(epoch_val['epoch'], epoch_val['val_loss'], 
             label='Validation Loss', color='darkorange', lw=2, linestyle='--', marker='s', markersize=6)

    # Best Epoch 가이드라인 및 점 표시
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.8,
                label=f'Best Epoch: {int(best_epoch)} (Loss: {best_loss:.4f})')
    plt.scatter(best_epoch, best_loss, color='red', s=80, zorder=5, edgecolors='black')

    # 그래프 포맷팅
    plt.title('Learning Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # X축을 에포크 번호로 명확히 표시
    plt.xticks(epoch_val['epoch']) 
    plt.tight_layout()

    # 5. 저장 및 출력
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_path = os.path.join(output_path, f"{png_title}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 그래프가 저장되었습니다: {save_path}")
    
    plt.show()


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