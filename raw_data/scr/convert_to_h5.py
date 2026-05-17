import h5py
import numpy as np
from tqdm import tqdm
import os
from utils import *

"""
h5 File Data Structure
/(Root Group)
┃
┣━━ attrs (Metadata)
┃   ┣━━ energy_thr: 0 (int64)
┃   ┣━━ vtx_cut_mm: 1000 (int64)
┃   ┗━━ min_unique_pmt: 25 (int64)
┃
┣━━ mcid (원본 ROOT 파일의 Event ID)
┃   ┣━━ Shape: (N,)            
┃   ┗━━ Dtype: int64
┃
┣━━ label  (ES: 1, 16N: 0)
┃   ┣━━ Shape: (N,)             
┃   ┗━━ Dtype: int64
┃
┗━━ data (이벤트 수, 채널, 높이, 너비)
    ┣━━ Shape: (N, C, H, W) 
    ┣━━ Dtype: float32
    ┗━━ Chunk: (1, 6, 91, 142)  
"""

# ==== Data Paths ====
ES_FILE = "/home/yujin/projects/wind/WIND_bkg_rejection/raw_data/WIND_66_4in_40p_ES_10k_internal_PMT.ntuple.root"
N16_FILE = "/home/yujin/projects/wind/WIND_bkg_rejection/raw_data/WIND_66_4in_40p_16N_10k_internal.ntuple.root"

OUTPUT_PATH = "/home/yujin/projects/wind/WIND_bkg_rejection/raw_data"
# ==== Parameters ====
ENERGY_CUT = 0 # MeV
VERTEX_CUT = 0 # mm
MIN_UNIQUE_PMTS = 4


def event_selection(root_file, label, energy_thr, vtx_cut_mm, min_unique_pmt):
    branches = ["mcPEx", "mcPEy", "mcPEz", "mcPECharge", "mcPEHitTime"]
    sub_branches = ["mcid", "mcke", "mcnhits", "mcx", "mcy", "mcz"]

    with uproot.open(f"{root_file}:output") as tree:
        full_branches = branches + sub_branches
        full_data = tree.arrays(full_branches, library="ak")

    print(f"[Event Selection] Load from: {root_file}")
    print(f" - Total number of event in the root file : {len(full_data)}")

    if label == 1: # ES
        data = es_event_selection(root_path=root_file,
                                  awk_data=full_data,
                                  energy_thr=energy_thr,
                                  vtx_cut_mm=vtx_cut_mm,
                                  min_unique_pmts=min_unique_pmt)
        print(f" - {len(data)} ES event is selected")
    else: # BKG
        data = bkg_event_selection(root_path=root_file,
                                   awk_data=full_data,
                                   energy_thr=energy_thr,
                                   vtx_cut_mm=vtx_cut_mm,
                                   min_unique_pmts=min_unique_pmt)
        print(f" - {len(data)} 16N event is selected")

    return data

def save_to_h5(data, label, energy_thr, vtx_cut_mm, min_unique_pmt, output_path, h5_name):
    num_events = len(data)
    if num_events == 0:
        print(f"[Saver] No events to save")
        return

    channel_functions = [
        ("charge", get_charge_map),   # (채널이름, 해당 처리 함수)
        ("first_hit_time", get_first_hit_time_map),
        ("z_correction", get_z_correction),
        ("r_correction", get_r_correction),
    ]
    
    num_channels = len(channel_functions)
    side_height, cap_res = 45, 23
    H = side_height + (2 * cap_res)
    W = 142
   
    stats = {}


    h5_path = os.path.join(output_path, h5_name)
    with h5py.File(h5_path, 'w') as f:
        # Attributes 
        f.attrs['energy_thr'] = energy_thr
        f.attrs['vtx_cut_mm'] = vtx_cut_mm
        f.attrs['min_unique_pmt'] = min_unique_pmt

        # mcid: (N,)
        dset_mcid = f.create_dataset('mcid', (num_events,), dtype='i8')
        
        # label: (N,)
        dset_label = f.create_dataset('label', (num_events,), dtype='i8')
        
        # data: (N, num_channels, H, W) - Chunked & Compressed
        dset_input = f.create_dataset(
            'input', (num_events, num_channels, H, W), 
            dtype='f4', chunks=(1, num_channels, H, W), 
            compression="gzip"
        )

        # Channel Metadata 
        for i, (name, _) in enumerate(channel_functions):
            dset_input.attrs[f'ch{i}'] = name

        print(f"==== Start H5 saving (Number of channels : {num_channels}) ====")
        for i in tqdm(range(num_events)):
            event = data[i]
            
            # 한 이벤트에 대한 모든 채널 이미지를 담을 임시 배열 [num_channels, H, W]
            event_canvas = np.zeros((num_channels, H, W), dtype=np.float32)
            
            # 정의된 채널 함수들을 순회하며 이미지 생성
            for ch_idx, (name, func) in enumerate(channel_functions):
                img, ch_name, total_raw_hits, empty_hits = func(event, width=W, side_height=side_height, cap_res=cap_res)
                event_canvas[ch_idx] = img

                # 채널별 통계 업데이트 (첫 루프에서 이름 등록)
                if ch_name not in stats:
                    stats[ch_name] = {"total": 0, "missing": 0}

                stats[ch_name]["total"] += total_raw_hits
                stats[ch_name]["missing"] += empty_hits

            dset_input[i] = event_canvas
            dset_label[i] = label
            dset_mcid[i] = int(event["mcid"])
    
    return stats

def render_summary(name, num_events, stats):
    print("\n" + "="*65)
    print(f" PREPROCESSING SUMMARY: {name}")
    print(f" - Total Selected Events: {num_events}")
    print("-" * 65)
    print(f"{'Channel Name':<20} | {'Total Hits':>12} | {'Missing':>10} | {'Loss %':>10}")
    print("-" * 65)
    
    for ch_name, val in stats.items():
        total = val["total"]
        missing = val["missing"]
        loss_rate = (missing / total * 100) if total > 0 else 0
        print(f"{ch_name:<20} | {total:>12,} | {missing:>10,} | {loss_rate:>9.4f}%")
    print("="*65 + "\n")

if __name__ == "__main__":

    es_h5_name = "WIND_ES_with_rz_corrections.h5"
    n16_h5_name = "WIND_16N_with_rz_corrections.h5"

    # ES    
    es_data = event_selection(root_file=ES_FILE,
                              label=1,
                              energy_thr=ENERGY_CUT,
                              vtx_cut_mm=VERTEX_CUT, 
                              min_unique_pmt=MIN_UNIQUE_PMTS)

    es_stat = save_to_h5(data=es_data,
                         label=1,
                         energy_thr=ENERGY_CUT,
                         vtx_cut_mm=VERTEX_CUT,
                         min_unique_pmt=MIN_UNIQUE_PMTS,
                         output_path=OUTPUT_PATH,
                         h5_name=es_h5_name)
    
    render_summary(es_h5_name, len(es_data), es_stat)
    
    # 16N
    n16_data = event_selection(root_file=N16_FILE, 
                               label=0, 
                               energy_thr=ENERGY_CUT, 
                               vtx_cut_mm=VERTEX_CUT, 
                               min_unique_pmt=MIN_UNIQUE_PMTS)

    n16_stat = save_to_h5(data=n16_data,
                          label=0,
                          energy_thr=ENERGY_CUT,
                          vtx_cut_mm=VERTEX_CUT,
                          min_unique_pmt=MIN_UNIQUE_PMTS,
                          output_path=OUTPUT_PATH,
                          h5_name=n16_h5_name)
    
    render_summary(n16_h5_name, len(n16_data), n16_stat)