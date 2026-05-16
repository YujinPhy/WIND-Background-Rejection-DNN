import h5py
import numpy as np
from tqdm import tqdm
import os
from pmt_geo import *
from utils import *
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data

# ==== Geometry & Coordinate Helpers ====
def get_full_pmt_geometry():
    """기존 build_table 함수들을 활용해 전체 PMT 좌표를 리턴"""
    xb, yb, zb, dxb, dyb, dzb = build_bottom_internal_table()
    xt, yt, zt, dxt, dyt, dzt = build_top_internal_table()
    xs, ys, zs, dxs, dys, dzs = build_side_internal_table()

    pos_x = xt + xb + xs
    pos_y = yt + yb + ys
    pos_z = zt + zb + zs
    
    dir_x = dxt + dxb + dxs
    dir_y = dyt + dyb + dys
    dir_z = dzt + dzb + dzs

    # 3. 배열 생성 (N, 3)
    pos = np.stack([pos_x, pos_y, pos_z], axis=1)
    dirs = np.stack([dir_x, dir_y, dir_z], axis=1)
    
    # Top: 0, Bottom: 1, Side: 2
    types = np.array(
        [0] * len(xt) +  # Top
        [1] * len(xb) +  # Bottom
        [2] * len(xs)    # Side
    )
    
    return pos, dirs, types

def cartesian_to_cylindrical(pmt_pos):
    """[x, y, z] -> [r, phi, z] (phi: 0 to 2pi)"""
    x, y, z = pmt_pos[:, 0], pmt_pos[:, 1], pmt_pos[:, 2]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return np.stack([r, phi, z], axis=1)

def get_pmt_id_map(pmt_pos):
    """PMT 좌표 기반 고유 ID 생성 및 인덱스 매핑 딕셔너리 리턴"""
    scale = 10**3
    OFFSET, BASE1, BASE2 = 10_000_000, 100_000_000, 100_000_000_000_000
    
    xi = np.round(pmt_pos[:, 0] * scale).astype(np.int64)
    yi = np.round(pmt_pos[:, 1] * scale).astype(np.int64)
    zi = np.round(pmt_pos[:, 2] * scale).astype(np.int64)
    
    ids = (xi + OFFSET) + (yi + OFFSET) * BASE1 + (zi + OFFSET) * BASE2
    return {pid: i for i, pid in enumerate(ids)}

# ==== Graph Connectivity (Edge Index) ====
def build_fixed_edge_index(pmt_cyl, k=6):
    """주기적 경계 조건을 고려한 고정 Edge Index 생성"""
    r, phi, z = pmt_cyl[:, 0], pmt_cyl[:, 1], pmt_cyl[:, 2]
    
    # kNN용 가상 좌표: 원통 옆면의 0-2pi 연결을 위해 cos, sin 사용
    # 상하단 캡의 거리감을 맞추기 위해 r_norm 적용
    r_max = np.max(r)
    z_max = np.abs(np.max(z))
    r_norm = r / r_max
    knn_coords = np.stack([
        r_norm * np.cos(phi), 
        r_norm * np.sin(phi), 
        z / z_max
    ], axis=1)

    knn = NearestNeighbors(n_neighbors=k+1).fit(knn_coords)
    _, indices = knn.kneighbors(knn_coords)

    source, target = [], []
    for i, neighbors in enumerate(indices):
        for n in neighbors:
            source.append(i)
            target.append(n)
    
    return torch.tensor([source, target], dtype=torch.long)

# ==== Graph Generation ====
def process_data_to_graphs(data, pmt_lookup, pmt_cyl, fixed_edges, label):
    graph_list = []
    num_pmts = len(pmt_cyl)
    
    # 정적 특징 (r, phi, z) 미리 텐서화
    static_x = torch.tensor(pmt_cyl, dtype=torch.float)
    
    # 룩업 변수들
    scale = 10**3
    OFFSET, BASE1, BASE2 = 10_000_000, 100_000_000, 100_000_000_000_000

    for i, event in enumerate(data):
        # 1. 기본 노드 행렬 초기화: [Q, T, r, phi, z]
        x = torch.zeros((num_pmts, 5), dtype=torch.float)
        x[:, 2:] = static_x
        
        # 2. 히트 데이터 추출
        hx, hy, hz = ak.to_numpy(event["mcPEx"]), ak.to_numpy(event["mcPEy"]), ak.to_numpy(event["mcPEz"])
        hq, ht = ak.to_numpy(event["mcPECharge"]), ak.to_numpy(event["mcPEHitTime"])
        
        if len(ht) == 0: continue
        t0 = np.min(ht)

        # 3. 히트 좌표를 PMT 인덱스로 변환하여 Q, T 채우기
        h_ids = (np.round(hx*scale).astype(np.int64)+OFFSET) + \
                (np.round(hy*scale).astype(np.int64)+OFFSET)*BASE1 + \
                (np.round(hz*scale).astype(np.int64)+OFFSET)*BASE2
        
        for pid, q_val, t_val in zip(h_ids, hq, ht):
            if pid in pmt_lookup:
                idx = pmt_lookup[pid]
                x[idx, 0] += np.log10(q_val + 1.0) # Charge (Log)
                x[idx, 1] = t_val - t0             # Relative Time
        
        # 4. PyG Data 객체 생성
        data = Data(x=x, edge_index=fixed_edges, y=torch.tensor([label], dtype=torch.long))
        graph_list.append(data)
        
        if i % 100 == 0:
            print(f" - Processing event {i}/{len(data)}", end='\r')
            
    return graph_list

# ==== Event Selection ====
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


# ==========================================
# 5. Execution Workflow
# ==========================================
def main_workflow(root_path, label, energy_thr=0.8):
    # 1. PMT 지오메트리 로드 및 그래프 장부(Edge Index) 생성
    print("Building Global Geometry...")
    pmt_pos, pmt_dirs, pmt_types = get_full_pmt_geometry()
    pmt_cyl = cartesian_to_cylindrical(pmt_pos)
    pmt_lookup = get_pmt_id_map(pmt_pos)
    fixed_edges = build_fixed_edge_index(pmt_cyl, k=6)
    
    # 2. 이벤트 셀렉션 (이전에 만든 함수)
    print("Selecting Events...")
    selected_data = event_selection(root_path, label, energy_thr, vtx_cut_mm=500, min_unique_pmt=10)
    
    # 3. 그래프 리스트 생성
    print(f"Generating Graphs for label {label}...")
    graphs = process_data_to_graphs(selected_data, pmt_lookup, pmt_cyl, fixed_edges, label)
    
    # 4. 결과 저장 (PyTorch 전용 포맷)
    # torch.save(graphs, f"dataset_label_{label}.pt")
    print(f"\nSuccessfully created {len(graphs)} graphs.")
    return graphs




















def build_fixed_edge_index(pmt_cyl, k=6):
    """
    Args:
        pmt_cyl: (N, 3) [r, phi, z] 배열 (phi 범위: 0 ~ 2*pi)
        k: 각 PMT당 연결할 이웃 수 (논문 기준 6)
    Returns:
        edge_index: torch.LongTensor (2, E)
    """
    r = pmt_cyl[:, 0]
    phi = pmt_cyl[:, 1]
    z = pmt_cyl[:, 2]

    # 1. kNN 연산용 가상 좌표 (knn_coords) 생성
    # 원통 표면의 주기성을 반영하기 위해 (cos phi, sin phi) 사용
    # z축은 거리 계산의 균형을 위해 스케일 조정 (예: 전체 높이 6000mm로 나눔)
    z_norm = z / 6000.0 
    knn_coords = np.stack([np.cos(phi), np.sin(phi), z_norm], axis=1)

    # 2. kNN 알고리즘 실행
    # k+1인 이유는 자기 자신(거리 0)이 포함되기 때문
    knn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(knn_coords)
    distances, indices = knn.kneighbors(knn_coords)

    # 3. Edge List (source, target) 구성
    source_nodes = []
    target_nodes = []

    for i in range(len(pmt_cyl)):
        # i번째 PMT의 이웃들(indices[i])을 연결
        for neighbor_idx in indices[i]:
            source_nodes.append(i)
            target_nodes.append(neighbor_idx)

    # PyTorch Geometric 형식으로 변환
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return edge_index



"""
1. PMT의 fixed 좌표, 방향, 타입 추출 
2. 원통형 좌표계로 변환(node 연결 시 각도 연속성을 위해)
3. Edge 연결

Node Features
 - Position (x,y,z)
 - PMT Type
 - PMT Charge
 - PMT Hit Times


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
ES_FILE = "/home/yujin/projects/wind/WIND_bkg_rejection/ROOT_FILEs/WIND_66_4in_40p_ES_10k_internal_PMT.ntuple.root"
N16_FILE = "/home/yujin/projects/wind/WIND_bkg_rejection/ROOT_FILEs/WIND_66_4in_40p_16N_10k_internal.ntuple.root"

OUTPUT_PATH = "/home/yujin/projects/wind/WIND_bkg_rejection/ROOT_FILEs"
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
    print(f"📊 PREPROCESSING SUMMARY: {name}")
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
                         h5_name="WIND_ES.h5")
    
    render_summary("WIND_ES.h5", len(es_data), es_stat)
    
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
                          h5_name="WIND_16N.h5")
    
    render_summary("WIND_16N.h5", len(n16_data), n16_stat)