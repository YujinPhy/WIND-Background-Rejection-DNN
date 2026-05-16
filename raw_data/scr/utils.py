import numpy as np
import uproot
import awkward as ak

# Geometry constants (mm)
TANK_RADIUS = 3000.0
Z_TOP = +3025.4
Z_BOT = -3025.4
TOPBOT_Z_TOL = 150
SIDE_R_TOL = 200
COORD_DECIMALS = 3

# ==== Helper Functions ====
def classify_pmt_type(x, y, z):
    """
    PMT 좌표 또는 PE 도달 좌표 기반 설치 영역(상단, 하단, 측면) 분류

    Args:
        x, y, z (ak.Array) : x, y, z 좌표 배열

    Returns:
        ak.Array: PMT 영역 인덱스 (0: 상단, 1: 하단, 2: 측면)
    """
    # r = ak.sqrt(x**2 + y**2)
    r = (x**2 + y**2)**0.5

    is_top = (z >= Z_TOP - TOPBOT_Z_TOL) & (z <= Z_TOP + TOPBOT_Z_TOL)
    is_bot = (z >= Z_BOT - TOPBOT_Z_TOL) & (z <= Z_BOT + TOPBOT_Z_TOL)
    is_side = (r >= TANK_RADIUS - SIDE_R_TOL) & (r <= TANK_RADIUS + SIDE_R_TOL)
    
    pmt_type = ak.where(is_top, 0, ak.where(is_bot, 1, 2))
    return pmt_type

def unique_pmt_count_per_event(x, y, z, decimals=3):
    """
    수만 개의 이벤트를 일괄 처리하여 이벤트당 유니크 PMT 개수를 반환
    """
    scale = 10 ** decimals
    # 2D 상태 그대로 정수화
    xi = ak.values_astype(ak.round(x * scale), "int64")
    yi = ak.values_astype(ak.round(y * scale), "int64")
    zi = ak.values_astype(ak.round(z * scale), "int64")

    OFFSET = 10_000_000
    BASE1 = 100_000_000
    BASE2 = 100_000_000_000_000

    pmt_id = (xi + OFFSET) + (yi + OFFSET) * BASE1 + (zi + OFFSET) * BASE2
    
    s = ak.sort(pmt_id, axis=1)
    n = ak.num(s, axis=1)
    diffs = s[:, 1:] != s[:, :-1]
    
    out = ak.sum(diffs, axis=1) + ak.values_astype(n > 0, "int64")
    
    return out 

def build_eventid_mask(event_ids: np.ndarray):
    """
    Creates a boolean lookup mask where mask[ID] is True if the ID exists.
    
    Args:
    event_ids (np.ndarray): 1D array containing selected event ID numbers.
    
    Returns:
    mask (np.ndarray or None): A boolean array where mask[ID] is True if ID was in event_ids.
    max_id (int): The maximum ID value found, used to define the mask's size.
    """
    if event_ids.size == 0:
        return None, -1

    max_id = int(event_ids.max())
    mask = np.zeros(max_id + 1, dtype=np.bool_)
    mask[event_ids] = True
    return mask, max_id

# ==== Event Selection ====
def get_selected_es_eventids_mask(es_root_path, energy_def, thr=0.8, is_total_e=False, mass_E=0.51099895):
    """
    Filters ES events by an energy threshold

    Args:
        es_root_path (str): Path to the ES ROOT file.
        energy_def (str): Reference for the cut ('kinetic' or 'total').
        thr (float): Energy threshold value (MeV).
        is_total_e (bool): True if 'mcke' in file is Total Energy (K + m). 
        
        False if it is already Kinetic Energy (K).
        mass_E (float): Rest mass of the particle (0.511 MeV for electron).

    Returns:
        mask (np.ndarray): Boolean array for eventID filtering.
        max_id (int): The highest eventID included in the mask.
        total_gen (int): Total number of events originally in the 'Gen' tree.
    """
    with uproot.open(es_root_path) as f:
        g = f["output"].arrays(["mcid", "mcke"], library="np") 

    ids = g["mcid"].astype(np.int64)
    raw_e = g["mcke"].astype(np.float64)

    if is_total_e: # Input is E_total
        e_total = raw_e
        e_kinetic = raw_e - mass_E
    else: # Input is E_kinetic
        e_total = raw_e + mass_E
        e_kinetic = raw_e

    if energy_def == "kinetic":
        eUse = e_kinetic
    elif energy_def == "total":
        eUse = e_total
    else:
        # Fallback to raw value if 'stored' or unknown string is passed
        eUse = raw_e
        print(f"[WARN] Unknown energy_def '{energy_def}'. Using raw 'mcke' values.")

    # Apply threshold and build mask
    sel = eUse >= thr
    sel_ids = np.sort(ids[sel])

    m, mx = build_eventid_mask(sel_ids)

    print(f" - Energy cut: thr={thr} MeV, criteria={energy_def}")
    print(f"     Input E  : {'Total' if is_total_e else 'Kinetic'} energy")
    print(f"     Selected : {sel_ids.size} / {ids.size} ({sel_ids.size/ids.size * 100:.2f}%)")
    
    return m, mx, int(ids.size)

def get_gen_vertex_mask_cylinder(root_path, det_radius_mm, det_half_height_mm, vtx_cut_mm):
    with uproot.open(root_path) as f:
        g = f["output"].arrays(["mcid", "mcx", "mcy", "mcz"], library="np")
    
    r_fiducial = (det_radius_mm - vtx_cut_mm)  # mm
    z_fiducial = (det_half_height_mm - vtx_cut_mm) # mm

    ids = g["mcid"].astype(np.int64) # mm
    x = g["mcx"].astype(np.float64) # mm
    y = g["mcy"].astype(np.float64) # mm
    z = g["mcz"].astype(np.float64) # mm

    r_sq = x**2 + y**2
    ok = (r_sq <= r_fiducial**2) & (np.abs(z) <= z_fiducial)
    
    sel_ids = np.sort(ids[ok])
    m, mx = build_eventid_mask(sel_ids)

    effective_height = 2.0 * z_fiducial
    volume_mm3 = np.pi * (r_fiducial**2) * effective_height
    volume_m3 = volume_mm3 / 1e9
    mass_ton = volume_m3 * 1.0  # 물 기준 (1 ton/m^3)

    print(f" - Vertex cut (Cylinder): Wall margin = {vtx_cut_mm} mm")
    print(f"     Fiducial R/Z    : R < {r_fiducial:.1f} mm, |Z| < {z_fiducial:.1f} mm")
    print(f"     Fiducial Volume : {volume_m3:.3f} m^3")
    print(f"     Fiducial Mass   : {mass_ton:.3f} tons (as Water)")
    print(f"     Selected        : {sel_ids.size} / {ids.size} ({sel_ids.size/ids.size*100:.2f}%)")
    
    return m, mx, int(ids.size), int(sel_ids.size)

def es_event_selection(root_path, awk_data, energy_thr, vtx_cut_mm, min_unique_pmts):
    event_ids = ak.to_numpy(awk_data["mcid"]).astype(int)
    e_mask, _, _ = get_selected_es_eventids_mask(root_path,
                                                 energy_def="total",
                                                 thr=energy_thr)
    e_passed_id = e_mask[event_ids]
    
    vtx_mask_lookup, _, _, _ = get_gen_vertex_mask_cylinder(root_path,
                                                            det_radius_mm=3000,
                                                            det_half_height_mm=3000,
                                                            vtx_cut_mm=vtx_cut_mm)
    vtx_passed_id = vtx_mask_lookup[event_ids]
    
    has_hits = ak.num(awk_data["mcPEHitTime"]) > 0

    uniq = unique_pmt_count_per_event(awk_data["mcPEx"], awk_data["mcPEy"], awk_data["mcPEz"], decimals=COORD_DECIMALS)
    passed_pmt_count = uniq >= min_unique_pmts

    final_mask = e_passed_id & vtx_passed_id & has_hits & passed_pmt_count

    filtered_data = awk_data[final_mask]
    return filtered_data

def bkg_event_selection(root_path, awk_data, energy_thr, vtx_cut_mm, min_unique_pmts):
    event_ids = ak.to_numpy(awk_data["mcid"]).astype(int)
    vtx_mask_lookup, _, _, _ = get_gen_vertex_mask_cylinder(root_path,
                                                            det_radius_mm=3000,
                                                            det_half_height_mm=3000,
                                                            vtx_cut_mm=vtx_cut_mm)
    vtx_passed_id = vtx_mask_lookup[event_ids]
    
    has_hits = ak.num(awk_data["mcPEHitTime"]) > 0

    uniq = unique_pmt_count_per_event(awk_data["mcPEx"], awk_data["mcPEy"], awk_data["mcPEz"], decimals=COORD_DECIMALS)
    passed_pmt_count = uniq >= min_unique_pmts

    final_mask = vtx_passed_id & has_hits & passed_pmt_count

    filtered_data = awk_data[final_mask]
    return filtered_data

# ==== Generate individual Channel map ====
def get_charge_map(event, width=142, side_height=45, cap_res=23):
    ch_name = "charge"

    x = ak.to_numpy(event["mcPEx"])
    y = ak.to_numpy(event["mcPEy"])
    z = ak.to_numpy(event["mcPEz"])
    charge = ak.to_numpy(event["mcPECharge"])
    
    z_min, z_max, r_max = -3000.0, 3000.0, 3000.0
    total_height = side_height + (2 * cap_res)

    total_hits = len(charge)
    if total_hits == 0:
        return np.zeros((total_height, width), dtype=np.float32), ch_name, 0, 0
    
    arr = np.zeros((total_height, width), dtype=np.float32)

    # 데이터 매핑 준비
    p_types = classify_pmt_type(x, y, z)
    phi = np.arctan2(y, x) % (2 * np.pi)
    phi_idx = (phi / (2 * np.pi) * (width - 1)).astype(int)
    phi_idx = np.clip(phi_idx, 0, width - 1)
    
    recorded_hits = 0
    for i in range(len(x)):
        pi = phi_idx[i]
        this_type = p_types[i]
        r = np.sqrt(x[i]**2 + y[i]**2)
        
        if this_type == 2:  # Side
            z_side_norm = (z[i] - z_min) / (z_max - z_min)
            zi = cap_res + int(z_side_norm * (side_height - 1))
        elif this_type == 0: # Top Cap
            ri = int((r / r_max) * (cap_res - 1))
            zi = (total_height - 1) - (cap_res - 1 - np.clip(ri, 0, cap_res - 1))
        else: # Bottom Cap
            ri = int((r / r_max) * (cap_res - 1))
            zi = np.clip(ri, 0, cap_res - 1)
        
        zi = np.clip(zi, 0, total_height - 1)

        # 데이터 채우기
        arr[zi, pi] += charge[i]
        recorded_hits += 1
    
    # 데이터 정규화
    ch_max = np.max(arr)
    if ch_max > 0:
        arr = arr / ch_max
        
    empty_hits = total_hits - recorded_hits

    return arr, ch_name, total_hits, empty_hits

def get_first_hit_time_map(event, width=142, side_height=45, cap_res=23):
    ch_name = "first_hit_time"

    x = ak.to_numpy(event["mcPEx"])
    y = ak.to_numpy(event["mcPEy"])
    z = ak.to_numpy(event["mcPEz"])
    pe_time = ak.to_numpy(event["mcPEHitTime"])
    
    z_min, z_max, r_max = -3000.0, 3000.0, 3000.0
    total_height = side_height + (2 * cap_res)


    total_hits = len(pe_time)
    if total_hits == 0:
        return np.zeros((total_height, width), dtype=np.float32), ch_name, 0, 0
    
    arr = np.zeros((total_height, width), dtype=np.float32)

    # 시간 채널 초기화 (나중에 min 비교를 위해 큰 값 사용)
    t0 = np.min(pe_time)
    arr[ :, :] = 999999.0 

    # 데이터 매핑 준비
    p_types = classify_pmt_type(x, y, z)
    phi = np.arctan2(y, x) % (2 * np.pi)
    phi_idx = (phi / (2 * np.pi) * (width - 1)).astype(int)
    phi_idx = np.clip(phi_idx, 0, width - 1)
    
    recorded_hits = 0
    for i in range(len(x)):
        pi = phi_idx[i]
        this_type = p_types[i]
        r = np.sqrt(x[i]**2 + y[i]**2)
        
        if this_type == 2:  # Side
            z_side_norm = (z[i] - z_min) / (z_max - z_min)
            zi = cap_res + int(z_side_norm * (side_height - 1))
        elif this_type == 0: # Top Cap
            ri = int((r / r_max) * (cap_res - 1))
            zi = (total_height - 1) - (cap_res - 1 - np.clip(ri, 0, cap_res - 1))
        else: # Bottom Cap
            ri = int((r / r_max) * (cap_res - 1))
            zi = np.clip(ri, 0, cap_res - 1)
        
        zi = np.clip(zi, 0, total_height - 1)

        # 데이터 채우기
        relative_t = pe_time[i] - t0
        if relative_t < arr[zi, pi]:
            arr[zi, pi] = relative_t
        recorded_hits += 1
    # 데이터 정규화
    hit_mask = arr[ :, :] < 999999.0
    
    # 현재 이벤트 내부의 max로 정규화
    valid_times = arr[hit_mask]
    if len(valid_times) > 0:
        t_max = np.max(valid_times)
        if t_max > 0:
            # 반전 정규화: 첫 히트(0ns) -> 1.0, 마지막 히트(t_max) -> 0.001 (또는 0.1)
            # 신호가 없는 곳은 0.0을 유지하도록 함
            arr[hit_mask] = 1.0 - (arr[hit_mask] / (t_max + 1e-9))
        else:
            # 모든 히트가 동시에 왔을 경우 (t_max=0)
            arr[hit_mask] = 1.0
            
    # 신호 없는 곳 초기화값(999999)을 0으로 청소
    arr[~hit_mask] = 0.0
    empty_hits = total_hits - recorded_hits

    return arr, ch_name, total_hits, empty_hits