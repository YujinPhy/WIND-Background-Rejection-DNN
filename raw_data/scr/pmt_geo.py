import math
from pathlib import Path

# ============================================================
# Geometry constants
# ============================================================

# Detector tank
TANK_R_INNER_MM = 3000.0
TANK_R_OUTER_MM = 3025.4
TANK_HALF_Z_OUTER_MM = 3025.4
TOTAL_LIQUID_HEIGHT_MM = 6000.0

# PMT model: wind_pmt_4in
# Updated geometry:
# z_edge  = [2.0, 2.0, 0.0, -85.0]
# z_inner = [1.0, 1.0, 0.0, -85.0]
#
# Placement reference:
#   PMT_FRONT_EDGE_Z_MM = +2.0 mm
#   PMT_BACK_EDGE_Z_MM  = -85.0 mm
#
# For INTERNAL mounting, PMT BACK EDGE touches the tank INNER wall.
PMT_FRONT_EDGE_Z_MM = 2.0
PMT_BACK_EDGE_Z_MM = -85.0
PMT_TOTAL_LENGTH_MM = PMT_FRONT_EDGE_Z_MM - PMT_BACK_EDGE_Z_MM  # 87.0 mm
PMT_GLASS_RADIUS_MM = 50.8

# Array settings from DetectorConstruction.cc (4in_40p)
TOP_BOTTOM_PITCH_MM = 135.0
SIDE_COUNT = 142
SIDE_COUNT_Z = 45
SIDE_SPACING_Z_MM = 133.0

OUTPUT = Path("PMTINFO.ratdb")


# ============================================================
# Formatting helpers
# ============================================================

def fmt_num(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    if "." not in s:
        s += ".0"
    return s


def fmt_arr(vals):
    out = []
    for v in vals:
        if isinstance(v, int):
            out.append(str(v))
        else:
            out.append(fmt_num(v))
    return "[" + ", ".join(out) + "]"


def write_table(f, name, x, y, z, dx, dy, dz, include_type=True):
    f.write("{\n")
    f.write(f'//// Total number of {name} entries : {len(x)}\n')
    f.write(f'"name": "{name}",\n')
    f.write('"valid_begin": [0,0],\n')
    f.write('"valid_end": [0,0],\n')
    f.write(f'"x": {fmt_arr(x)},\n')
    f.write(f'"y": {fmt_arr(y)},\n')
    f.write(f'"z": {fmt_arr(z)},\n')
    f.write(f'"dir_x": {fmt_arr(dx)},\n')
    f.write(f'"dir_y": {fmt_arr(dy)},\n')
    f.write(f'"dir_z": {fmt_arr(dz)}')
    if include_type:
        f.write(",\n")
        f.write(f'"type":  {fmt_arr([1] * len(x))}\n')
    else:
        f.write("\n")
    f.write("}\n\n")


# ============================================================
# Position builders
# ============================================================

def build_bottom_internal_table():
    xs_pmt, ys_pmt, zs_pmt = [], [], []

    # Bottom PMTs face +z (toward detector center)
    #
    # Tank inner wall plane:
    #   z_wall_inner = -TOTAL_LIQUID_HEIGHT_MM / 2
    #
    # For dir_z = +1:
    #   z_back = z_origin + PMT_BACK_EDGE_Z_MM
    #
    # Internal mounting rule:
    #   PMT back edge touches the inner wall
    #
    # So:
    #   z_origin = z_wall_inner - PMT_BACK_EDGE_Z_MM
    #            = z_wall_inner + 85 mm

    z_wall_inner = -TOTAL_LIQUID_HEIGHT_MM / 2.0
    z_pmt = z_wall_inner - PMT_BACK_EDGE_Z_MM

    nx = int(2 * TANK_R_INNER_MM / TOP_BOTTOM_PITCH_MM) + 1
    ny = int(2 * TANK_R_INNER_MM / TOP_BOTTOM_PITCH_MM) + 1

    for ix in range(nx):
        x = -TANK_R_INNER_MM + ix * TOP_BOTTOM_PITCH_MM
        for iy in range(ny):
            y = -TANK_R_INNER_MM + iy * TOP_BOTTOM_PITCH_MM

            if x * x + y * y <= TANK_R_INNER_MM * TANK_R_INNER_MM:
                xs_pmt.append(x)
                ys_pmt.append(y)
                zs_pmt.append(z_pmt)

    n = len(xs_pmt)
    dx = [0.0] * n
    dy = [0.0] * n
    dz = [1.0] * n

    return xs_pmt, ys_pmt, zs_pmt, dx, dy, dz


def build_top_internal_table():
    xs_pmt, ys_pmt, zs_pmt = [], [], []

    # Top PMTs face -z (toward detector center)
    #
    # Tank inner wall plane:
    #   z_wall_inner = +TOTAL_LIQUID_HEIGHT_MM / 2
    #
    # For dir_z = -1:
    #   z_back = z_origin + dir_z * PMT_BACK_EDGE_Z_MM
    #          = z_origin + (-1)*(-85)
    #          = z_origin + 85
    #
    # Internal mounting rule:
    #   z_back = z_wall_inner
    #
    # So:
    #   z_origin = z_wall_inner - 85 mm
    #            = z_wall_inner + PMT_BACK_EDGE_Z_MM

    z_wall_inner = TOTAL_LIQUID_HEIGHT_MM / 2.0
    z_pmt = z_wall_inner + PMT_BACK_EDGE_Z_MM

    nx = int(2 * TANK_R_INNER_MM / TOP_BOTTOM_PITCH_MM) + 1
    ny = int(2 * TANK_R_INNER_MM / TOP_BOTTOM_PITCH_MM) + 1

    for ix in range(nx):
        x = -TANK_R_INNER_MM + ix * TOP_BOTTOM_PITCH_MM
        for iy in range(ny):
            y = -TANK_R_INNER_MM + iy * TOP_BOTTOM_PITCH_MM

            if x * x + y * y <= TANK_R_INNER_MM * TANK_R_INNER_MM:
                xs_pmt.append(x)
                ys_pmt.append(y)
                zs_pmt.append(z_pmt)

    n = len(xs_pmt)
    dx = [0.0] * n
    dy = [0.0] * n
    dz = [-1.0] * n

    return xs_pmt, ys_pmt, zs_pmt, dx, dy, dz


def build_side_internal_table():
    xs_pmt, ys_pmt, zs_pmt = [], [], []
    dxs, dys, dzs = [], [], []

    # Side PMTs face inward
    #
    # inward unit vector  = (-cos, -sin, 0)
    # outward unit vector = ( cos,  sin, 0)
    #
    # Tank inner wall point:
    #   wall = outward * TANK_R_INNER_MM
    #
    # Local back edge is at z = -85 mm.
    # World back-edge point:
    #   p_back = p_origin + dir * PMT_BACK_EDGE_Z_MM
    #
    # Internal mounting rule:
    #   p_back = wall
    #
    # Since dir = inward = (-outward),
    #   p_origin = wall - dir * PMT_BACK_EDGE_Z_MM
    #            = wall - inward * (-85)
    #            = wall + inward * 85
    #
    # That means the PMT body extends inward from the side wall.

    for i in range(SIDE_COUNT):
        angle = 2.0 * math.pi * i / SIDE_COUNT
        ca = math.cos(angle)
        sa = math.sin(angle)

        # inward-facing direction
        dx = -ca
        dy = -sa
        dz = 0.0

        # point on inner cylindrical wall
        wall_x = TANK_R_INNER_MM * ca
        wall_y = TANK_R_INNER_MM * sa

        # origin so that PMT back edge touches inner wall
        pmt_x = wall_x - dx * PMT_BACK_EDGE_Z_MM
        pmt_y = wall_y - dy * PMT_BACK_EDGE_Z_MM

        for j in range(SIDE_COUNT_Z):
            z = -TOTAL_LIQUID_HEIGHT_MM / 2.0 + (j + 0.5) * SIDE_SPACING_Z_MM

            xs_pmt.append(pmt_x)
            ys_pmt.append(pmt_y)
            zs_pmt.append(z)

            dxs.append(dx)
            dys.append(dy)
            dzs.append(dz)

    return xs_pmt, ys_pmt, zs_pmt, dxs, dys, dzs


# ============================================================
# Main
# ============================================================

def main():
    xb_p, yb_p, zb_p, dxb_p, dyb_p, dzb_p = build_bottom_internal_table()
    xt_p, yt_p, zt_p, dxt_p, dyt_p, dzt_p = build_top_internal_table()
    xs_p, ys_p, zs_p, dxs_p, dys_p, dzs_p = build_side_internal_table()

    with OUTPUT.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated for wind_pmt_4in\n")
        f.write("// INTERNAL mounting version\n")
        f.write("// PMTs are mounted on the INNER tank surface.\n")
        f.write("// No cookie is used.\n")
        f.write("// PMT BACK edge (z = -85 mm in PMT local coordinates) touches the INNER wall.\n\n")

        write_table(
            f, "PMTINFO_BOTTOM_INTERNAL_4in_40p",
            xb_p, yb_p, zb_p,
            dxb_p, dyb_p, dzb_p,
            include_type=True
        )

        write_table(
            f, "PMTINFO_TOP_INTERNAL_4in_40p",
            xt_p, yt_p, zt_p,
            dxt_p, dyt_p, dzt_p,
            include_type=True
        )

        write_table(
            f, "PMTINFO_SIDE_INTERNAL_4in_40p",
            xs_p, ys_p, zs_p,
            dxs_p, dys_p, dzs_p,
            include_type=True
        )

    print(f"Wrote {OUTPUT}")
    print(f"Bottom PMTs : {len(xb_p)}")
    print(f"Top PMTs    : {len(xt_p)}")
    print(f"Side PMTs   : {len(xs_p)}")
    print(f"Total PMTs  : {len(xb_p) + len(xt_p) + len(xs_p)}")


if __name__ == "__main__":
    main()