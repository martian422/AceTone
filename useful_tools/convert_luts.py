import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os

def read_cube_file(path):
    lut_size = None
    data = []
    header_lines = []

    with open(path, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                header_lines.append(line.rstrip("\n"))
                continue
            if line_stripped.upper().startswith('LUT_3D_SIZE'):
                lut_size = int(float(line_stripped.split()[1]))
                header_lines.append(f"LUT_3D_SIZE {lut_size}")
            else:
                parts = line_stripped.split()
                if len(parts) == 3:
                    try:
                        vals = [float(p) for p in parts]
                        data.append(vals)
                    except ValueError:
                        continue
                else:
                    header_lines.append(line.rstrip("\n"))

    if lut_size is None:
        raise ValueError(f"{path} 未找到 LUT_3D_SIZE")

    expected_len = lut_size ** 3
    if len(data) < expected_len:
        print(f"[警告] {path} 数据不足 {expected_len} 行，实际 {len(data)} 行 → 用 0 填充")
        for _ in range(expected_len - len(data)):
            data.append([0.0, 0.0, 0.0])
    elif len(data) > expected_len:
        print(f"[警告] {path} 数据超过 {expected_len} 行，实际 {len(data)} 行 → 截断")
        data = data[:expected_len]

    data = np.array(data).reshape((lut_size, lut_size, lut_size, 3))
    return lut_size, data, header_lines

def resize_lut(lut, target_size=32):
    orig_size = lut.shape[0]
    x = np.linspace(0, 1, orig_size)
    y = np.linspace(0, 1, orig_size)
    z = np.linspace(0, 1, orig_size)
    interp_func = RegularGridInterpolator((x, y, z), lut, bounds_error=False, fill_value=None)

    new_x = np.linspace(0, 1, target_size)
    new_y = np.linspace(0, 1, target_size)
    new_z = np.linspace(0, 1, target_size)
    grid = np.stack(np.meshgrid(new_x, new_y, new_z, indexing='ij'), -1)
    lut_resized = interp_func(grid)
    return lut_resized

def save_cube_file(path, lut, original_header):
    target_size = lut.shape[0]
    # 更新 header：LUT_3D_SIZE 改为 target_size
    new_header = []
    replaced = False
    for line in original_header:
        if line.upper().startswith("LUT_3D_SIZE"):
            new_header.append(f"LUT_3D_SIZE {target_size}")
            replaced = True
        else:
            new_header.append(line)
    if not replaced:
        new_header.append(f"LUT_3D_SIZE {target_size}")

    with open(path, 'w') as f:
        for line in new_header:
            f.write(f"{line}\n")
        for r in range(target_size):
            for g in range(target_size):
                for b in range(target_size):
                    rgb = lut[r, g, b]
                    f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

def save_npy_file(path, lut):
    np.save(path, lut)

def process_all_luts(input_dir, output_dir_cube, output_dir_npy, target_size=32, save_cube=False):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.cube'):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, input_dir)

                out_cube_path = os.path.join(output_dir_cube, rel_path)
                out_npy_path = os.path.join(output_dir_npy, rel_path)
                out_npy_path = os.path.splitext(out_npy_path)[0] + ".npy"

                if save_cube:
                    os.makedirs(os.path.dirname(out_cube_path), exist_ok=True)
                os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)

                try:
                    orig_size, lut, header = read_cube_file(in_path)
                    lut_resized = resize_lut(lut, target_size=target_size)
                    if save_cube:
                        save_cube_file(out_cube_path, lut_resized, header)
                    save_npy_file(out_npy_path, lut_resized)
                    if save_cube:
                        print(f"[OK] {rel_path} {orig_size}³ → {target_size}³ saved as .cube and .npy")
                    else:
                        print(f"[OK] {rel_path} {orig_size}³ → {target_size}³ saved as .npy")
                except Exception as e:
                    print(f"[Error] Failed to process {in_path}: {e}")

if __name__ == "__main__":
    folder_A = "datasets/LUTS-final/cube-3d-update"  # Source LUT folder
    folder_B = "datasets/LUTS-final/cube-3d-interpolate"  # Output .cube folder
    folder_C = "datasets/LUTS-final/cube-3d-npy"  # Output .npy folder
    process_all_luts(folder_A, folder_B, folder_C, target_size=32)
