import torch
import torch.optim as optim
import argparse
import numpy as np
import os
import datetime
import mrcfile
import re
import torch.nn.functional as F

# --- Atom Parsers ---
def parse_structure_file(file_path):
    """Robustly parses a PDB or CIF file to get structured atom data including element type."""
    atom_data = []
    try:
        with open(file_path, 'r') as f:
            file_lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 结构文件未找到于 {file_path}"); return None

    is_cif = file_path.lower().endswith('.cif')
    if is_cif:
        header, data_lines = [], []
        in_loop, header_done = False, False
        for line in file_lines:
            s = line.strip()
            if s == 'loop_': in_loop = True; header_done = False; header = []
            elif s.startswith('_atom_site.') and in_loop: header.append(s)
            elif in_loop and s and not s.startswith('#') and not s.startswith('_'):
                header_done = True; data_lines.append(s)
        
        if not header or not data_lines: return None
        col_map = {name: i for i, name in enumerate(header)}
        x_col, y_col, z_col = col_map.get('_atom_site.Cartn_x'), col_map.get('_atom_site.Cartn_y'), col_map.get('_atom_site.Cartn_z')
        atom_col, chain_col = col_map.get('_atom_site.label_atom_id'), col_map.get('_atom_site.auth_asym_id')
        res_seq_col, res_name_col = col_map.get('_atom_site.auth_seq_id'), col_map.get('_atom_site.label_comp_id')
        type_symbol_col = col_map.get('_atom_site.type_symbol')

        if any(c is None for c in [x_col, y_col, z_col, atom_col, chain_col, res_seq_col, res_name_col]): return None
        
        for line in data_lines:
            try:
                parts = line.split()
                element = ''
                if type_symbol_col is not None and len(parts) > type_symbol_col:
                    element = parts[type_symbol_col].strip()
                if not element:
                    element = re.sub(r'[^A-Za-z]', '', parts[atom_col])[0]

                atom_data.append({
                    'chain': parts[chain_col], 'res_seq': int(parts[res_seq_col]), 'res_name': parts[res_name_col],
                    'atom_name': parts[atom_col].strip('"'), 'coords': [float(parts[x_col]), float(parts[y_col]), float(parts[z_col])],
                    'element': element
                })
            except (ValueError, IndexError): continue
    else: # PDB
        for line in file_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    element = line[76:78].strip()
                    if not element:
                        element = line[12:16].strip().lstrip('0123456789')
                        element = re.sub(r'[^A-Za-z]', '', element)[0]

                    atom_data.append({
                        'chain': line[21], 'res_seq': int(line[22:26]), 'res_name': line[17:20].strip(),
                        'atom_name': line[12:16].strip(), 'coords': [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        'element': element
                    })
                except (ValueError, IndexError): continue
    
    print(f"从 '{os.path.basename(file_path)}' 解析了 {len(atom_data)} 个原子。")
    if not atom_data: return None
    return atom_data

# --- Map Processing & Voxelization ---
def parse_mrc(file_path):
    try:
        with mrcfile.open(file_path, permissive=True) as mrc: 
            native_dtype = mrc.data.dtype.newbyteorder('='); data = mrc.data.astype(native_dtype)
            map_data = torch.tensor(data, dtype=torch.float32); hdr = mrc.header
            if not (hdr.mx>0 and hdr.my>0 and hdr.mz>0 and hdr.cella.x>0 and hdr.cella.y>0 and hdr.cella.z>0): return None, None, None
            vx=float(hdr.cella.x)/float(hdr.mx); vy=float(hdr.cella.y)/float(hdr.my); vz=float(hdr.cella.z)/float(hdr.mz)
            voxel_size = torch.tensor([vx, vy, vz], dtype=torch.float32)
            origin = torch.tensor([float(hdr.origin.x), float(hdr.origin.y), float(hdr.origin.z)], dtype=torch.float32)
            print(f"密度图 '{os.path.basename(file_path)}' 已加载。 维度: {map_data.shape}, 体素大小: {voxel_size.numpy()} Å, 文件头原点: {origin.numpy()} Å")
            return map_data, voxel_size, origin
    except Exception as e: print(f"解析MRC文件时发生意外错误: {e}"); return None, None, None

def voxelize_structure(coords, atom_weights, map_shape, voxel_size, map_origin):
    """使用三线性插值（splatting）并根据原子权重将结构可微地体素化到网格上。"""
    device = coords.device
    volume = torch.zeros(map_shape, dtype=torch.float32, device=device)
    
    c_v = (coords - map_origin) / voxel_size
    c0 = torch.floor(c_v).long()
    f = c_v - c0.float()

    corners = [
        c0, c0 + torch.tensor([1, 0, 0], device=device, dtype=torch.long), c0 + torch.tensor([0, 1, 0], device=device, dtype=torch.long),
        c0 + torch.tensor([1, 1, 0], device=device, dtype=torch.long), c0 + torch.tensor([0, 0, 1], device=device, dtype=torch.long),
        c0 + torch.tensor([1, 0, 1], device=device, dtype=torch.long), c0 + torch.tensor([0, 1, 1], device=device, dtype=torch.long),
        c0 + torch.tensor([1, 1, 1], device=device, dtype=torch.long)
    ]

    interp_weights = [
        (1 - f[:, 0]) * (1 - f[:, 1]) * (1 - f[:, 2]),
        f[:, 0] * (1 - f[:, 1]) * (1 - f[:, 2]),
        (1 - f[:, 0]) * f[:, 1] * (1 - f[:, 2]),
        f[:, 0] * f[:, 1] * (1 - f[:, 2]),
        (1 - f[:, 0]) * (1 - f[:, 1]) * f[:, 2],
        f[:, 0] * (1 - f[:, 1]) * f[:, 2],
        (1 - f[:, 0]) * f[:, 1] * f[:, 2],
        f[:, 0] * f[:, 1] * f[:, 2]
    ]

    dims = torch.tensor([map_shape[2], map_shape[1], map_shape[0]], device=device, dtype=torch.long)

    for corner_coords, interp_w in zip(corners, interp_weights):
        valid_mask = (corner_coords >= 0).all(dim=1) & (corner_coords < dims).all(dim=1)
        
        corner_valid = corner_coords[valid_mask]
        interp_w_valid = interp_w[valid_mask]
        atom_w_valid = atom_weights[valid_mask]
        
        value_to_add = interp_w_valid * atom_w_valid
        
        if corner_valid.shape[0] > 0:
            flat_indices = corner_valid[:, 2] * (map_shape[1] * map_shape[2]) + \
                           corner_valid[:, 1] * map_shape[2] + \
                           corner_valid[:, 0]
            volume.view(-1).index_add_(0, flat_indices, value_to_add)

    if volume.sum() > 0:
        volume /= volume.sum()
        
    return volume

# --- Sinkhorn-FFT Core Algorithm ---
def create_tm_fft_kernel(shape, d0, voxel_size, device):
    """创建用于FFT卷积的TM-align风格核的傅里叶变换。"""
    D, H, W = shape
    vx, vy, vz = voxel_size[0], voxel_size[1], voxel_size[2]
    
    x = torch.arange(W, device=device); y = torch.arange(H, device=device); z = torch.arange(D, device=device)
    
    x_sq = (torch.min(x, W - x) * vx)**2
    y_sq = (torch.min(y, H - y) * vy)**2
    z_sq = (torch.min(z, D - z) * vz)**2
    
    cost_sq = x_sq.view(1, 1, -1) + y_sq.view(1, -1, 1) + z_sq.view(-1, 1, 1)
    
    kernel = 1.0 / (1.0 + cost_sq / (d0**2))
    
    return torch.fft.rfftn(kernel)

def sinkhorn_iterations_fft(p, q, k_fft, num_iters=10):
    """使用FFT进行Sinkhorn迭代。"""
    v = torch.ones_like(q)
    p_shape = p.shape
    for _ in range(num_iters):
        K_v = torch.fft.irfftn(torch.fft.rfftn(v) * k_fft, s=p_shape)
        K_v[K_v < 1e-9] = 1e-9
        u = p / K_v
        
        K_u = torch.fft.irfftn(torch.fft.rfftn(u) * k_fft, s=p_shape)
        K_u[K_u < 1e-9] = 1e-9
        v = q / K_u
    return u, v

# --- Transformation & Other Helpers ---
def get_transformation_matrix(params):
    w, u = params[:3], params[3:]; W = torch.zeros((3,3),dtype=params.dtype,device=params.device); W[0,1],W[0,2]=-w[2],w[1]; W[1,0],W[1,2]=w[2],-w[0]; W[2,0],W[2,1]=-w[1],w[0]
    return torch.linalg.matrix_exp(W), u

def calculate_rmsd(c1, c2): 
    return torch.sqrt(torch.mean(torch.sum((c1 - c2)**2, dim=1)))

# --- Main Execution ---
def main():
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="使用受TM-align启发的Sinkhorn-FFT将结构拟合到Cryo-EM密度图中。")
    parser.add_argument("--mobile_structure", required=True, help="需要拟合的移动结构文件 (.cif 或 .pdb)。")
    parser.add_argument("--target_map", required=True, help="目标Cryo-EM密度图文件 (.mrc)。")
    parser.add_argument("--gold_standard_structure", required=True, help="用于计算最终RMSD的金标准结构文件。")
    parser.add_argument("--output", default=None, help="保存拟合后的PDB结构的路径。")
    parser.add_argument("--lr", type=float, default=0.01, help="优化学习率。")
    parser.add_argument("--steps", type=int, default=100, help="优化步数。")
    parser.add_argument("--d0", type=float, default=8.0, help="TM-align风格打分中的d0参数。")
    parser.add_argument("--score_scale", type=float, default=10000.0, help="对最终得分进行缩放的系数。")
    parser.add_argument("--sinkhorn_iter", type=int, default=10, help="内部Sinkhorn迭代次数。")
    parser.add_argument("--sigma_level", type=float, default=3.0, help="用于目标图阈值处理的Sigma水平。")
    parser.add_argument("--mobile_sigma_level", type=float, default=1.0, help="用于移动结构体素阈值处理的Sigma水平。")
    args = parser.parse_args()

    print(f"\n--- Cryo-EM Sinkhorn-FFT 拟合程序 (v4: 得分缩放) ---\n程序开始于: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- 数据加载和准备 ---
    mobile_data = parse_structure_file(args.mobile_structure)
    gold_data = parse_structure_file(args.gold_standard_structure)
    if not mobile_data or not gold_data: return

    ATOMIC_NUMBERS = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16}
    all_mobile_coords = torch.tensor([a['coords'] for a in mobile_data], dtype=torch.float32)
    atom_weights = torch.tensor([ATOMIC_NUMBERS.get(a['element'].upper(), 6.0) for a in mobile_data], dtype=torch.float32)
    print(f"已为 {len(atom_weights)} 个原子创建了基于原子类型的权重。")

    mobile_ca_map = {(a['chain'], a['res_seq']): a['coords'] for a in mobile_data if a['atom_name'] == 'CA'}
    gold_ca_map = {(a['chain'], a['res_seq']): a['coords'] for a in gold_data if a['atom_name'] == 'CA'}
    common_keys = sorted(list(mobile_ca_map.keys() & gold_ca_map.keys()))
    print(f"在两个结构中找到了 {len(common_keys)} 个共同的C-alpha原子用于计算最终RMSD。")
    if not common_keys: return
    common_mobile_ca_coords = torch.tensor([mobile_ca_map[k] for k in common_keys], dtype=torch.float32)
    common_gold_ca_coords = torch.tensor([gold_ca_map[k] for k in common_keys], dtype=torch.float32)

    map_data, voxel_size, origin = parse_mrc(args.target_map)
    if map_data is None: return

    # --- 目标图预处理 (p) ---
    print(f"\n--- 正在预处理目标密度图 (sigma_level={args.sigma_level}) ---")
    positive_densities = map_data[map_data > 0]
    if positive_densities.numel() == 0: print("错误: 密度图中找不到任何正值。"); return
    map_mean, map_std = positive_densities.mean(), positive_densities.std()
    density_threshold = map_mean + args.sigma_level * map_std
    mask = map_data > density_threshold
    p = map_data.clone(); p[~mask] = 0
    if p.sum() > 0: p /= p.sum()
    else: print("错误: 阈值化后目标密度图总和为零。"); return

    # --- 预对齐 ---
    target_indices = mask.nonzero(as_tuple=False)
    target_coords = target_indices.float()[:, [2, 1, 0]] * voxel_size + origin
    struct_center = all_mobile_coords.mean(dim=0)
    map_com_angstrom = target_coords.mean(dim=0)
    initial_t = map_com_angstrom - struct_center
    print(f"--- 预对齐 (结构质心 -> 阈值化后密度图质心) ---\n  - 计算得到的初始平移 (Å): {initial_t.numpy()}")
    
    transform_params = torch.zeros(6, requires_grad=True)
    with torch.no_grad(): transform_params[3:] = initial_t
    
    optimizer = optim.Adam([transform_params], lr=args.lr)
    best_rmsd = float('inf')
    best_transform_params = transform_params.detach().clone()

    # --- 创建TM-align风格的卷积核 ---
    d0 = args.d0
    print(f"\n正在使用手动设置的d0创建TM-align风格的FFT核: {d0:.4f}")
    k_fft = create_tm_fft_kernel(p.shape, d0, voxel_size, p.device)
    print("FFT核已创建。")

    print(f"\n--- 开始Sinkhorn-FFT优化 ({args.steps} 步, lr={args.lr}) ---")
    for step in range(args.steps):
        optimizer.zero_grad()
        R, t = get_transformation_matrix(transform_params)
        transformed_coords = (R @ all_mobile_coords.T).T + t
        q = voxelize_structure(transformed_coords, atom_weights, p.shape, voxel_size, origin)
        
        if args.mobile_sigma_level > 0:
            pos_q = q[q > 0]
            if pos_q.numel() > 0:
                q_mean, q_std = pos_q.mean(), pos_q.std()
                q_thresh = q_mean + args.mobile_sigma_level * q_std
                q[q < q_thresh] = 0

        if q.sum() > 0: q /= q.sum()
        else: 
            print(f"  步骤 {step:04d}: 结构被完全过滤，跳过。")
            continue

        u, v = sinkhorn_iterations_fft(p, q, k_fft, num_iters=args.sinkhorn_iter)
        K_u = torch.fft.irfftn(torch.fft.rfftn(u) * k_fft, s=p.shape)
        
        raw_score = torch.sum(q * K_u)
        score = raw_score * args.score_scale

        loss = -score
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == args.steps - 1:
            with torch.no_grad():
                transformed_common_ca = (R @ common_mobile_ca_coords.T).T + t
                current_rmsd = calculate_rmsd(transformed_common_ca, common_gold_ca_coords)
                print(f"  步骤 {step:04d}: TM-style得分 = {score.item():.4f} (raw: {raw_score.item():.6f}), RMSD = {current_rmsd.item():.4f} Å")
                
                if current_rmsd < best_rmsd:
                    best_rmsd = current_rmsd.item()
                    best_transform_params = transform_params.detach().clone()
                    print(f"          *** 新的最佳RMSD: {best_rmsd:.4f} Å ***")

    print("\n--- 优化完成 ---")
    R_final, t_final = get_transformation_matrix(best_transform_params)
    
    final_coords = (R_final @ all_mobile_coords.T).T + t_final
    final_rmsd = best_rmsd
    print(f"\n最终最佳RMSD (对比 '{os.path.basename(args.gold_standard_structure)}' 的CA原子): {final_rmsd:.4f} Å")

    output_filename = args.output if args.output else f"{os.path.splitext(os.path.basename(args.mobile_structure))[0]}_tm_fft_v4_rmsd_{final_rmsd:.2f}.pdb"
    print(f"\n--- 正在将拟合后的PDB写入 '{output_filename}' ---")
    with open(output_filename, 'w') as f:
        for i, atom in enumerate(mobile_data):
            x,y,z = final_coords[i]
            f.write(f"ATOM  {i+1:5d} {atom['atom_name']:<4s} {atom['res_name']:<3s} {atom['chain']}{atom['res_seq']:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")

    print(f"成功！您现在可以一起查看 '{args.target_map}' 和 '{output_filename}'。")
    print(f"\n程序结束于: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n总执行时间: {datetime.datetime.now() - start_time}")

if __name__ == '__main__':
    main()
