node = hou.pwd()
geo = node.geometry()

# Add code to modify contents of geo.
# Use drop down menu to select examples.
"""
Robust Weight Transfer - Houdini Python SOP 移植版

元: Robust Weight Transfer for Blender (https://github.com/sentfromspacevr/robust-weight-transfer)
Academic: "Robust Skin Weights Transfer via Weight Inpainting" - SIGGRAPH ASIA 2023

Input 0: Source geometry (body) - boneCapture を unpack した状態
Input 1: Target geometry (clothing) - ウェイト転送先

必要なパッケージ: numpy, scipy
オプション（爆速化）: pykdtree, pypardiso
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from scipy.spatial import cKDTree
    from scipy import sparse
    from scipy.sparse import linalg as splinalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from pykdtree.kdtree import KDTree as PyKDTree
    _PYKDTREE_AVAILABLE = True
except ImportError:
    PyKDTree = None
    _PYKDTREE_AVAILABLE = False

try:
    import pypardiso
    _PYPARDISO_AVAILABLE = True
except ImportError:
    _PYPARDISO_AVAILABLE = False

# 実測で遅くなったため OFF。必要なら True に変更して試す。
HAS_PYKDTREE = False
HAS_PYPARDISO = False

import hou


def _progress(msg, pct=None, bar_len=30):
    """コンソールにプログレス表示（Houdini Shell に即時出力）"""
    if pct is not None:
        filled = int(bar_len * pct)
        bar = "[" + "=" * filled + " " * (bar_len - filled) + "] {:5.1f}%".format(pct * 100)
        print("{} {}".format(msg, bar), flush=True)
    else:
        print("{}".format(msg), flush=True)


# =============================================================================
# Point-to-Triangle 最近接点 (libigl 代替) - ベクトル化で高速化
# =============================================================================

def _point_to_triangles_sq_batch(P, A, B, C):
    """
    点Pから複数三角形への最近接点を一括計算（NumPy ベクトル化）
    - P:(3,), A,B,C:(N,3) -> P から N 三角形へ
    - P:(M,3), A,B,C:(M,3) -> ペアごと P[i] から 三角形 i へ
    """
    P = np.asarray(P, dtype=np.float64)
    A, B, C = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), np.asarray(C, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(1, 3)
    if P.shape[0] == 1 and A.shape[0] > 1:
        P = np.broadcast_to(P, (A.shape[0], 3))
    AP = P - A
    AB = B - A
    AC = C - A

    d1 = np.sum(AB * AP, axis=1)
    d2 = np.sum(AC * AP, axis=1)
    d3 = np.sum(AB * (P - B), axis=1)
    d4 = np.sum(AC * (P - B), axis=1)
    d5 = np.sum(AB * (P - C), axis=1)
    d6 = np.sum(AC * (P - C), axis=1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-14, 1.0, denom)
    inv_denom = 1.0 / denom

    u = vb * inv_denom
    v = vc * inv_denom
    w = 1 - u - v

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    s = u + v + w
    s = np.where(s < 1e-14, 1.0, s)
    u, v, w = u / s, v / s, w / s

    C_out = u[:, np.newaxis] * A + v[:, np.newaxis] * B + w[:, np.newaxis] * C
    diff = P - C_out
    sqrD = np.sum(diff * diff, axis=1)
    Bary = np.stack([u, v, w], axis=1)
    return sqrD, C_out, Bary




def _build_vertex_to_triangles(F):
    """頂点インデックス -> 接続三角形インデックスのリスト（後方互換）"""
    from collections import defaultdict
    v2t = defaultdict(list)
    for ti, t in enumerate(F):
        for v in t:
            v2t[int(v)].append(ti)
    return v2t


def _build_v2t_numpy(F, n_verts):
    """CSR スタイルで v2t を numpy 配列として返す（高速検索用）。sorted_tris[splits[v]:splits[v+1]] が頂点 v の三角形群"""
    n_faces = F.shape[0]
    tri_ids = np.repeat(np.arange(n_faces, dtype=np.int32), 3)
    vert_ids = F.flatten()
    order = np.argsort(vert_ids, kind="stable")
    sorted_tris = tri_ids[order]
    sorted_verts = vert_ids[order]
    splits = np.searchsorted(sorted_verts, np.arange(n_verts + 1))
    return sorted_tris, splits


def find_closest_point_on_surface(P, V, F, progress_cb=None):
    """
    メッシュ表面上の最近接点を取得（libigl.point_mesh_squared_distance 代替）
    P: #P x 3, V: #V x 3, F: #F x 3
    返値: sqrD, I, C, B
    """
    n_points = P.shape[0]
    n_faces = F.shape[0]
    prog = progress_cb or (lambda *a: None)
    
    if n_points == 0:
        sqrD = np.array([], dtype=np.float64)
        I = np.array([], dtype=np.int32)
        C = np.zeros((0, 3), dtype=np.float64)
        B = np.zeros((0, 3), dtype=np.float64)
        return sqrD, I, C, B
    if n_faces == 0:
        sqrD = np.full(n_points, np.inf, dtype=np.float64)
        I = np.zeros(n_points, dtype=np.int32)
        C = np.zeros((n_points, 3), dtype=np.float64)
        B = np.zeros((n_points, 3), dtype=np.float64)
        return sqrD, I, C, B
    
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    
    tri_verts = V[F]  # (n_faces, 3, 3)
    
    if HAS_SCIPY and n_faces > 50:
        k_nn = min(12, V.shape[0])
        prog("  空間インデックス構築（k={} 近傍頂点）...".format(k_nn))
        sorted_tris, splits = _build_v2t_numpy(F, V.shape[0])
        if HAS_PYKDTREE:
            tree = PyKDTree(np.ascontiguousarray(V, dtype=np.float64))
            _, nearest_k = tree.query(np.ascontiguousarray(P, dtype=np.float64), k=k_nn)
        else:
            tree = cKDTree(V)
            try:
                _, nearest_k = tree.query(P, k=k_nn, workers=-1)
            except TypeError:
                _, nearest_k = tree.query(P, k=k_nn)
        if nearest_k.ndim == 1:
            nearest_k = nearest_k.reshape(-1, 1)
        max_tris = 96
        candidates = np.full((n_points, max_tris), -1, dtype=np.int32)
        for i in range(n_points):
            tri_list = np.concatenate([sorted_tris[splits[v]:splits[v + 1]] for v in nearest_k[i]])
            tris = np.unique(tri_list) if len(tri_list) > 0 else np.array([0], dtype=np.int32)
            n = min(len(tris), max_tris)
            if n > 0:
                candidates[i, :n] = tris[:n]
                candidates[i, n:] = tris[0]
            else:
                candidates[i, 0] = 0

        n_filled = (candidates >= 0).sum(axis=1)
        actual_max = max(1, int(np.minimum(np.max(n_filled), max_tris)))
        cand_use = candidates[:, :actual_max]

        sqrD = np.full(n_points, np.inf)
        I = np.zeros(n_points, dtype=np.int32)
        C_arr = np.zeros((n_points, 3))
        B_arr = np.zeros((n_points, 3))

        prog("  最近接点検索中（真の最近接を近似）...")
        batch_size = 4000
        n_batches = (n_points + batch_size - 1) // batch_size
        for bi, start in enumerate(range(0, n_points, batch_size)):
            end = min(start + batch_size, n_points)
            bs = end - start
            if n_batches > 5 and bi % max(1, n_batches // 8) == 0:
                prog("  最近接点検索...", end / n_points)
            cand_batch = cand_use[start:end]
            P_batch = P[start:end]
            tri_inds = cand_batch.flatten()
            tri_inds = np.maximum(tri_inds, 0)
            tris = tri_verts[tri_inds]
            P_exp = np.repeat(P_batch, actual_max, axis=0)
            A, B_pts, C_pts = tris[:, 0], tris[:, 1], tris[:, 2]
            s_all, c_all, b_all = _point_to_triangles_sq_batch(P_exp, A, B_pts, C_pts)
            s_all = s_all.reshape(bs, actual_max)
            c_all = c_all.reshape(bs, actual_max, 3)
            b_all = b_all.reshape(bs, actual_max, 3)
            best_j = np.argmin(s_all, axis=1)
            sqrD[start:end] = s_all[np.arange(bs), best_j]
            I[start:end] = cand_batch[np.arange(bs), best_j]
            C_arr[start:end] = c_all[np.arange(bs), best_j]
            B_arr[start:end] = b_all[np.arange(bs), best_j]
        C, B = C_arr, B_arr
    else:
        prog("  最近接点検索中（全三角形・ベクトル化）...")
        sqrD = np.full(n_points, np.inf)
        I = np.zeros(n_points, dtype=np.int32)
        C = np.zeros((n_points, 3))
        B = np.zeros((n_points, 3))
        A_all, B_all, C_all = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
        batch_size = min(500, max(100, 500000 // max(n_faces, 1)))
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            bs = end - start
            if (start // batch_size) % 4 == 0:
                prog("  最近接点検索...", end / n_points)
            P_batch = P[start:end]
            P_exp = np.repeat(P_batch, n_faces, axis=0)
            A_exp = np.tile(A_all, (bs, 1))
            B_exp = np.tile(B_all, (bs, 1))
            C_exp = np.tile(C_all, (bs, 1))
            s_all, c_all, b_all = _point_to_triangles_sq_batch(P_exp, A_exp, B_exp, C_exp)
            s_all = s_all.reshape(bs, n_faces)
            c_all = c_all.reshape(bs, n_faces, 3)
            b_all = b_all.reshape(bs, n_faces, 3)
            best_j = np.argmin(s_all, axis=1)
            sqrD[start:end] = s_all[np.arange(bs), best_j]
            I[start:end] = best_j
            C[start:end] = c_all[np.arange(bs), best_j]
            B[start:end] = b_all[np.arange(bs), best_j]

    return sqrD, I, C, B


def interpolate_attribute_from_bary(A, B, I, F):
    """重心座標で頂点属性を補間"""
    F_closest = F[I, :]
    a1 = A[F_closest[:, 0], :]
    a2 = A[F_closest[:, 1], :]
    a3 = A[F_closest[:, 2], :]
    b1 = B[:, 0].reshape(-1, 1)
    b2 = B[:, 1].reshape(-1, 1)
    b3 = B[:, 2].reshape(-1, 1)
    return a1 * b1 + a2 * b2 + a3 * b3


def find_matches_closest_surface(source_verts, source_triangles, source_normals,
                                  target_verts, target_normals, source_weights,
                                  dist_threshold_sq, angle_threshold_deg, flip_vertex_normal,
                                  progress_cb=None, source_triangles_vtx=None):
    """ソースメッシュ上でターゲット頂点のマッチングを行い、補間ウェイトを返す。
    source_triangles_vtx: Vertex キャプチャ時は三角形の頂点インデックス (n_tris, 3)
    """
    sqrD, I, C, B = find_closest_point_on_surface(
        target_verts, source_verts, source_triangles, progress_cb=progress_cb
    )
    F_for_attr = source_triangles_vtx if source_triangles_vtx is not None else source_triangles
    W2 = interpolate_attribute_from_bary(source_weights, B, I, F_for_attr)
    dominant = np.argmax(B, axis=1)
    use_direct = (B[np.arange(len(B)), dominant] > 0.9999) | (sqrD < 1e-12)
    if np.any(use_direct):
        F_closest = F_for_attr[I]
        vid_direct = F_closest[np.arange(len(F_closest)), dominant]
        W2[use_direct] = source_weights[vid_direct[use_direct]]
    N1_interp = interpolate_attribute_from_bary(source_normals, B, I, source_triangles)
    
    n1 = np.linalg.norm(N1_interp, axis=1, keepdims=True)
    n2 = np.linalg.norm(target_normals, axis=1, keepdims=True)
    n1 = np.where(n1 < 1e-10, 1, n1)
    n2 = np.where(n2 < 1e-10, 1, n2)
    norm_N1 = N1_interp / n1
    norm_N2 = target_normals / n2
    
    dot = np.einsum('ij,ij->i', norm_N1, norm_N2)
    dot = np.clip(dot, -1.0, 1.0)
    deg_angles = np.degrees(np.arccos(dot))
    
    is_dist_ok = sqrD <= dist_threshold_sq
    is_angle_ok = deg_angles <= angle_threshold_deg
    if flip_vertex_normal:
        deg_mirror = 180 - deg_angles
        is_angle_ok = np.logical_or(is_angle_ok, deg_mirror <= angle_threshold_deg)
    
    matched = np.logical_and(is_dist_ok, is_angle_ok)
    return matched, W2


# =============================================================================
# Mesh Laplacian (robust_laplacian 代替 - 単純なグラフラプラシアン)
# =============================================================================

def mesh_adjacency_matrix(V, F):
    """辺から隣接行列を作成（NumPy ベクトル化）"""
    n_verts = V.shape[0]
    if F.shape[0] == 0:
        return sparse.eye(n_verts).tocsr()
    e0 = F[:, [0, 1, 2]].reshape(-1)
    e1 = F[:, [1, 2, 0]].reshape(-1)
    mask = e0 != e1
    e0, e1 = e0[mask], e1[mask]
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_verts, n_verts))
    adj.data = np.minimum(adj.data, 1.0)
    return adj


def mesh_laplacian(V, F):
    """グラフラプラシアン L = D - A (符号: igl と合わせるため -L を使う)"""
    adj = mesh_adjacency_matrix(V, F)
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    L = D - adj
    M = sparse.diags(np.ones(V.shape[0]))
    return L, M


def point_cloud_laplacian(V):
    """点群ラプラシアン（k近傍ベース・NumPy ベクトル化）"""
    if not HAS_SCIPY:
        n = V.shape[0]
        return sparse.eye(n), sparse.eye(n)
    n = V.shape[0]
    k = min(10, n - 1)
    tree = cKDTree(V)
    dists, idx = tree.query(V, k=k + 1)
    dists = np.maximum(dists[:, 1:], 1e-10)
    idx = idx[:, 1:]
    w = 1.0 / dists
    w = w / w.sum(axis=1, keepdims=True)
    rows = np.repeat(np.arange(n), k)
    cols = idx.flatten()
    data = (-w).flatten()
    L = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    L = L + sparse.diags(np.ones(n))
    M = sparse.identity(n)
    return L, M


# =============================================================================
# Inpainting (libigl min_quad_with_fixed 代替)
# =============================================================================

def inpaint(V2, F2, W2, matched, point_cloud_mode=False, progress_cb=None):
    """未マッチ頂点のウェイトを Laplacian で inpaint（元論文・Blender 版と同様）"""
    if not HAS_SCIPY:
        return False, W2.copy()
    prog = progress_cb or (lambda *a: None)
    prog("  Laplacian 構築...")
    if point_cloud_mode:
        L, M = point_cloud_laplacian(V2)
    else:
        L, M = mesh_laplacian(V2, F2)
    
    L = -L
    Mdiag = np.array(M.diagonal()).flatten()
    Mdiag = np.where(Mdiag < 1e-15, 1.0, Mdiag)
    Minv = sparse.diags(1.0 / Mdiag)
    
    b_idx = np.where(matched)[0]
    bc = W2[matched, :].astype(np.float64)
    n_verts = V2.shape[0]
    n_bones = W2.shape[1]
    
    if len(b_idx) == 0:
        return False, W2.copy()
    
    unknown = np.where(~matched)[0]
    n_unknown = len(unknown)
    if n_unknown == 0:
        return True, W2.copy()
    
    prog("  Q_uu 構築（省メモリ）...")
    L_uu = L[unknown, :][:, unknown]
    L_ux = L[unknown, :]
    L_xu = L[:, unknown]
    Q_uu = -L_uu + L_ux @ (Minv @ L_xu)
    Q_uu = Q_uu.tocsc().astype(np.float64)
    
    L_ub = L[unknown, :][:, b_idx]
    L_xb = L[:, b_idx]
    bc_orig = bc.astype(np.float64)
    W_out = W2.copy().astype(np.float64)
    use_iterative = n_unknown > 15000
    solve_uu = None
    if not use_iterative:
        if HAS_PYPARDISO:
            try:
                solve_uu = pypardiso.factorized(Q_uu)
            except Exception:
                pass
        if solve_uu is None:
            try:
                solve_uu = splinalg.factorized(Q_uu.tocsc())
            except Exception:
                use_iterative = True
    prog("  {} ボーン分を解く...".format(n_bones))

    def _solve_one_bone(bone):
        bc_b = np.ascontiguousarray(bc_orig[:, bone], dtype=np.float64)
        tmp = L_xb @ bc_b
        tmp = np.array(Minv @ tmp).flatten()
        rhs_u = np.array(L_ub @ bc_b).flatten() - np.array(L_ux @ tmp).flatten()
        try:
            if solve_uu is not None:
                w_u = solve_uu(rhs_u)
            elif HAS_PYPARDISO:
                w_u = pypardiso.spsolve(Q_uu, rhs_u)
            elif use_iterative:
                w_u, info = splinalg.cg(Q_uu, rhs_u, tol=1e-8, maxiter=min(2000, n_unknown))
                if info != 0:
                    w_u = splinalg.spsolve(Q_uu, rhs_u)
            else:
                w_u = splinalg.spsolve(Q_uu, rhs_u)
            return bone, np.clip(w_u, 0, 1)
        except Exception:
            return bone, np.zeros(n_unknown, dtype=np.float64)

    if use_iterative and n_bones >= 4:
        n_workers = min(8, n_bones)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for bone, w_u in ex.map(_solve_one_bone, range(n_bones)):
                if n_bones > 8 and bone % max(1, n_bones // 8) == 0:
                    prog("  Inpaint...", (bone + 1) / n_bones)
                W_out[unknown, bone] = w_u
    else:
        for bone in range(n_bones):
            if n_bones > 8 and bone % max(1, n_bones // 8) == 0:
                prog("  Inpaint...", (bone + 1) / n_bones)
            bone, w_u = _solve_one_bone(bone)
            W_out[unknown, bone] = w_u
    W_out[matched, :] = bc_orig
    W_out = np.clip(W_out, 0, 1)
    row_sum = W_out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-10, 1, row_sum)
    W_out = W_out / row_sum
    
    return True, W_out.astype(np.float32)


def _adjacency_list_from_matrix(adj):
    """隣接行列から隣接リストを生成（smooth_weights 用）"""
    adj = adj.tocsr()
    n = adj.shape[0]
    result = [[] for _ in range(n)]
    for i in range(n):
        for j in adj.indices[adj.indptr[i]:adj.indptr[i+1]]:
            if i != j:
                result[i].append(int(j))
    return result


def smooth_weights(verts, weights, matched, adjacency_matrix, adjacency_list,
                  num_smooth_iter_steps=4, smooth_alpha=0.2, distance_threshold=0.05):
    """
    元 Blender 版と同等: inpaint 境界付近のウェイトをスムージング
    マッチ済み頂点は固定し、未マッチ周辺のみ distance_threshold 内で平滑化
    """
    if not HAS_SCIPY:
        return weights.copy()
    not_matched = ~matched
    VIDs_to_smooth = np.zeros(verts.shape[0], dtype=bool)

    def get_points_within_distance(V, vid):
        queue = [vid]
        while queue:
            vv = queue.pop()
            if vv < len(adjacency_list):
                for nn in adjacency_list[vv]:
                    if not VIDs_to_smooth[nn] and np.linalg.norm(V[vid] - V[nn]) < distance_threshold:
                        VIDs_to_smooth[nn] = True
                        queue.append(nn)

    for i in range(verts.shape[0]):
        if not_matched[i]:
            get_points_within_distance(verts, i)

    adj_mat = adjacency_matrix.astype(np.float32)
    degrees = np.array(adj_mat.sum(axis=1)).flatten()
    degrees = np.where(degrees < 1e-10, 1, degrees)
    smooth_mat = sparse.diags(1.0 / degrees) @ adj_mat
    w = weights.astype(np.float32).copy()
    for _ in range(num_smooth_iter_steps):
        w_smooth = smooth_mat @ w
        w = (1 - smooth_alpha) * w + smooth_alpha * w_smooth
        w[~VIDs_to_smooth] = weights[~VIDs_to_smooth]
    out = w
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-10, 1, row_sum)
    return (out / row_sum).astype(np.float32)


def smooth_weights_global(weights, adjacency_matrix, num_smooth_iter_steps=4, smooth_alpha=0.2):
    """
    全頂点を対象にウェイトをスムージング（境界線をぼかす用）。
    スムージング回数・強度を上げるとボーン境界がなめらかになる。
    """
    if not HAS_SCIPY or adjacency_matrix is None:
        return weights.copy()
    adj_mat = adjacency_matrix.astype(np.float64)
    degrees = np.array(adj_mat.sum(axis=1)).flatten()
    degrees = np.where(degrees < 1e-10, 1, degrees)
    smooth_mat = sparse.diags(1.0 / degrees) @ adj_mat
    w = weights.astype(np.float64).copy()
    for _ in range(num_smooth_iter_steps):
        w = (1 - smooth_alpha) * w + smooth_alpha * (smooth_mat @ w)
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-10, 1, row_sum)
    return (w / row_sum).astype(np.float32)


def limit_mask(weights, adj_matrix, dilation_repeat=5, limit_num=4):
    """頂点あたりのボーン数を制限"""
    if not HAS_SCIPY or weights.shape[1] <= limit_num:
        return np.zeros_like(weights)
    
    count = np.count_nonzero(weights, axis=1)
    to_limit = count > limit_num
    k = weights.shape[1] - limit_num
    weights_inds = np.argpartition(weights, kth=k, axis=1)[:, :k]
    row_indices = np.arange(weights.shape[0])[:, None]
    erode_mask = np.zeros_like(weights, dtype=bool)
    erode_mask[row_indices, weights_inds] = True
    erode_mask = np.logical_and(erode_mask, to_limit[:, np.newaxis])
    erode_mask = sparse.csr_matrix(erode_mask.astype(np.float32))
    
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees = np.where(degrees < 1e-10, 1, degrees)
    smooth_mat = sparse.diags(1.0 / degrees) @ adj_matrix
    
    for _ in range(dilation_repeat):
        avg_weights = smooth_mat @ erode_mask
        erode_mask = erode_mask.maximum(avg_weights)
    
    return erode_mask.toarray()


# =============================================================================
# Houdini Geometry 入出力（一括APIで高速化）
# =============================================================================

def geo_to_arrays(geo, need_vertex_indices=False):
    """hou.Geometry から頂点・面・法線を取得（pointFloatAttribValues で一括読み込み）
    need_vertex_indices: True で F_vtx（各三角形の頂点リニアインデックス）も返す
    """
    try:
        geo = geo.freeze()
    except Exception:
        pass
    try:
        n_pts = geo.pointCount()
        n_prims = geo.primCount()
    except (AttributeError, hou.OperationFailed):
        n_pts = len(geo.points())
        n_prims = len(geo.prims())
    if n_pts == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.ones((0, 3)) * 0.001
    try:
        V = np.frombuffer(geo.pointFloatAttribValuesAsString("P"), dtype=np.float32).reshape(-1, 3).astype(np.float64)
    except Exception:
        V = np.array(geo.pointFloatAttribValues("P"), dtype=np.float64).reshape(-1, 3)
    F = []
    F_vtx = [] if need_vertex_indices else None
    vert_offset = 0
    if n_prims > 0:
        prs = list(geo.prims())
        for pi in range(n_prims):
            pr = prs[pi]
            vs = pr.vertices()
            nv = len(vs)
            if nv >= 3:
                p0 = vs[0].point().number()
                for i in range(1, nv - 1):
                    F.append([p0, vs[i].point().number(), vs[i + 1].point().number()])
                    if need_vertex_indices:
                        F_vtx.append([vert_offset + 0, vert_offset + i, vert_offset + i + 1])
            vert_offset += nv
    F = np.array(F, dtype=np.int32) if F else np.zeros((0, 3), dtype=np.int32)
    if need_vertex_indices and F_vtx:
        F_vtx = np.array(F_vtx, dtype=np.int32)
    elif need_vertex_indices:
        F_vtx = np.zeros((0, 3), dtype=np.int32)
    N = np.zeros((n_pts, 3), dtype=np.float32)
    if geo.findPointAttrib("N"):
        try:
            N = np.frombuffer(geo.pointFloatAttribValuesAsString("N"), dtype=np.float32).reshape(-1, 3)[:n_pts]
        except Exception:
            N = np.array(geo.pointFloatAttribValues("N"), dtype=np.float32).reshape(-1, 3)[:n_pts]
    else:
        N = _compute_vertex_normals(V, F)
        N = np.where(np.linalg.norm(N, axis=1, keepdims=True) < 1e-6, N + 0.001, N)
    nlen = np.linalg.norm(N, axis=1, keepdims=True)
    nlen = np.where(nlen < 1e-10, 1.0, nlen)
    N = (N / nlen).astype(np.float32)
    if need_vertex_indices:
        return V, F, N, F_vtx
    return V, F, N


def _compute_vertex_normals(V, F):
    """面法線から頂点法線を計算（np.add.at でベクトル化）"""
    N = np.zeros_like(V)
    if F.shape[0] == 0:
        return N + 0.001
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn /= np.where(ln < 1e-12, 1.0, ln)
    np.add.at(N, F[:, 0], fn)
    np.add.at(N, F[:, 1], fn)
    np.add.at(N, F[:, 2], fn)
    return N


def get_capture_weights_dense(geo, prefix="boneCapture"):
    """
    Unpack された capture 属性から密なウェイト配列を取得（一括API優先）
    返値: (W, bone_names, slot_size, capture_class)
    capture_class: "point" または "vertex"（Vertex のとき F_vtx が必要）
    """
    idx_name = prefix + "_index"
    data_name = prefix + "_data"
    idx_attrib = geo.findPointAttrib(idx_name)
    data_attrib = geo.findPointAttrib(data_name)
    is_vertex = False
    if idx_attrib is None or data_attrib is None:
        idx_attrib = geo.findVertexAttrib(idx_name) if hasattr(geo, 'findVertexAttrib') else None
        data_attrib = geo.findVertexAttrib(data_name) if hasattr(geo, 'findVertexAttrib') else None
        if idx_attrib is not None and data_attrib is not None:
            is_vertex = True
    if idx_attrib is None or data_attrib is None:
        return None, [], 4, "point"
    
    try:
        n_pts = geo.pointCount()
        n_verts = len(list(geo.iterVertices())) if hasattr(geo, 'iterVertices') else n_pts
    except (AttributeError, hou.OperationFailed):
        n_pts = len(geo.points())
        n_verts = n_pts
    if is_vertex:
        n_elements = n_verts
    else:
        n_elements = n_pts
    
    bone_names = []
    if hasattr(geo, 'globalAttribs'):
        for a in geo.globalAttribs():
            n = a.name()
            if prefix in n and ("pCaptPath" in n or "path" in n.lower()):
                try:
                    bone_names.append(str(a.strings()[0]) if hasattr(a, 'strings') else str(n))
                except Exception:
                    pass
    
    W = None
    try:
        if is_vertex:
            idx_flat = np.array(geo.vertexIntAttribValues(idx_name), dtype=np.int32)
            data_flat = np.array(geo.vertexFloatAttribValues(data_name), dtype=np.float32)
        else:
            idx_flat = np.array(geo.pointIntAttribValues(idx_name), dtype=np.int32)
            data_flat = np.array(geo.pointFloatAttribValues(data_name), dtype=np.float32)
        arr_size = len(idx_flat) // max(n_elements, 1)
        if arr_size < 1:
            arr_size = 4
        idx_flat = idx_flat.reshape(n_elements, arr_size)
        data_flat = data_flat.reshape(n_elements, arr_size)
        max_idx = int(np.max(idx_flat[idx_flat >= 0])) if np.any(idx_flat >= 0) else 0
        n_bones = max_idx + 1
        if n_bones <= 0:
            n_bones = 1
        W = np.zeros((n_elements, n_bones), dtype=np.float32)
        for j in range(idx_flat.shape[1]):
            bi = idx_flat[:, j]
            w = data_flat[:, j]
            valid = (bi >= 0) & (bi < n_bones) & (w > 1e-8)
            W[valid, bi[valid]] = w[valid]
        slot_size = arr_size
    except Exception:
        slot_size = 4
    
    if W is None:
        max_idx = -1
        elem_iter = list(geo.iterVertices()) if is_vertex else list(geo.iterPoints())
        n_iter = len(elem_iter)
        for i in range(n_iter):
            try:
                idx = np.atleast_1d(elem_iter[i].attribValue(idx_name))
                if len(idx) and np.any(idx >= 0):
                    max_idx = max(max_idx, int(np.max(idx)))
            except Exception:
                pass
        n_bones = max(max_idx + 1, 1)
        W = np.zeros((n_iter, n_bones), dtype=np.float32)
        for i in range(n_iter):
            try:
                idx = np.atleast_1d(elem_iter[i].attribValue(idx_name))
                data = np.atleast_1d(elem_iter[i].attribValue(data_name))
                for j, (bi, w) in enumerate(zip(idx[: len(data)], data)):
                    bi = int(bi)
                    if bi >= 0 and bi < n_bones and w > 1e-8:
                        W[i, bi] = float(w)
            except Exception:
                pass
        slot_size = 4
    capture_class = "vertex" if is_vertex else "point"
    return W, bone_names, slot_size, capture_class


def set_capture_weights_from_dense(geo, W, prefix="boneCapture", bone_names=None, progress_cb=None, slot_size=4, capture_class="point"):
    """
    密なウェイト配列を Unpack 形式の capture 属性に書き戻し（完全ベクトル化）。
    capture_class: "point" または "vertex"（Vertex 時は各頂点が参照するポイントのウェイトを複写）
    """
    prog = progress_cb or (lambda *a: None)
    n_pts, n_bones = W.shape
    idx_name = prefix + "_index"
    data_name = prefix + "_data"
    is_vertex = (capture_class == "vertex")
    if is_vertex:
        verts = list(geo.iterVertices())
        n_elems = len(verts)
        pt_of_v = np.array([v.point().number() for v in verts], dtype=np.int32)
        W_exp = W[pt_of_v, :]
    else:
        n_elems = n_pts
        W_exp = W
    W_mask = np.where(W_exp > 1e-6, W_exp, 0.0).astype(np.float32)
    k = min(slot_size, n_bones)
    if n_bones > k:
        kth = n_bones - k
        part = np.argpartition(W_mask, kth=kth, axis=1)[:, kth:]
    else:
        part = np.tile(np.arange(n_bones), (n_elems, 1))
    topk_vals = np.take_along_axis(W_mask, part, axis=1)
    order = np.argsort(topk_vals, axis=1)[:, ::-1]
    topk_idx = np.take_along_axis(part, order, axis=1)
    topk_vals = np.take_along_axis(topk_vals, order, axis=1)
    s = topk_vals.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)
    topk_vals = (topk_vals / s).astype(np.float32)
    zero_mask = topk_vals < 1e-6
    topk_idx = topk_idx.astype(np.int32)
    topk_idx[zero_mask] = -1
    topk_vals[zero_mask] = 0.0
    if k < slot_size:
        pad_i = np.full((n_elems, slot_size - k), -1, dtype=np.int32)
        pad_v = np.zeros((n_elems, slot_size - k), dtype=np.float32)
        topk_idx = np.hstack([topk_idx, pad_i])
        topk_vals = np.hstack([topk_vals, pad_v])
    idx_flat = np.ascontiguousarray(topk_idx.flatten())
    data_flat = np.ascontiguousarray(topk_vals.flatten())
    n_geo_elems = len(list(geo.iterVertices())) if is_vertex else len(geo.points())
    if n_geo_elems != n_elems:
        raise hou.NodeError("要素数不一致: geo={} vs W={}".format(n_geo_elems, n_elems))
    try:
        if is_vertex:
            geo.setVertexIntAttribValuesFromString(idx_name, idx_flat)
            geo.setVertexFloatAttribValuesFromString(data_name, data_flat)
        else:
            geo.setPointIntAttribValuesFromString(idx_name, idx_flat)
            geo.setPointFloatAttribValuesFromString(data_name, data_flat)
        prog("  ウェイト書き込み完了（ベクトル化）")
    except (hou.OperationFailed, TypeError):
        try:
            if is_vertex:
                geo.setVertexIntAttribValues(idx_name, idx_flat.tolist())
                geo.setVertexFloatAttribValues(data_name, data_flat.tolist())
            else:
                geo.setPointIntAttribValues(idx_name, idx_flat.tolist())
                geo.setPointFloatAttribValues(data_name, data_flat.tolist())
            prog("  ウェイト書き込み完了（ベクトル化）")
        except Exception:
            elem_iter = list(geo.iterVertices()) if is_vertex else list(geo.points())
            for i in range(n_elems):
                idx_tup = tuple(int(x) for x in idx_flat[i * slot_size : (i + 1) * slot_size])
                data_tup = tuple(float(x) for x in data_flat[i * slot_size : (i + 1) * slot_size])
                elem_iter[i].setAttribValue(idx_name, idx_tup)
                elem_iter[i].setAttribValue(data_name, data_tup)


def _extract_bone_name(path):
    """パスからボーン名を取得（例: /obj/char/rig/Hips -> Hips）"""
    s = str(path).strip().rstrip("/")
    if not s:
        return ""
    return s.split("/")[-1] if "/" in s else s


def remap_weights_to_skeleton(W_old, old_paths, new_paths, progress_cb=None):
    """
    ウェイトを古いスケルトン（old_paths）から新しいスケルトン（new_paths）へボーン名でマッピングして転写。
    戻り値: W_new (n_points, n_new_bones)
    """
    prog = progress_cb or (lambda *a: None)
    n_pts = W_old.shape[0]
    n_new = len(new_paths)
    if n_new == 0:
        prog("  [ボーンマップ] 新しい pCaptPath が 0 件のためスキップ")
        return W_old
    new_names = [_extract_bone_name(p) for p in new_paths]
    old_to_new = np.full(W_old.shape[1], -1, dtype=np.int32)
    matched = 0
    for old_idx in range(W_old.shape[1]):
        if old_idx >= len(old_paths):
            break
        old_name = _extract_bone_name(old_paths[old_idx])
        for new_idx, nn in enumerate(new_names):
            if nn == old_name:
                old_to_new[old_idx] = new_idx
                matched += 1
                break
    prog("  ボーンマップ: {} / {} 一致（新しいスケルトン {} ボーン）".format(matched, W_old.shape[1], n_new))
    W_new = np.zeros((n_pts, n_new), dtype=np.float32)
    for old_idx in range(W_old.shape[1]):
        new_idx = old_to_new[old_idx]
        if new_idx >= 0:
            W_new[:, new_idx] += W_old[:, old_idx]
    row_sum = W_new.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-10, 1, row_sum)
    return (W_new / row_sum).astype(np.float32)


def _get_capture_metadata(geo, prefix="boneCapture"):
    """pCaptPath, pCaptSkelRoot 等の detail 属性を取得（検証用）"""
    out = {"paths": [], "skel": None}
    if not hasattr(geo, 'globalAttribs'):
        return out
    for a in geo.globalAttribs():
        n = a.name()
        try:
            v = geo.attribValue(a.name())
            if "pCaptPath" in n or (n.endswith("_pCaptPath") or n == "pCaptPath"):
                out["paths"] = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
            elif "pCaptSkelRoot" in n or "SkelRoot" in n:
                out["skel"] = v[0] if isinstance(v, (list, tuple)) and v else v
        except Exception:
            pass
    return out


def copy_capture_metadata(src_geo, out_geo, prefix="boneCapture", progress_cb=None):
    """Capture Attribute Pack 用の detail メタデータ（pCaptPath, pCaptData）をソースから出力へコピー"""
    prog = progress_cb or (lambda *a: None)
    if not hasattr(src_geo, 'globalAttribs'):
        return 0
    n_copied = 0
    try:
        attrs = src_geo.globalAttribs()
    except Exception:
        attrs = ()
    for attr in attrs:
        aname = attr.name()
        is_capture = prefix in aname or "pCapt" in aname or "CaptPath" in aname or "CaptData" in aname or "SkelRoot" in aname
        if not is_capture:
            continue
        try:
            val = src_geo.attribValue(aname)
            if out_geo.findGlobalAttrib(aname) is None:
                size = max(len(val) if isinstance(val, (list, tuple)) else 1, 1)
                if "Path" in aname or "path" in aname.lower() or (isinstance(val, (list, tuple)) and val and isinstance(val[0], str)):
                    out_geo.addArrayAttrib(hou.attribType.Global, aname, hou.attribData.String, size)
                else:
                    out_geo.addArrayAttrib(hou.attribType.Global, aname, hou.attribData.Float, size)
            out_geo.setGlobalAttribValue(aname, val)
            n_copied += 1
        except Exception as e:
            prog("  [警告] {} のコピー失敗: {}".format(aname, str(e)[:50]))
    return n_copied


def ensure_capture_attribs(geo, prefix, n_bones, num_slots=4, capture_class="point"):
    """capture 用の _index, _data アトリビュートを追加（既存ならスキップ）"""
    idx_name = prefix + "_index"
    data_name = prefix + "_data"
    atype = hou.attribType.Vertex if capture_class == "vertex" else hou.attribType.Point
    find_attr = geo.findVertexAttrib if capture_class == "vertex" else geo.findPointAttrib
    if find_attr(idx_name) is None:
        geo.addArrayAttrib(atype, idx_name, hou.attribData.Int, num_slots)
    if find_attr(data_name) is None:
        geo.addArrayAttrib(atype, data_name, hou.attribData.Float, num_slots)


# =============================================================================
# メイン処理（Python SOP から呼び出し）
# =============================================================================

def run_robust_weight_transfer(
    node,
    max_distance=0.05,
    max_angle_deg=30.0,
    flip_normal=True,
    inpaint_point_cloud=True,
    limit_bones=4,
    smooth_enable=False,
    debug_dominant_bone_color=False,
    smooth_repeat=4,
    smooth_alpha=0.2,
    smooth_global_repeat=4,
    smooth_global_alpha=0.2,
    smooth_distance_scale=1.5,
    blur_pass=3,
    blur_alpha=0.5,
):
    """
    Python SOP のメインロジック
    Input 0: Source (body) - Capture Attribute Unpack 済み
    Input 1: Target (clothing) - 転送先
    """
    prog = lambda msg, pct=None: _progress(msg, pct)
    total_start = time.perf_counter()
    prog("  爆速化: pykdtree={}, pypardiso={} (利用可能だが未使用)".format(
        "利用可" if _PYKDTREE_AVAILABLE else "-", "利用可" if _PYPARDISO_AVAILABLE else "-"))
    
    inputs = node.inputs()
    if len(inputs) < 2:
        return False, "Input 0 (source) と Input 1 (target) が必要です"
    
    # 全入力を先に参照して Cook 順を確定（おいて行かれ防止）
    for i in range(min(3, len(inputs))):
        if inputs[i] is not None:
            try:
                _ = inputs[i].geometry()
            except Exception:
                pass
    
    t0 = time.perf_counter()
    prog("[1/6] ジオメトリ読み込み...")
    try:
        src_geo = inputs[0].geometry().freeze()
        tgt_geo = inputs[1].geometry().freeze()
    except Exception:
        src_geo = inputs[0].geometry()
        tgt_geo = inputs[1].geometry()
    out_geo = node.geometry()
    
    ref_geo = None
    if len(inputs) >= 3 and inputs[2] is not None:
        try:
            ref_geo = inputs[2].geometry()
        except Exception:
            pass
    
    if src_geo is None or tgt_geo is None:
        return False, "ソースまたはターゲットのジオメトリがありません"
    
    src_V, src_F, src_N = geo_to_arrays(src_geo)
    tgt_V, tgt_F, tgt_N = geo_to_arrays(tgt_geo)
    src_F_vtx = None
    prog("  ソース: {} 頂点, {} 面 | ターゲット: {} 頂点, {} 面".format(
        len(src_V), len(src_F), len(tgt_V), len(tgt_F)))
    
    if src_F.shape[0] == 0:
        return False, "ソースメッシュにポリゴンがありません"
    if len(tgt_V) == 0:
        return False, "ターゲット（衣装）に頂点がありません。Input 1 のジオメトリを確認してください"
    
    if ref_geo is not None:
        prog("  Input 2 接続: 新しいスケルトンへボーンマップを実行します")
    prog("  [1/6] 所要時間: {:.2f} sec".format(time.perf_counter() - t0))

    t0 = time.perf_counter()
    prog("[2/6] Capture ウェイト読み込み...")
    src_weights, bone_names, slot_size, capture_class = get_capture_weights_dense(src_geo)
    if src_weights is None or src_weights.shape[1] == 0:
        return False, "ソースに boneCapture がありません。Capture Attribute Unpack を適用してください"
    prog("  {} ボーン, {} 要素 ({})".format(src_weights.shape[1], src_weights.shape[0], capture_class))
    
    if capture_class == "vertex":
        _, _, _, src_F_vtx = geo_to_arrays(src_geo, need_vertex_indices=True)
    prog("  [2/6] 所要時間: {:.2f} sec".format(time.perf_counter() - t0))

    t0 = time.perf_counter()
    prog("[3/6] 最近接点マッチング...")
    dist_sq = max_distance ** 2
    matched, W2 = find_matches_closest_surface(
        src_V, src_F, src_N,
        tgt_V, tgt_N, src_weights,
        dist_sq, max_angle_deg, flip_normal,
        progress_cb=prog,
        source_triangles_vtx=src_F_vtx,
    )
    n_matched = np.sum(matched)
    n_total = len(matched)
    prog("  マッチ: {}/{} 頂点 ({:.1f}%)".format(n_matched, n_total, 100 * n_matched / max(1, n_total)))
    prog("  [3/6] 所要時間: {:.2f} sec".format(time.perf_counter() - t0))

    t0 = time.perf_counter()
    prog("[4/6] Inpaint（未マッチ頂点補間）...")
    if n_matched < n_total:
        if HAS_SCIPY:
            ok, W2 = inpaint(tgt_V, tgt_F, W2, matched,
                             point_cloud_mode=inpaint_point_cloud,
                             progress_cb=prog)
        else:
            ok, W2 = False, W2
        if ok:
            prog("  Inpaint 完了")
            if smooth_enable and HAS_SCIPY:
                prog("  Smoothing（未マッチ周辺）...")
                adj = mesh_adjacency_matrix(tgt_V, tgt_F)
                adj = adj + sparse.eye(tgt_V.shape[0])
                adj_list = _adjacency_list_from_matrix(adj)
                W2 = smooth_weights(tgt_V, W2, matched, adj, adj_list,
                                    num_smooth_iter_steps=smooth_repeat,
                                    smooth_alpha=smooth_alpha,
                                    distance_threshold=max_distance * smooth_distance_scale)
                prog("  Smoothing（マッチ頂点含む・全体）...")
                W2 = smooth_weights_global(W2, adj,
                                           num_smooth_iter_steps=smooth_global_repeat,
                                           smooth_alpha=smooth_global_alpha)
        else:
            prog("  Inpaint スキップ（scipy 未検出または失敗）")
    else:
        prog("  全頂点マッチのためスキップ")
    prog("  [4/6] 所要時間: {:.2f} sec".format(time.perf_counter() - t0))

    t0 = time.perf_counter()
    prog("[5/6] ボーン数制限 (max={})...".format(limit_bones))
    if limit_bones > 0 and HAS_SCIPY and W2.shape[1] > limit_bones:
        adj = mesh_adjacency_matrix(tgt_V, tgt_F)
        adj = adj + sparse.eye(tgt_V.shape[0])
        mask = limit_mask(W2, adj, dilation_repeat=5, limit_num=limit_bones)
        W2 = (1 - mask) * W2
        W2[W2 < 1e-5] = 0
        row_sum = W2.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum < 1e-10, 1, row_sum)
        W2 = W2 / row_sum
        prog("  制限完了")
    else:
        prog("  スキップ")

    if ref_geo is not None:
        src_meta = _get_capture_metadata(src_geo, "boneCapture")
        ref_meta = _get_capture_metadata(ref_geo, "boneCapture")
        old_paths = src_meta["paths"]
        new_paths = ref_meta["paths"]
        if old_paths and new_paths:
            prog("[5.5/6] ボーンマップ（新しいスケルトンへ）...")
            W2 = remap_weights_to_skeleton(W2, old_paths, new_paths, progress_cb=prog)
    
    # 境界ぼかしをウェイトに適用（変形と表示が一致、スライダーで調整可）
    if HAS_SCIPY and tgt_F.shape[0] > 0 and blur_pass > 0 and blur_alpha > 0:
        adj = mesh_adjacency_matrix(tgt_V, tgt_F) + sparse.eye(tgt_V.shape[0])
        W2 = smooth_weights_global(W2, adj, num_smooth_iter_steps=blur_pass, smooth_alpha=blur_alpha)
        prog("  境界ぼかしをウェイトに適用（{}回・{:.2f}）".format(blur_pass, blur_alpha))
    prog("  [5/6] 所要時間: {:.2f} sec".format(time.perf_counter() - t0))

    t0 = time.perf_counter()
    prog("[6/6] 出力ジオメトリ書き込み...")
    out_geo.copy(tgt_geo)
    
    if debug_dominant_bone_color and W2.shape[1] > 0:
        dom = np.argmax(W2, axis=1)
        max_w = np.max(W2, axis=1)  # 優勢ウェイトの強さ（1=単一ボーン、低い=ブレンド域）
        h = ((dom * 137.5) % 360) / 360.0
        import colorsys
        # ブレンド域は明度を下げて薄く表示（なめらかなところが薄くなる）
        rgb = np.array([
            colorsys.hsv_to_rgb(float(h[i]), 0.9, 0.2 + 0.8 * float(max_w[i]))
            for i in range(len(h))
        ], dtype=np.float32)
        # 表示用に同じ境界ぼかし（ウェイトと同じ回数・強度で変形と見た目を一致）
        if HAS_SCIPY and tgt_F.shape[0] > 0 and rgb.shape[0] == tgt_V.shape[0] and blur_pass > 0 and blur_alpha > 0:
            adj = mesh_adjacency_matrix(tgt_V, tgt_F)
            degrees = np.array(adj.sum(axis=1)).flatten()
            degrees = np.where(degrees < 1e-10, 1, degrees)
            smooth_mat = sparse.diags(1.0 / degrees) @ adj
            for _ in range(blur_pass):
                blended = smooth_mat @ np.asarray(rgb, dtype=np.float64)
                rgb = np.asarray((1 - blur_alpha) * rgb + blur_alpha * blended, dtype=np.float32)
        n_out_pts = out_geo.pointCount()
        rgb = np.ascontiguousarray(rgb[:n_out_pts, :3].astype(np.float32))
        if rgb.shape[0] != n_out_pts:
            rgb = np.vstack([rgb, np.zeros((n_out_pts - rgb.shape[0], 3), dtype=np.float32)])
        if out_geo.findPointAttrib("Cd") is None:
            out_geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
        out_geo.setPointFloatAttribValuesFromString("Cd", rgb.reshape(-1))
        prog("  デバッグ: 優勢ボーンを Cd で表示（境界ぼかし＝ウェイトと同じ）")
    
    meta_src = ref_geo if ref_geo is not None else src_geo
    n_meta = copy_capture_metadata(meta_src, out_geo, "boneCapture", progress_cb=prog)
    if n_meta == 0 and hasattr(meta_src, 'globalAttribs'):
        all_global = [a.name() for a in meta_src.globalAttribs()]
        prog("  [重要] 体に capture 系 detail が 0 件。Capture Attribute Unpack の「Unpack Properties」を ON にしてください")
        prog("  体入力の globalAttribs: {}".format(", ".join(all_global[:10]) + ("..." if len(all_global) > 10 else "") if all_global else "(なし)"))
    
    out_meta = _get_capture_metadata(out_geo, "boneCapture")
    n_out_p = len(out_meta["paths"])
    if n_out_p == 0:
        prog("  [深刻] pCaptPath: 0 件。Unpack の「Unpack Properties」を ON にしてください")
    else:
        prog("  pCaptPath: {} 件".format(n_out_p))
    
    ensure_capture_attribs(out_geo, "boneCapture", W2.shape[1], num_slots=slot_size, capture_class=capture_class)
    set_capture_weights_from_dense(out_geo, W2, "boneCapture", bone_names, progress_cb=prog, slot_size=slot_size, capture_class=capture_class)
    
    has_idx = out_geo.findPointAttrib("boneCapture_index") is not None
    has_data = out_geo.findPointAttrib("boneCapture_data") is not None
    if not has_idx or not has_data:
        return False, "Capture 属性の書き込みに失敗しました。Capture Attribute Pack の Class を Point に設定し、入力接続を確認してください。"
    
    prog("  出力: boneCapture_index={}, boneCapture_data={}, pts={}".format(has_idx, has_data, len(out_geo.points())))
    try:
        out_geo.incrementAllDataIds()
    except Exception:
        pass
    prog("  [6/6] 所要時間: {:.2f} sec".format(time.perf_counter() - t0))
    prog("  合計所要時間: {:.2f} sec".format(time.perf_counter() - total_start))
    
    return True, "完了: {} 頂点マッチ、{} ボーン (slot_size={})".format(n_matched, W2.shape[1], slot_size)


# =============================================================================
# Python SOP エントリポイント（__rwt_gui_mode__ 時は setup スクリプトが呼ぶ）
# =============================================================================

def _eval_parm(node, name, default):
    """ノードにパラメータがあればその値を、なければ default を返す（スライダー連動用）"""
    p = node.parm(name)
    if p is not None:
        try:
            return p.eval()
        except Exception:
            pass
    return default


if not globals().get("__rwt_gui_mode__"):
    node = hou.pwd()
    output_geo = node.geometry()
    inputs = node.inputs()

    if len(inputs) >= 2:
        # Input 0: 体（Unpack済み）, Input 1: 衣装, Input 2: 任意・新しいスケルトンの参照（Unpack済みジオメトリ）
        # スライダー（rwt_folder）があればパラメータから取得、なければデフォルト
        ok, msg = run_robust_weight_transfer(
            node,
            max_distance=_eval_parm(node, "max_distance", 0.25),
            max_angle_deg=_eval_parm(node, "max_angle_deg", 60.0),
            flip_normal=_eval_parm(node, "flip_normal", True),
            inpaint_point_cloud=_eval_parm(node, "inpaint_point_cloud", False),
            limit_bones=_eval_parm(node, "limit_bones", 0),
            smooth_enable=_eval_parm(node, "smooth_enable", True),
            smooth_repeat=_eval_parm(node, "smooth_repeat", 4),
            smooth_alpha=_eval_parm(node, "smooth_alpha", 0.18),
            smooth_global_repeat=_eval_parm(node, "smooth_global_repeat", 4),
            smooth_global_alpha=_eval_parm(node, "smooth_global_alpha", 0.2),
            smooth_distance_scale=_eval_parm(node, "smooth_distance_scale", 1.2),
            debug_dominant_bone_color=_eval_parm(node, "debug_dominant_bone_color", True),
            blur_pass=_eval_parm(node, "blur_pass", 3),
            blur_alpha=_eval_parm(node, "blur_alpha", 0.5),
        )
        if not ok:
            raise hou.NodeError(msg)
        print(msg)
    else:
        raise hou.NodeError("Input 0 (source body) と Input 1 (target clothing) を接続してください。Input 2 は任意（ボーン追加・削除した新しいスケルトンの参照）")
