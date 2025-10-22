\
"""
Verificador de correspondências de pontos (A vs B).

Checa:
1) Mesma contagem de pontos (N_A == N_B)
2) Pontos finitos e dentro do domínio (opcional, se passadas as imagens)
3) Duplicatas próximas (mesmo CSV)
4) Homografia/afim global (LS) e erro RMS de reprojeção
5) Similaridade (Procrustes) e erro RMS (escala/rotação/translação)
6) Triângulos com orientação invertida entre A e B (com base na Delaunay de pontos médios)

Uso:
    python tools/check_points.py --ptsA data/pontos_A.csv --ptsB data/pontos_B.csv [--imgA data/A.png --imgB data/B.png] [--report report.txt]
"""
import argparse
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

def read_points_csv(path):
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            a = s.split(",")
            if len(a) >= 2:
                xs.append(float(a[0])); ys.append(float(a[1]))
    if not xs:
        return np.zeros((0,2), dtype=np.float32)
    return np.stack([xs,ys], axis=1).astype(np.float32)

def rms(x):
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0

def fit_affine(A, B):
    """A (N,2) -> B (N,2), resolve 6 params por LS. Retorna M 2x3 e erros por ponto."""
    N = A.shape[0]
    X = np.zeros((2*N, 6), dtype=np.float64)
    y = np.zeros((2*N,), dtype=np.float64)
    X[0::2, 0:3] = np.c_[A, np.ones(N)]           # x terms
    X[1::2, 3:6] = np.c_[A, np.ones(N)]           # y terms
    y[0::2] = B[:,0]
    y[1::2] = B[:,1]
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    M = np.array([[sol[0], sol[1], sol[2]],
                  [sol[3], sol[4], sol[5]]], dtype=np.float64)
    # reprojeção
    Ah = np.c_[A, np.ones(N)]
    B_hat = (M @ Ah.T).T
    err = np.linalg.norm(B_hat - B, axis=1)
    return M, err

def procrustes_similarity(A, B):
    """Ajusta similaridade B ~ s*R*A + t. Retorna s, R (2x2), t (2,), err por ponto."""
    A = A.astype(np.float64); B = B.astype(np.float64)
    muA = A.mean(axis=0); muB = B.mean(axis=0)
    A0 = A - muA; B0 = B - muB
    normA = np.sqrt((A0**2).sum()); normB = np.sqrt((B0**2).sum())
    if normA < 1e-12 or normB < 1e-12:
        s = 1.0; R = np.eye(2); t = muB - muA
        err = np.linalg.norm((s*(A@R.T) + t) - B, axis=1)
        return s, R, t, err
    A0 /= normA; B0 /= normB
    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    s = (B0 * (A0 @ R.T)).sum() / (A0*A0).sum()
    # Reescala para o espaço original
    s = s * (normB / normA)
    t = muB - (s*(muA @ R.T))
    B_hat = (s*(A @ R.T)) + t
    err = np.linalg.norm(B_hat - B, axis=1)
    return s, R, t, err

def has_duplicates(P, tol=1e-6):
    """Retorna índices de possíveis duplicatas (distância < tol)."""
    if P.shape[0] < 2: return []
    idxs = []
    for i in range(P.shape[0]):
        d2 = ((P - P[i])**2).sum(axis=1)
        d2[i] = 1e9
        j = int(np.argmin(d2))
        if d2[j] < tol**2:
            idxs.append(tuple(sorted((i,j))))
    # únicos
    idxs = sorted(set(idxs))
    return idxs

def tri_orientation(p, tri):
    """Sinal da área (orientação) de um triângulo (índices i,j,k)."""
    i,j,k = tri
    x1,y1 = p[i]; x2,y2 = p[j]; x3,y3 = p[k]
    return np.sign((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))

def check_orientations(pA, pB):
    """Com base na Delaunay dos pontos médios, checa triângulos com orientação diferente em A e B."""
    pM = 0.5*pA + 0.5*pB
    simp = Delaunay(pM).simplices
    flips = []
    for tri in simp:
        sA = tri_orientation(pA, tri)
        sB = tri_orientation(pB, tri)
        if sA == 0 or sB == 0:
            continue
        if sA != sB:
            flips.append(tuple(tri))
    return simp, flips

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptsA", required=True)
    ap.add_argument("--ptsB", required=True)
    ap.add_argument("--imgA", default=None)
    ap.add_argument("--imgB", default=None)
    ap.add_argument("--report", default=None, help="Se fornecido, escreve relatório de texto neste caminho")
    args = ap.parse_args()

    pA = read_points_csv(args.ptsA)
    pB = read_points_csv(args.ptsB)
    lines = []
    ok = True

    lines.append(f"Arquivo A: {args.ptsA}  | N_A={pA.shape[0]}")
    lines.append(f"Arquivo B: {args.ptsB}  | N_B={pB.shape[0]}")

    # 1) N iguais
    if pA.shape[0] != pB.shape[0]:
        ok = False
        lines.append("ERRO: N_A != N_B")
    else:
        lines.append("OK: N_A == N_B")

    # 2) bounds se imagens fornecidas
    if args.imgA and args.imgB:
        Aimg = np.asarray(Image.open(args.imgA).convert("RGB"))
        Bimg = np.asarray(Image.open(args.imgB).convert("RGB"))
        HA, WA = Aimg.shape[:2]; HB, WB = Bimg.shape[:2]
        def bound_checks(P, W, H, name):
            out = []
            for i,(x,y) in enumerate(P):
                if not (np.isfinite(x) and np.isfinite(y)):
                    out.append((i,"NaN/Inf"))
                if x < 0 or y < 0 or x > W-1 or y > H-1:
                    out.append((i,"fora dos limites"))
            return out
        bA = bound_checks(pA, WA, HA, "A")
        bB = bound_checks(pB, WB, HB, "B")
        if bA:
            ok = False
            lines.append(f"ERRO: {len(bA)} pontos de A fora/inválidos: {bA[:5]}{'...' if len(bA)>5 else ''}")
        else:
            lines.append("OK: pontos de A dentro dos limites")
        if bB:
            ok = False
            lines.append(f"ERRO: {len(bB)} pontos de B fora/inválidos: {bB[:5]}{'...' if len(bB)>5 else ''}")
        else:
            lines.append("OK: pontos de B dentro dos limites")

    # 3) duplicatas por CSV
    dA = has_duplicates(pA); dB = has_duplicates(pB)
    if dA:
        ok = False
        lines.append(f"ERRO: duplicatas em A (pares): {dA[:5]}{'...' if len(dA)>5 else ''}")
    else:
        lines.append("OK: sem duplicatas em A")
    if dB:
        ok = False
        lines.append(f"ERRO: duplicatas em B (pares): {dB[:5]}{'...' if len(dB)>5 else ''}")
    else:
        lines.append("OK: sem duplicatas em B")

    # 4) afim global
    if pA.shape[0] >= 3 and pA.shape[0] == pB.shape[0]:
        M, err_aff = fit_affine(pA, pB)
        lines.append(f"Afim LS: RMS = {rms(err_aff):.3f} px (mediana {np.median(err_aff):.3f} px)")
    else:
        lines.append("Afim LS: insuficiente (N<3 ou N_A!=N_B)")

    # 5) similaridade (Procrustes)
    if pA.shape[0] >= 2 and pA.shape[0] == pB.shape[0]:
        s, R, t, err_sim = procrustes_similarity(pA, pB)
        ang = np.degrees(np.arctan2(R[1,0], R[0,0]))
        lines.append(f"Similaridade: escala={s:.4f}, rotação={ang:.2f}°, RMS = {rms(err_sim):.3f} px")
    else:
        lines.append("Similaridade: insuficiente (N<2 ou N_A!=N_B)")

    # 6) flips de orientação (Delaunay em pM)
    if pA.shape[0] >= 3 and pA.shape[0] == pB.shape[0]:
        simp, flips = check_orientations(pA, pB)
        if flips:
            ok = False
            lines.append(f"ERRO: {len(flips)} triângulos com orientação invertida entre A e B (pode causar dobras). Ex.: {flips[:3]}")
        else:
            lines.append("OK: sem inversões de orientação (base Delaunay nos pontos médios)")
    else:
        lines.append("Orientação: insuficiente (N<3 ou N_A!=N_B)")

    lines.append("STATUS FINAL: " + ("OK" if ok else "ATENÇÃO — veja erros acima"))

    report = "\n".join(lines)
    print(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nRelatório salvo em: {args.report}")

if __name__ == "__main__":
    main()
