import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

# ------------------------- Funções fornecidas (prontas) -------------------------

# Conversão imagem <-> array float [0,1]
def carrega_img_float(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def float2img(input_arr):
    arr = (input_arr * 255).round().clip(0,255).astype(np.uint8)
    return Image.fromarray(arr)

# CSV de pontos (x,y)
def carrega_csv(path):
    """
    Lê CSV simples (sem cabeçalho) com 2 colunas: x,y (em pixels).
    Retorna array float32 de shape (N,2). Ignora linhas vazias e '#'.
    """
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            if len(parts) < 2:
                raise ValueError(f"Linha inválida no CSV: {line}")
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            xs.append(x); ys.append(y)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    return pts

def adiciona_cantos(pts, W, H):
    """
    Garante existência de 4 cantos e 4 médios de borda:
    - Cantos: (0,0), (W-1,0), (W-1,H-1), (0,H-1)
    - Médios: ((W-1)/2, 0), (W-1, (H-1)/2), ((W-1)/2, H-1), (0, (H-1)/2)
    Não duplica se já existirem (tolerância 1e-3 em distância²).
    """
    Wf = float(W - 1)
    Hf = float(H - 1)

    corners = np.array([
        [0.0, 0.0],
        [Wf,  0.0],
        [Wf,  Hf ],
        [0.0, Hf ]
    ], dtype=np.float32)

    mids = np.array([
        [Wf/2.0, 0.0],     # topo
        [Wf,     Hf/2.0],  # direita
        [Wf/2.0, Hf],      # base
        [0.0,    Hf/2.0]   # esquerda
    ], dtype=np.float32)

    targets = np.vstack([corners, mids])  # 8 pontos

    out = np.array(pts, dtype=np.float32).copy()
    if out.size == 0:
        return targets

    for c in targets:
        d2 = ((out - c) ** 2).sum(axis=1) if out.shape[0] else np.array([np.inf])
        if d2.min() > 1e-3:
            out = np.vstack([out, c])

    return out

# Desenho de malha (overlay) com cor/alpha
def _desenha_segmento(img, p, q, color=(1.0, 0.0, 0.0), alpha=0.8):
    """
    Desenha linha entre p e q (Bresenham), cor em [0,1], alpha em [0,1].
    img é float [0,1]. Não faz AA (anti-aliasing).
    """
    H, W = img.shape[:2]
    x0, y0 = int(round(p[0])), int(round(p[1]))
    x1, y1 = int(round(q[0])), int(round(q[1]))
    dx = abs(x1 - x0); dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    r, g, b = color
    a = float(alpha)

    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            if a >= 1.0:
                img[y0, x0, 0] = r
                img[y0, x0, 1] = g
                img[y0, x0, 2] = b
            else:
                img[y0, x0, :] = (1.0 - a) * img[y0, x0, :] + a * np.array([r, g, b], dtype=np.float32)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return img

def desenha_triangulos(img_float, points, triangles, color=(1.0, 0.0, 0.0), alpha=0.8):
    """
    Desenha sobreposição dos triângulos sobre a imagem (somente arestas).
    color: (r,g,b) em [0,1]; alpha: 0..1
    """
    out = img_float.copy()
    for (i, j, k) in triangles:
        for a, b in [(i, j), (j, k), (k, i)]:
            out = _desenha_segmento(out, points[a], points[b], color=color, alpha=alpha)
    return out

# Delaunay
def indices_delaunay(points_xy):
    tri = Delaunay(points_xy)
    return tri.simplices.copy()

# GIF e padding
def frames2gif(frames, out_dir, fname="morph.gif", fps=30,
               palettesize=256, dither=Image.FLOYDSTEINBERG, optimize=True):
    """
    Salva sequência de PIL.Image como GIF animado no out_dir/fname.
    """
    assert len(frames) > 0, "Lista de frames vazia"
    w, h = frames[0].size
    duration_ms = max(1, int(1000 / fps))

    proc = []
    for im in frames:
        if im.size != (w, h):
            im = im.resize((w, h), Image.LANCZOS)
        im = im.convert("P", palette=Image.ADAPTIVE, colors=palettesize, dither=dither)
        proc.append(im)

    gif_path = os.path.join(out_dir, fname)
    proc[0].save(
        gif_path,
        save_all=True,
        append_images=proc[1:],
        duration=duration_ms,
        loop=0,
        optimize=optimize,
        disposal=2
    )
    return gif_path

def padding(img, base=16):
    """
    Faz padding para múltiplos de 'base' (ex.: 16) — útil p/ codecs de vídeo.
    img: np.float32 [0,1] ou uint8; retorna no mesmo dtype.
    """
    h, w = img.shape[:2]
    new_h = ((h + base - 1) // base) * base
    new_w = ((w + base - 1) // base) * base
    border_value = (0,0,0) if img.ndim == 3 else 0
    return cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w,
                              borderType=cv2.BORDER_CONSTANT, value=border_value)
