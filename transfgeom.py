#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ.setdefault("NUMBA_DISABLE_TBB_FORK_WARNING", "1")

"""
transfgeom.py

Telas:
1) Anotações pareadas
   - Carregar imagem A/B
   - Adicionar/Mover/Remover pontos (pareados)
   - Salvar/Carregar CSV A/B
   - "Adicionar cantos ao salvar"
   - Botão "Checar pontos…" abre relatório (check_points.py) em janela flutuante

2) Remover Fundo (IA)
   - Selecionar imagem de entrada
   - Escolher cor do fundo (preto/branco)
   - Salvar imagem processada

3) Morph
   - Selecionar imagens A/B (ou reutilizar as carregadas na tela de Anotações)
   - Selecionar CSV A/B (ou reutilizar os da tela de Anotações)
   - Parâmetros: num_frames, fps, k (sigmoide), exibir malhas, exibir funções
   - Saídas: diretório, gif_out, video_out (opcionais)
   - Botão "Gerar" executa o pipeline do morph (importando funções de morph.py)

Dependências: numpy, pillow, tkinter, ttk, rembg, scipy, imageio, opencv-python
(Matplotlib é usado apenas dentro do MorphTab opcionalmente, sem impactar as anotações)
"""

import traceback
import threading
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import re

# Opcional/externo — usamos as funções prontas do morph.py, se disponível
try:
    import lib.morph as M
except Exception:
    M = None

# Opcional/externo — checagem de pontos
try:
    import lib.check_points as CP
except Exception:
    CP = None

# Opcional/externo — remoção de fundo
try:
    from rembg import remove, new_session
except Exception:
    remove = None
    new_session = None


# ------------- Tipos de arquivos --------------
FILETYPES_IMGS = [
    ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
    ("Todos os arquivos", "*"),
]
FILETYPES_PNGJPG = [
    ("PNG", "*.png"),
    ("JPG", "*.jpg *.jpeg"),
    ("Todos os arquivos", "*"),
]
FILETYPES_CSV = [
    ("CSV", "*.csv"),
    ("Todos os arquivos", "*"),
]


# ---------- Utilidades compartilhadas ----------

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

def write_points_csv(path, pts):
    with open(path, "w", encoding="utf-8") as f:
        for x,y in pts:
            f.write(f"{x:.6f},{y:.6f}\n")

def add_corners_basic(pts, W, H):
    # Versão básica (4 cantos) — suficiente para anotações
    corners = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    keep = []
    for p in pts:
        if not any(np.allclose(p,c,atol=1e-6) for c in corners):
            keep.append(p)
    out = np.array(keep, dtype=np.float32) if keep else np.zeros((0,2), dtype=np.float32)
    return np.vstack([out, corners])


# ---------- Viewer sem Matplotlib ----------

class ImageAnnotCanvas(ttk.Frame):
    """
    Viewer de imagem com anotações em tk.Canvas.
    - Preserva aspect ratio (sem distorção) com letterbox/pillarbox.
    - Converte cliques no canvas para coordenadas da imagem (x_img, y_img).
    - Desenha/atualiza pontos e rótulos sem jitter.
    """
    def __init__(self, master, on_click=None, point_radius=4, **kwargs):
        super().__init__(master, **kwargs)
        self.on_click = on_click
        self.point_radius = point_radius

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Estado
        self.img_np = None       # np.uint8 [H,W,3]
        self._tkimg = None       # PhotoImage
        self.H = self.W = 0
        self.pts = np.zeros((0,2), dtype=np.float32)

        # Retângulo útil de exibição dentro do canvas
        self.view_x = 0
        self.view_y = 0
        self.view_w = 1
        self.view_h = 1

        # Objetos desenhados
        self._img_id = None
        self._pt_ids = []    # círculos
        self._lbl_ids = []   # textos

        # Eventos
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_click)

    # ------- API pública -------
    def set_image(self, img_np):
        self.img_np = img_np
        self.H, self.W = img_np.shape[0], img_np.shape[1]
        self._render()

    def set_points(self, pts):
        """pts: array Nx2 em coords da IMAGEM (float32)."""
        self.pts = np.asarray(pts, dtype=np.float32) if pts is not None else np.zeros((0,2), np.float32)
        self._draw_points()

    # ------- Conversões coords -------
    def canvas_to_img(self, x_canvas, y_canvas):
        if self.img_np is None or self.view_w <= 0 or self.view_h <= 0:
            return None, None
        u = (x_canvas - self.view_x) / self.view_w
        v = (y_canvas - self.view_y) / self.view_h
        if u < 0 or v < 0 or u > 1 or v > 1:
            return None, None
        x_img = u * (self.W - 1)
        y_img = v * (self.H - 1)
        return float(x_img), float(y_img)

    def img_to_canvas(self, x_img, y_img):
        if self.img_np is None or self.W <= 1 or self.H <= 1:
            return None, None
        x = self.view_x + (x_img / (self.W - 1)) * self.view_w
        y = self.view_y + (y_img / (self.H - 1)) * self.view_h
        return x, y

    # ------- Internos -------
    def _on_resize(self, _event):
        self._render()

    def _on_click(self, event):
        if self.on_click is None:
            return
        xi, yi = self.canvas_to_img(event.x, event.y)
        if xi is not None and yi is not None:
            self.on_click(xi, yi)

    def _render(self):
        # Remove imagem anterior (se houver)
        if self._img_id is not None:
            try:
                self.canvas.delete(self._img_id)
            except Exception:
                pass
            self._img_id = None

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if self.img_np is None or self.W == 0 or self.H == 0:
            self._clear_points()
            return

        W, H = self.W, self.H

        # ---- NOVA REGRA DE ESCALA (sem upscale) ----
        # Só reduz quando necessário; se couber, mantém tamanho original
        if W <= cw and H <= ch:
            # Sem redimensionar
            target_w = W
            target_h = H
        else:
            # Precisa reduzir em pelo menos uma dimensão
            if W > cw and H > ch:
                scale = min(cw / W, ch / H)  # "nivela pela maior delas" (maior excesso)
            elif W > cw:  # só largura estoura
                scale = cw / W
            else:         # só altura estoura
                scale = ch / H
            # segurança
            scale = max(1e-6, min(1.0, scale))
            target_w = max(1, int(round(W * scale)))
            target_h = max(1, int(round(H * scale)))
        # --------------------------------------------

        # Calcula retângulo de exibição centralizado (letterbox/pillarbox)
        self.view_w, self.view_h = target_w, target_h
        self.view_x = (cw - self.view_w) // 2
        self.view_y = (ch - self.view_h) // 2

        # Redimensiona a imagem só se necessário (mantendo aspecto)
        from PIL import Image, ImageTk
        pil = Image.fromarray(self.img_np)
        if target_w != W or target_h != H:
            # Escolha o filtro: NEAREST (pixelado) ou BILINEAR/BICUBIC para fotos
            pil = pil.resize((self.view_w, self.view_h), resample=Image.NEAREST)
        self._tkimg = ImageTk.PhotoImage(pil)

        # Desenha imagem
        self._img_id = self.canvas.create_image(self.view_x, self.view_y, image=self._tkimg, anchor="nw")

        # Redesenha pontos/labels de acordo com a nova escala
        self._draw_points()


    def _clear_points(self):
        for i in self._pt_ids:
            try: self.canvas.delete(i)
            except Exception: pass
        for i in self._lbl_ids:
            try: self.canvas.delete(i)
            except Exception: pass
        self._pt_ids.clear()
        self._lbl_ids.clear()

    def _draw_points(self):
        self._clear_points()
        if self.pts is None or self.pts.shape[0] == 0 or self.img_np is None:
            return
        r = self.point_radius
        for idx, (ximg, yimg) in enumerate(self.pts):
            xc, yc = self.img_to_canvas(ximg, yimg)
            if xc is None:
                continue
            pid = self.canvas.create_oval(xc-r, yc-r, xc+r, yc+r, outline="cyan", width=2)
            tid = self.canvas.create_text(xc+8, yc-8, text=str(idx), fill="yellow",
                                          font=("Arial", 10, "bold"))
            self._pt_ids.append(pid)
            self._lbl_ids.append(tid)


# ---------- Tela 1: Anotações pareadas (sem Matplotlib) ----------

class AnnotateTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        # --- Estilos visuais para botões selecionados ---
        style = ttk.Style(self)
        style.configure("Mode.TRadiobutton", padding=4)
        style.configure("Side.TRadiobutton", padding=4)

        # Muitos temas do ttk ignoram 'background', então deixamos o banner e os contornos como principal feedback.
        # Ainda assim, vale mapear o 'foreground' para diferenciar quando selecionado.
        style.map("Mode.TRadiobutton", foreground=[("selected", "#0d6efd")])
        style.map("Side.TRadiobutton", foreground=[("selected", "#0d6efd")])

        # Estados de imagem/pontos
        self.A = (np.ones((300, 300, 3), dtype=np.uint8) * 240)
        self.WA, self.HA = self.A.shape[1], self.A.shape[0]
        self.B = (np.ones((300, 300, 3), dtype=np.uint8) * 240)
        self.WB, self.HB = self.B.shape[1], self.B.shape[0]
        self.ptsA = np.zeros((0,2), dtype=np.float32)
        self.ptsB = np.zeros((0,2), dtype=np.float32)
        self.pathA = None
        self.pathB = None
        self.csvA = None
        self.csvB = None

        # UI flags
        self.active = "A"      # "A" ou "B"
        self.mode = "add"      # add | move | remove

        # Layout principal
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = ttk.Frame(self)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.left = ttk.Frame(main)
        self.left.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.right = ttk.Frame(main, width=300)
        self.right.grid(row=0, column=1, sticky="ns")
        self.right.grid_propagate(False)

        # ----- Viewers lado a lado -----
        mid = ttk.Frame(self.left)
        mid.pack(fill=tk.BOTH, expand=True)

        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(0, weight=1)

        # Wrappers com highlight para contorno colorido
        self.wrapL = tk.Frame(mid, bd=0, highlightthickness=3)
        self.wrapL.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.wrapR = tk.Frame(mid, bd=0, highlightthickness=3)
        self.wrapR.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        self.viewL = ImageAnnotCanvas(self.wrapL, on_click=lambda x,y: self._on_click_side("A", x, y))
        self.viewL.pack(fill=tk.BOTH, expand=True)

        self.viewR = ImageAnnotCanvas(self.wrapR, on_click=lambda x,y: self._on_click_side("B", x, y))
        self.viewR.pack(fill=tk.BOTH, expand=True)

        # Inicializa com placeholders
        self.viewL.set_image(self.A)
        self.viewR.set_image(self.B)

        # Painel direito
        ttk.Label(self.right, text="Anotações pareadas", font=("Arial", 12, "bold")).pack(pady=(6,4))

        ttk.Button(self.right, text="Carregar imagem A…", command=self.load_image_a)\
            .pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(self.right, text="Carregar imagem B…", command=self.load_image_b)\
            .pack(fill=tk.X, padx=6, pady=2)

        # Banner de estado atual
        self.banner = ttk.Label(self.right, text="", font=("Arial", 11, "bold"), anchor="center")
        self.banner.pack(fill=tk.X, padx=6, pady=(6,8))

        # Ativação (lado ativo) — Radiobuttons
        ttk.Label(self.right, text="Lado ativo").pack(pady=(4,2))
        self.active_var = tk.StringVar(value=self.active)
        ttk.Radiobutton(self.right, text="Esquerda (A)", value="A",
                        variable=self.active_var, command=self._ui_active_changed,
                        style="Side.TRadiobutton").pack(fill=tk.X, padx=6, pady=2)
        ttk.Radiobutton(self.right, text="Direita (B)", value="B",
                        variable=self.active_var, command=self._ui_active_changed,
                        style="Side.TRadiobutton").pack(fill=tk.X, padx=6, pady=2)

        # Modos de edição — Radiobuttons
        ttk.Label(self.right, text="Modo de edição").pack(pady=(8,2))
        self.mode_var = tk.StringVar(value=self.mode)
        ttk.Radiobutton(self.right, text="Adicionar", value="add",
                        variable=self.mode_var, command=self._ui_mode_changed,
                        style="Mode.TRadiobutton").pack(fill=tk.X, padx=6, pady=2)
        ttk.Radiobutton(self.right, text="Mover", value="move",
                        variable=self.mode_var, command=self._ui_mode_changed,
                        style="Mode.TRadiobutton").pack(fill=tk.X, padx=6, pady=2)
        ttk.Radiobutton(self.right, text="Remover", value="remove",
                        variable=self.mode_var, command=self._ui_mode_changed,
                        style="Mode.TRadiobutton").pack(fill=tk.X, padx=6, pady=2)

        ttk.Button(self.right, text="Desfazer último", command=self.undo).pack(fill=tk.X, padx=6, pady=6)


        self.var_corners = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.right, text="Adicionar cantos ao salvar", variable=self.var_corners)\
            .pack(fill=tk.X, padx=6, pady=(0,8))

        ttk.Label(self.right, text="Salvar / Carregar CSV").pack(pady=(6,4))
        ttk.Button(self.right, text="Salvar A…", command=self.saveA).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(self.right, text="Salvar B…", command=self.saveB).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(self.right, text="Carregar A…", command=self.loadA_csv).pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(self.right, text="Carregar B…", command=self.loadB_csv).pack(fill=tk.X, padx=6, pady=2)

        ttk.Separator(self.right).pack(fill=tk.X, padx=6, pady=8)
        ttk.Button(self.right, text="Checar pontos…", command=self.check_points_popup).pack(fill=tk.X, padx=6, pady=4)

        self.status = ttk.Label(self.right, text="", anchor="w", justify="left")
        self.status.pack(fill=tk.X, padx=6, pady=8)

        self.redraw()

    # Acesso externo (por MorphTab)
    def get_current_data(self):
        return {
            "imgA_path": self.pathA,
            "imgB_path": self.pathB,
            "ptsA": self.ptsA.copy(),
            "ptsB": self.ptsB.copy(),
            "csvA": self.csvA,
            "csvB": self.csvB,
            "WA": self.WA, "HA": self.HA,
            "WB": self.WB, "HB": self.HB,
        }

    # ---------- Modos / estado ----------
    def set_mode(self, m):
        self.mode = m
        # reflete nos radiobuttons se vier de um atalho/código
        if hasattr(self, "mode_var"):
            self.mode_var.set(m)
        self._update_ui_highlights()
        self.redraw()

    def set_active(self, which):
        self.active = which
        if hasattr(self, "active_var"):
            self.active_var.set(which)
        self._update_ui_highlights()
        self.redraw()


    def _ui_mode_changed(self):
        # usuário clicou num radiobutton: aplica no estado e reflete contornos/banner
        self.set_mode(self.mode_var.get())

    def _ui_active_changed(self):
        self.set_active(self.active_var.get())

    def _update_ui_highlights(self):
        # Contorno colorido do viewer ativo
        active_color = "#0d6efd"   # azul
        idle_color   = "#444444"   # cinza
        if self.active == "A":
            self.wrapL.configure(highlightbackground=active_color, highlightcolor=active_color)
            self.wrapR.configure(highlightbackground=idle_color,   highlightcolor=idle_color)
        else:
            self.wrapR.configure(highlightbackground=active_color, highlightcolor=active_color)
            self.wrapL.configure(highlightbackground=idle_color,   highlightcolor=idle_color)

        # Banner com estado atual
        mapa_modo = {"add": "Adicionar", "move": "Mover", "remove": "Remover"}
        self.banner.config(text=f"Modo: {mapa_modo.get(self.mode, self.mode)}  |  Ativo: {self.active}")


    # ---------- Utilidades ----------
    @staticmethod
    def nearest_idx(pts, x, y):
        if pts.shape[0] == 0:
            return None
        d = np.sum((pts - np.array([x,y]))**2, axis=1)
        return int(np.argmin(d))

    def both_loaded(self):
        return (self.pathA is not None) and (self.pathB is not None)
    
    def _clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def _map_rel(self, x, y, Wsrc, Hsrc, Wdst, Hdst):
        """Mapeia (x,y) por posição relativa: (u,v) em A vira (u,v) em B."""
        if Wsrc <= 1 or Hsrc <= 1 or Wdst <= 1 or Hdst <= 1:
            return 0.0, 0.0
        u = x / (Wsrc - 1)
        v = y / (Hsrc - 1)
        xd = u * (Wdst - 1)
        yd = v * (Hdst - 1)
        # clamp defensivo
        return self._clamp(xd, 0.0, Wdst - 1), self._clamp(yd, 0.0, Hdst - 1)


    # ---------- IO imagens ----------
    def load_image_a(self):
        path = filedialog.askopenfilename(filetypes=FILETYPES_IMGS)
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.A = np.asarray(img)
            self.HA, self.WA = self.A.shape[0], self.A.shape[1]
            self.pathA = path
            self.viewL.set_image(self.A)  # atualiza viewer
            self.redraw()
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar imagem A:\n{e}")

    def load_image_b(self):
        path = filedialog.askopenfilename(filetypes=FILETYPES_IMGS)
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.B = np.asarray(img)
            self.HB, self.WB = self.B.shape[0], self.B.shape[1]
            self.pathB = path
            self.viewR.set_image(self.B)  # atualiza viewer
            self.redraw()
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar imagem B:\n{e}")

    # ---------- Cliques (vindos dos viewers) ----------
    def _on_click_side(self, tgt, x, y):
        if not self.both_loaded():
            messagebox.showinfo("Carregar imagens", "Carregue as imagens A e B antes de anotar.")
            return

        # >>> use o setter para sincronizar UI
        if tgt != self.active:
            self.set_active(tgt)

        Ais = (tgt == "A")
        pts = self.ptsA if Ais else self.ptsB
        other_pts = self.ptsB if Ais else self.ptsA

        if self.mode == "add":
            pts = np.vstack([pts, [x, y]])
            if Ais:
                xb, yb = self._map_rel(x, y, self.WA, self.HA, self.WB, self.HB)
                other_pts = np.vstack([other_pts, [xb, yb]])
            else:
                xa, ya = self._map_rel(x, y, self.WB, self.HB, self.WA, self.HA)
                other_pts = np.vstack([other_pts, [xa, ya]])

            # >>> após ADD: alterna lado e entra em "move" via setters
            self.set_active("B" if Ais else "A")
            self.set_mode("move")

        elif self.mode == "move":
            idx = self.nearest_idx(pts, x, y)
            if idx is not None:
                pts[idx] = [x, y]

            # >>> após MOVE: alterna lado e volta para "add" via setters
            self.set_active("B" if Ais else "A")
            self.set_mode("add")

        elif self.mode == "remove":
            idx = self.nearest_idx(pts, x, y)
            if idx is not None:
                if idx < other_pts.shape[0]:
                    other_pts = np.delete(other_pts, idx, axis=0)
                pts = np.delete(pts, idx, axis=0)
            # (remove não alterna o fluxo)

        if Ais:
            self.ptsA = pts
            self.ptsB = other_pts
        else:
            self.ptsB = pts
            self.ptsA = other_pts

        # >>> setters já chamam _update_ui_highlights() + redraw()
        # então aqui basta garantir os pontos atualizados:
        self.redraw()


    # ---------- Desfazer ----------
    def undo(self):
        # Se tamanhos iguais e >0, desfaz último PAR (mais previsível)
        if self.ptsA.shape[0] > 0 and self.ptsA.shape[0] == self.ptsB.shape[0]:
            self.ptsA = self.ptsA[:-1, :]
            self.ptsB = self.ptsB[:-1, :]
            self.redraw()
            return

        # Caso listas estejam desequilibradas (ex.: auto-pair desativado)
        if self.active == "A":
            if self.ptsA.shape[0] > 0:
                self.ptsA = self.ptsA[:-1, :]
                # tenta manter pares (se ainda houver sobras em B, corta também)
                if self.ptsB.shape[0] > self.ptsA.shape[0]:
                    self.ptsB = self.ptsB[:-1, :]
        else:
            if self.ptsB.shape[0] > 0:
                self.ptsB = self.ptsB[:-1, :]
                if self.ptsA.shape[0] > self.ptsB.shape[0]:
                    self.ptsA = self.ptsA[:-1, :]
        self.redraw()


    # ---------- Salvar/Carregar CSV ----------
    def suggested_outname(self, side="A"):
        base = "pontos"
        ref = self.pathA if side == "A" else self.pathB
        if ref:
            stem = os.path.splitext(os.path.basename(ref))[0]
            base = f"pontos_{stem}"
        return f"{base}.csv"

    def saveA(self):
        if not self.pathA:
            messagebox.showinfo("Salvar A", "Carregue a imagem A antes de salvar os pontos.")
            return
        pts = self.ptsA.copy()
        if self.var_corners.get():
            pts = add_corners_basic(pts, self.WA, self.HA)
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile=self.suggested_outname("A"),
                                            filetypes=FILETYPES_CSV)
        if path:
            write_points_csv(path, pts)
            self.csvA = path
            messagebox.showinfo("Salvar", f"A salvo em:\n{path}")

    def saveB(self):
        if not self.pathB:
            messagebox.showinfo("Salvar B", "Carregue a imagem B antes de salvar os pontos.")
            return
        pts = self.ptsB.copy()
        if self.var_corners.get():
            pts = add_corners_basic(pts, self.WB, self.HB)
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile=self.suggested_outname("B"),
                                            filetypes=FILETYPES_CSV)
        if path:
            write_points_csv(path, pts)
            self.csvB = path
            messagebox.showinfo("Salvar", f"B salvo em:\n{path}")

    def loadA_csv(self):
        path = filedialog.askopenfilename(filetypes=FILETYPES_CSV)
        if path:
            try:
                self.ptsA = read_points_csv(path)
                self.csvA = path
                self.redraw()
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar A:\n{e}")

    def loadB_csv(self):
        path = filedialog.askopenfilename(filetypes=FILETYPES_CSV)
        if path:
            try:
                self.ptsB = read_points_csv(path)
                self.csvB = path
                self.redraw()
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar B:\n{e}")

    # ---------- Redesenho ----------
    def redraw(self):
        # Atualiza viewers com os pontos correntes
        self.viewL.set_points(self.ptsA)
        self.viewR.set_points(self.ptsB)

        nA, nB = self.ptsA.shape[0], self.ptsB.shape[0]
        msg = f"Pontos A: {nA} — Pontos B: {nB}\n"
        if not self.both_loaded():
            msg += "Carregue as imagens A e B para habilitar a edição.\n"
        elif nA != nB:
            msg += "⚠ N_A != N_B (garanta igualdade antes do morph)\n"
        msg += f"Modo: {self.mode} | Ativa: {self.active}"
        self.status["text"] = msg

    # ---------- Checagem de pontos ----------
    def check_points_popup(self):
        if self.ptsA.shape[0] == 0 or self.ptsB.shape[0] == 0:
            messagebox.showinfo("Checar pontos", "Carregue/anote pontos A e B antes de checar.")
            return

        def run_check():
            try:
                pA = self.ptsA
                pB = self.ptsB
                nA = int(pA.shape[0]); nB = int(pB.shape[0])

                # --- Bounds (se conseguirmos ler dimensões reais) ---
                bA = []; bB = []
                if self.pathA and self.pathB:
                    try:
                        Aimg = np.asarray(Image.open(self.pathA).convert("RGB"))
                        Bimg = np.asarray(Image.open(self.pathB).convert("RGB"))
                        HA, WA = Aimg.shape[:2]; HB, WB = Bimg.shape[:2]
                        def bound_checks(P, W, H):
                            out = []
                            for i,(x,y) in enumerate(P):
                                if not (np.isfinite(x) and np.isfinite(y)):
                                    out.append((i,"NaN/Inf"))
                                if x < 0 or y < 0 or x > W-1 or y > H-1:
                                    out.append((i,"fora dos limites"))
                            return out
                        bA = bound_checks(pA, WA, HA)
                        bB = bound_checks(pB, WB, HB)
                    except Exception:
                        pass

                # --- Duplicatas (bem próximo) ---
                def has_duplicates(P, tol=1e-6):
                    if P.shape[0] < 2: return []
                    idxs = []
                    for i in range(P.shape[0]):
                        d2 = ((P - P[i])**2).sum(axis=1)
                        d2[i] = 1e9
                        j = int(np.argmin(d2))
                        if d2[j] < tol**2:
                            idxs.append(tuple(sorted((i,j))))
                    idxs = sorted(set(idxs))
                    return idxs
                dA = has_duplicates(pA)
                dB = has_duplicates(pB)

                # --- Ajustes afim / similaridade ---
                rms_aff = None; med_aff = None
                escala = None; rot_deg = None; rms_sim = None
                flips = []

                if (CP is not None) and (nA == nB) and (nA >= 2):
                    if nA >= 3:
                        _, err_aff = CP.fit_affine(pA, pB)
                        rms_aff = float(np.sqrt(np.mean(err_aff**2))) if err_aff.size else 0.0
                        med_aff = float(np.median(err_aff)) if err_aff.size else 0.0
                    s, R, t, err_sim = CP.procrustes_similarity(pA, pB)
                    escala = float(s)
                    rot_deg = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
                    rms_sim = float(np.sqrt(np.mean(err_sim**2))) if err_sim.size else 0.0

                    if nA >= 3:
                        _, flips_l = CP.check_orientations(pA, pB)
                        # normaliza para inteiros “puros” na visualização
                        def to_int_trip(tup):
                            try:
                                return tuple(int(x) for x in tup)
                            except Exception:
                                return tup
                        flips = [to_int_trip(t) for t in flips_l]

                # --- Monta relatório amigável ---
                def bloco_pontos():
                    ok = (nA == nB)
                    return (
                        "📋 **Número de pontos marcados**\n"
                        f"- Imagem A: {nA}\n"
                        f"- Imagem B: {nB}\n"
                        + ("✅ Os dois conjuntos possuem o mesmo número de pontos."
                        if ok else "⚠️ As imagens possuem quantidades diferentes de pontos!")
                    )

                def bloco_bounds():
                    msgs = []
                    if bA:
                        msgs.append(f"⚠️ A: {len(bA)} ponto(s) fora/inválido(s) (ex.: {bA[:5]})")
                    if bB:
                        msgs.append(f"⚠️ B: {len(bB)} ponto(s) fora/inválido(s) (ex.: {bB[:5]})")
                    if not msgs:
                        return "✅ Todos os pontos parecem dentro dos limites das imagens."
                    return "\n".join(msgs)

                def bloco_dup():
                    msgs = []
                    if dA:
                        msgs.append(f"⚠️ Duplicatas em A: {dA[:5]}")
                    if dB:
                        msgs.append(f"⚠️ Duplicatas em B: {dB[:5]}")
                    if not msgs:
                        return "✅ Nenhuma duplicata aparente nos pontos."
                    return "\n".join(msgs)

                def bloco_afim():
                    if rms_aff is None:
                        return "ℹ️ Transformação afim: não avaliada (poucos pontos ou módulos ausentes)."
                    return (
                        "**Erro médio da transformação afim (Least Squares)**\n"
                        f"- RMS = {rms_aff:.3f} px\n"
                        f"- Mediana = {med_aff:.3f} px\n\n"
                        "💡 Esses valores indicam o quanto, em média, os pontos transformados da imagem A se desviam "
                        "dos correspondentes na imagem B. Quanto menores os valores, mais consistentes estão as marcações."
                    )

                def bloco_sim():
                    if rms_sim is None or escala is None or rot_deg is None:
                        return "ℹ️ Transformação de similaridade: não avaliada (poucos pontos ou módulos ausentes)."
                    return (
                        "**Transformação de similaridade (escala e rotação)**\n"
                        f"- Escala estimada: {escala:.4f}\n"
                        f"- Rotação estimada: {rot_deg:.2f}°\n"
                        f"- RMS = {rms_sim:.3f} px\n\n"
                        "📐 A transformação de similaridade tenta alinhar por rotação, translação e escala. "
                        "Escala ~1 e rotação pequena indicam alinhamento razoável em tamanho/orientação."
                    )

                def bloco_flips():
                    if not flips:
                        return "✅ **Orientação dos triângulos**: nenhum triângulo invertido foi detectado (Delaunay nos pontos médios)."
                    exemplos = ", ".join(str(t) for t in flips[:3])
                    return (
                        f"⚠️ **Orientação dos triângulos**: {len(flips)} triângulos com orientação invertida.\n"
                        f"Exemplos: [{exemplos}]\n\n"
                        "🔍 Triângulos invertidos indicam que parte da malha ficou “espelhada” entre as imagens, "
                        "em geral por inconsistência na ordem dos pontos correspondentes; isso pode causar distorções locais no morph."
                    )

                def bloco_recos():
                    return (
                        "🧭 **Recomendações**\n"
                        "- Garanta que os pontos sigam a mesma ordem lógica nas duas imagens (ex.: varrer da esquerda p/ direita, topo p/ base).\n"
                        "- Evite pontos muito colados (triângulos degenerados pioram a malha).\n"
                        "- Se RMS estiver alto (p.ex. > ~20 px), revise possíveis pares errados: poucos erros elevam o RMS.\n"
                        "- Triângulos invertidos não impedem o morph, mas degradam a suavidade da animação."
                    )

                texto = (
                    "📋 **RELATÓRIO DE CORRESPONDÊNCIA DE PONTOS**\n\n"
                    + bloco_pontos() + "\n\n"
                    + "**Validação de limites**\n" + bloco_bounds() + "\n\n"
                    + "**Duplicatas**\n" + bloco_dup() + "\n\n"
                    + bloco_afim() + "\n\n"
                    + bloco_sim() + "\n\n"
                    + bloco_flips() + "\n\n"
                    + bloco_recos()
                )

                self.show_text_popup("Relatório de correspondência", texto)
            except Exception as e:
                self.show_text_popup("Erro ao checar", f"{e}\n\n{traceback.format_exc()}")

        threading.Thread(target=run_check, daemon=True).start()


    def show_text_popup(self, title, text):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("820x560")
        win.minsize(520, 360)

        # Frame principal
        container = ttk.Frame(win)
        container.pack(fill=tk.BOTH, expand=True)

        # ScrolledText com fonte monoespaçada (melhor p/ relatório)
        txt = scrolledtext.ScrolledText(
            container, wrap="word", undo=False,
            font=("TkFixedFont", 11), padx=8, pady=8
        )
        txt.pack(fill=tk.BOTH, expand=True)

        # Tags de estilo
        txt.tag_configure("bold", font=("TkFixedFont", 11, "bold"))
        # (opcional) destaque leve para “seções”
        txt.tag_configure("h", foreground="#0d6efd")

        # Inserção “markdown-lite”: só **negrito**
        # Mantém o restante do texto como está.
        def insert_markdown(tw, s):
            bold_pat = re.compile(r"\*\*(.+?)\*\*")
            pos = 0
            for m in bold_pat.finditer(s):
                pre = s[pos:m.start()]
                if pre:
                    tw.insert("end", pre)
                tw.insert("end", m.group(1), ("bold",))
                pos = m.end()
            if pos < len(s):
                tw.insert("end", s[pos:])

        # Dica: se quiser colorir linhas-título, aplique tag "h" nas linhas que começam com emoji:
        lines = text.splitlines(keepends=True)
        for line in lines:
            if line.lstrip().startswith(("📋", "🧭", "📐", "💡", "⚠️", "✅", "ℹ️", "🔍")):
                # Quebra a linha para aplicar markdown + cor
                start_idx = txt.index("end-1c")
                insert_markdown(txt, line)
                end_idx = txt.index("end-1c")
                txt.tag_add("h", start_idx, end_idx)
            else:
                insert_markdown(txt, line)

        txt.config(state="disabled")

        # Barra de botões (Copiar / Salvar / Fechar)
        bar = ttk.Frame(container)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        def do_copy():
            try:
                win.clipboard_clear()
                # pega todo o conteúdo já renderizado
                raw = text
                win.clipboard_append(raw)
            except Exception:
                pass

        def do_save():
            p = filedialog.asksaveasfilename(
                parent=win, defaultextension=".txt",
                filetypes=[("Texto", "*.txt"), ("Todos", "*")]
            )
            if p:
                try:
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(text)
                    messagebox.showinfo("Salvar relatório", f"Relatório salvo em:\n{p}", parent=win)
                except Exception as e:
                    messagebox.showerror("Salvar relatório", f"Falha ao salvar:\n{e}", parent=win)

        ttk.Button(bar, text="Copiar", command=do_copy).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(bar, text="Salvar…", command=do_save).pack(side=tk.LEFT, padx=0, pady=6)
        ttk.Button(bar, text="Fechar", command=win.destroy).pack(side=tk.RIGHT, padx=6, pady=6)



# ---------- Tela 2: Remoção de Fundo ----------

class NoBgTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.in_path = tk.StringVar(value="")
        self.out_path = tk.StringVar(value="")
        self.bg_white = tk.BooleanVar(value=False)

        # Layout
        pad = {"padx": 8, "pady": 4}
        ttk.Label(self, text="Remoção de Plano de Fundo (IA)", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", **pad)

        ttk.Label(self, text="Imagem de entrada:").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.in_path, width=60).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(self, text="…", command=self.pick_in).grid(row=1, column=2, sticky="w", **pad)

        ttk.Label(self, text="Imagem de saída:").grid(row=2, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.out_path, width=60).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(self, text="…", command=self.pick_out).grid(row=2, column=2, sticky="w", **pad)

        ttk.Checkbutton(self, text="Fundo branco (desmarcado = fundo preto)", variable=self.bg_white).grid(row=3, column=1, sticky="w", **pad)

        ttk.Button(self, text="Processar", command=self.run_remove).grid(row=4, column=1, sticky="e", **pad)

        self.columnconfigure(1, weight=1)

    def pick_in(self):
        p = filedialog.askopenfilename(filetypes=FILETYPES_IMGS)
        if p:
            self.in_path.set(p)

    def pick_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=FILETYPES_PNGJPG)
        if p:
            self.out_path.set(p)

    def run_remove(self):
        if remove is None or new_session is None:
            messagebox.showerror("Dependência ausente", "Pacote 'rembg' não encontrado.")
            return
        inp = self.in_path.get().strip()
        out = self.out_path.get().strip()
        if not inp or not out:
            messagebox.showinfo("Remover fundo", "Informe arquivos de entrada e saída.")
            return

        def work():
            try:
                # força CPU, alinhado ao script no_bg.py
                session = new_session("u2net", providers=["CPUExecutionProvider"])
                img = Image.open(inp)
                sem_fundo = remove(img, session=session)
                cor = (255,255,255) if self.bg_white.get() else (0,0,0)
                fundo = Image.new("RGB", sem_fundo.size, cor)
                fundo.paste(sem_fundo, mask=sem_fundo.split()[3])
                fundo.save(out)
                messagebox.showinfo("OK", f"Imagem salva em:\n{out}")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao processar:\n{e}")

        threading.Thread(target=work, daemon=True).start()


# ---------- Tela 3: Morph ----------

class MorphTab(ttk.Frame):
    def __init__(self, master, annotate_tab: AnnotateTab):
        super().__init__(master)
        self.master = master
        self.annotate_tab = annotate_tab

        # Entradas
        self.imgA = tk.StringVar(value="")
        self.imgB = tk.StringVar(value="")
        self.csvA = tk.StringVar(value="")
        self.csvB = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value="output")
        self.gif_out = tk.StringVar(value="morph.gif")
        self.video_out = tk.StringVar(value="")  # vazio = não gera
        self.num_frames = tk.IntVar(value=60)
        self.fps = tk.IntVar(value=30)
        self.k_sig = tk.DoubleVar(value=6.0)
        self.show_mesh = tk.BooleanVar(value=False)
        
        # Layout
        pad = {"padx": 8, "pady": 4}
        ttk.Label(self, text="Morph entre duas imagens", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=4, sticky="w", **pad)

        # Fonte de dados
        ttk.Button(self, text="Usar dados da aba Anotações", command=self.pull_from_annotate).grid(row=1, column=0, columnspan=4, sticky="w", **pad)

        ttk.Label(self, text="Imagem A:").grid(row=2, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.imgA, width=50).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(self, text="…", command=lambda:self.pick_path(self.imgA)).grid(row=2, column=2, sticky="w", **pad)

        ttk.Label(self, text="Imagem B:").grid(row=3, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.imgB, width=50).grid(row=3, column=1, sticky="we", **pad)
        ttk.Button(self, text="…", command=lambda:self.pick_path(self.imgB)).grid(row=3, column=2, sticky="w", **pad)

        ttk.Label(self, text="CSV A:").grid(row=4, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.csvA, width=50).grid(row=4, column=1, sticky="we", **pad)
        ttk.Button(self, text="…", command=lambda:self.pick_path(self.csvA, csv=True)).grid(row=4, column=2, sticky="w", **pad)

        ttk.Label(self, text="CSV B:").grid(row=5, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.csvB, width=50).grid(row=5, column=1, sticky="we", **pad)
        ttk.Button(self, text="…", command=lambda:self.pick_path(self.csvB, csv=True)).grid(row=5, column=2, sticky="w", **pad)

        ttk.Label(self, text="Pasta saída:").grid(row=6, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.out_dir, width=50).grid(row=6, column=1, sticky="we", **pad)
        ttk.Button(self, text="…", command=self.pick_outdir).grid(row=6, column=2, sticky="w", **pad)

        ttk.Label(self, text="GIF (opcional):").grid(row=7, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.gif_out, width=30).grid(row=7, column=1, sticky="w", **pad)

        ttk.Label(self, text="Vídeo MP4 (opcional):").grid(row=8, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.video_out, width=30).grid(row=8, column=1, sticky="w", **pad)

        ttk.Separator(self).grid(row=9, column=0, columnspan=4, sticky="we", **pad)

        ttk.Label(self, text="num_frames").grid(row=10, column=0, sticky="e", **pad)
        ttk.Spinbox(self, from_=2, to=1000, textvariable=self.num_frames, width=8).grid(row=10, column=1, sticky="w", **pad)
        ttk.Label(self, text="fps").grid(row=10, column=2, sticky="e", **pad)
        ttk.Spinbox(self, from_=1, to=120, textvariable=self.fps, width=8).grid(row=10, column=3, sticky="w", **pad)

        ttk.Label(self, text="k (sigmoide RGB)").grid(row=11, column=0, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.k_sig, width=10).grid(row=11, column=1, sticky="w", **pad)

        # Botão para plotar funções (independente da geração)
        self.plot_btn = ttk.Button(self, text="Plotar funções", command=self.plot_funcs)
        self.plot_btn.grid(row=12, column=0, columnspan=1, sticky="w", padx=8, pady=4)


        ttk.Checkbutton(self, text="Salvar malhas (gera PNGs auxiliares)", variable=self.show_mesh).grid(row=13, column=1, sticky="w", **pad)
        

        # Área de execução: progresso + status
        self.prog = ttk.Progressbar(self, mode="determinate", maximum=100, value=0)
        self.prog.grid(row=14, column=0, columnspan=3, sticky="we", padx=8, pady=(6,2))

        self.status_lbl = ttk.Label(self, text="", anchor="w")
        self.status_lbl.grid(row=15, column=0, columnspan=4, sticky="we", padx=8, pady=(0,8))

        # Guarde referência do botão gerar para desabilitar habilitar
        self.run_btn = ttk.Button(self, text="Gerar", command=self.run_morph)
        self.run_btn.grid(row=13, column=3, sticky="e", padx=8, pady=4)

        for c in (1,):
            self.columnconfigure(c, weight=1)

    def _set_busy(self, busy: bool):
        """Habilita/desabilita controles durante execução e zera progresso ao finalizar."""
        def apply():
            state = "disabled" if busy else "normal"
            for w in [
                # campos de entrada principais
                # (se quiser ser mais granular, inclua cada Entry/Spinbox/Buttons)
            ]:
                try: w.configure(state=state)
                except Exception: pass
            # Botões principais
            try: self.run_btn.configure(state=state)
            except Exception: pass
            try: self.plot_btn.configure(state=state)
            except Exception: pass

            if not busy:
                self.prog['value'] = 0
                self.status_lbl['text'] = ""
        self.after(0, apply)

    def _progress(self, pct=None, msg=None):
        """Atualiza barra e status a partir da thread de trabalho."""
        def apply():
            if pct is not None:
                self.prog['value'] = max(0, min(100, pct))
            if msg is not None:
                self.status_lbl['text'] = msg
        self.after(0, apply)


    def pick_path(self, var, csv=False):
        p = filedialog.askopenfilename(filetypes=FILETYPES_CSV if csv else FILETYPES_IMGS)
        if p:
            var.set(p)

    def pick_outdir(self):
        p = filedialog.askdirectory()
        if p: self.out_dir.set(p)

    def pull_from_annotate(self):
        data = self.annotate_tab.get_current_data()
        if data["imgA_path"]: self.imgA.set(data["imgA_path"])
        if data["imgB_path"]: self.imgB.set(data["imgB_path"])
        if data["csvA"]: self.csvA.set(data["csvA"])
        if data["csvB"]: self.csvB.set(data["csvB"])
        # Se não tiver CSV salvo, oferecemos salvar rápido
        if not data["csvA"] or not data["csvB"]:
            if data["ptsA"].shape[0] == data["ptsB"].shape[0] and data["ptsA"].shape[0] > 0:
                if messagebox.askyesno("Salvar CSVs", "Você ainda não salvou os CSVs. Deseja salvar agora?"):
                    baseA = os.path.splitext(os.path.basename(data["imgA_path"] or "A"))[0]
                    baseB = os.path.splitext(os.path.basename(data["imgB_path"] or "B"))[0]
                    fa = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"pontos_{baseA}.csv", filetypes=FILETYPES_CSV)
                    if fa:
                        write_points_csv(fa, data["ptsA"])
                        self.csvA.set(fa)
                    fb = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"pontos_{baseB}.csv", filetypes=FILETYPES_CSV)
                    if fb:
                        write_points_csv(fb, data["ptsB"])
                        self.csvB.set(fb)

    def plot_funcs(self):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            # Usa as funções do morph.py se disponíveis; senão, fallback equivalente
            F = max(2, int(self.num_frames.get()))
            k = float(self.k_sig.get())
            x = np.linspace(0, 1, F)
            if M is not None:
                y_lin = M.linear(x)
                y_sig = M.sigmoide(x, k)
            else:
                # fallback simples
                y_lin = x
                y_sig = 1.0/(1.0 + np.exp(-k*(x-0.5)))

            plt.figure()
            plt.plot(x, y_lin, label="Linear (malha)")
            plt.plot(x, y_sig, label=f"Sigmoide (k={k:g})")
            plt.title("Funções de Interpolação")
            plt.xlabel("t")
            plt.ylabel("f(t)")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Plot", f"Falha ao plotar funções:\n{e}")

    def run_morph(self):
        self._set_busy(True)
        self._progress(0, "Preparando...")

        imgA = self.imgA.get().strip()
        imgB = self.imgB.get().strip()
        csvA = self.csvA.get().strip()
        csvB = self.csvB.get().strip()
        outd = self.out_dir.get().strip() or "output"
        gifn = self.gif_out.get().strip()
        mp4n = self.video_out.get().strip()
        F = max(2, int(self.num_frames.get()))
        fps = int(self.fps.get())
        k = float(self.k_sig.get())
        
        if not (imgA and imgB and csvA and csvB):
            messagebox.showinfo("Morph", "Informe imagem A/B e CSV A/B (ou puxe da aba de Anotações).")
            return

        def work():
            try:
                os.makedirs(outd, exist_ok=True)

                if M is None:
                    raise RuntimeError("morph.py não encontrado no PYTHONPATH/pasta corrente.")

                A = M.carrega_img_float(imgA)
                B = M.carrega_img_float(imgB)
                H, W = A.shape[:2]
                if B.shape[:2] != (H, W):
                    raise ValueError("As imagens A e B devem ter o MESMO tamanho (HxW).")

                pA = M.carrega_csv(csvA)
                pB = M.carrega_csv(csvB)
                pA = M.adiciona_cantos(pA, W, H)
                pB = M.adiciona_cantos(pB, W, H)
                if pA.shape != pB.shape:
                    raise ValueError("Listas de pontos (após cantos) devem ter mesmo tamanho e ordem.")

                # Triangulação em pontos médios
                T = M.indices_pontos_medios(pA, pB)

                # Total de passos para a barra:
                total_steps = 0
                total_steps += F                   # gerar frames
                # salvar: gif ou mp4 contam como 1 passo cada; PNG final conta 1
                will_save_gif = bool(gifn)
                will_save_mp4 = bool(mp4n)
                if will_save_gif: total_steps += 1
                if will_save_mp4: total_steps += 1
                if not (will_save_gif or will_save_mp4): total_steps += 1  # PNG final

                step = 0
                def bump(msg):
                    nonlocal step
                    step += 1
                    pct = int(round(100 * step / max(1, total_steps)))
                    self._progress(pct, msg)

                # Overlays de malha (gera PNGs auxiliares), fora da barra (opcional)
                if bool(self.show_mesh.get()):
                    self._progress(None, "Desenhando malhas...")
                    overlayA = M.desenha_triangulos(A, pA, T)
                    M.float2img(overlayA).save(os.path.join(outd, "overlay_triangulacaoA.png"))
                    overlayB = M.desenha_triangulos(B, pB, T)
                    M.float2img(overlayB).save(os.path.join(outd, "overlay_triangulacaoB.png"))

                frames = []
                for f in range(F):
                    t = 0.0 if F == 1 else f / (F - 1)
                    alfa = M.linear(t)          # geometria
                    beta = M.sigmoide(t, k)     # mistura RGB

                    It = M.gera_frame(A, B, pA, pB, T, alfa, beta)

                    if self.show_mesh.get():
                        # Pontos intermediários para desenhar a malha no frame corrente
                        pT = (1.0 - alfa) * pA + alfa * pB
                        It = M.desenha_triangulos(It, pT, T, color=(1.0, 0.0, 0.0), alpha=0.9)
                    frames.append(It)
                    bump(f"Gerando frames... ({f+1}/{F})")

                # Saídas
                saved = []
                if gifn:
                    self._progress(None, "Salvando GIF...")
                    frames_png = [M.float2img(fr) for fr in frames]
                    gif_path = M.frames2gif(frames_png, outd, fname=gifn, fps=fps)
                    saved.append(("GIF", gif_path))
                    bump("GIF salvo.")

                if mp4n:
                    self._progress(None, "Salvando MP4...")
                    import imageio.v2 as imageio
                    frames_padded = [M.float2img(M.padding(f)) for f in frames]
                    mp4_path = os.path.join(outd, mp4n)
                    imageio.mimwrite(mp4_path, frames_padded, fps=fps, quality=8, format='ffmpeg')
                    saved.append(("MP4", mp4_path))
                    bump("MP4 salvo.")

                if not saved:
                    self._progress(None, "Salvando PNG final...")
                    finalP = os.path.join(outd, "morph_final.png")
                    M.float2img(frames[-1]).save(finalP)
                    saved.append(("PNG", finalP))
                    bump("PNG salvo.")

                msg_done = "Arquivos gerados:\n" + "\n".join([f"- {t}: {p}" for t,p in saved])
                self._progress(100, "Concluído.")
                # Mostrar caixa ao final (pela thread principal)
                self.after(0, lambda: messagebox.showinfo("Concluído", msg_done))

            except Exception as e:
                traceback.print_exc()
                self.after(0, lambda: messagebox.showerror("Erro no morph", f"{e}"))
            finally:
                self._set_busy(False)

        threading.Thread(target=work, daemon=True).start()

# ---------- App principal ----------

class TransfGeomApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transformações Geométricas - Warping e Morphing")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_annot = AnnotateTab(self.notebook)
        self.notebook.add(self.tab_annot, text="Anotações")

        self.tab_nobg = NoBgTab(self.notebook)
        self.notebook.add(self.tab_nobg, text="Remover Fundo")

        self.tab_morph = MorphTab(self.notebook, self.tab_annot)
        self.notebook.add(self.tab_morph, text="Morph")


def main():
    app = TransfGeomApp()
    app.mainloop()


if __name__ == "__main__":
    main()
