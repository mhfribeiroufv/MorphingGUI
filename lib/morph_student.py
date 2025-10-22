import numpy as np
from .morph_core import indices_delaunay  # os alunos podem reutilizar

# ------------------------- Funções a implementar pelos estudantes -------------------------

def pontos_medios(pA, pB):
    """
    Retorna os pontos médios (N,2) entre pA e pB.
    """
    raise NotImplementedError("Implemente: pontos_medios")

def indices_pontos_medios(pA, pB):
    """
    Calcula a triangulação de Delaunay nos pontos médios e retorna (M,3) int.
    Dica: use pontos_medios + indices_delaunay().
    """
    raise NotImplementedError("Implemente: indices_pontos_medios")

# Interpoladoras
def linear(t, a=1.0, b=0.0):
    """
    Interpolação linear: a*t + b (espera-se mapear t em [0,1]).
    """
    raise NotImplementedError("Implemente: linear")

def sigmoide(t, k):
    """
    Sigmoide centrada em 0.5, normalizada para [0,1].
    k controla a "inclinação": maior k => transição mais rápida no meio.
    """
    raise NotImplementedError("Implemente: sigmoide")

def dummy(t):
    """
    Função 'dummy' que pode ser usada como exemplo de função constante.
    """
    raise NotImplementedError("Implemente: dummy")

# Geometria / warping por triângulos
def _det3(a, b, c):
    """
    Determinante 2D para área assinada (auxiliar das baricêntricas).
    """
    raise NotImplementedError("Implemente: _det3")

def _transf_baricentrica(pt, tri):
    """
    pt: (x,y)
    tri: (3,2) com vértices v1,v2,v3
    Retorna (w1,w2,w3); espera-se w1+w2+w3=1 quando pt está no plano do tri.
    """
    raise NotImplementedError("Implemente: _transf_baricentrica")

def _check_bari(w1, w2, w3, eps=1e-6):
    """
    Testa inclusão de ponto no triângulo usando baricêntricas (com tolerância).
    """
    raise NotImplementedError("Implemente: _check_bari")

def _tri_bbox(tri, W, H):
    """
    Retorna bounding box inteiro (xmin,xmax,ymin,ymax), recortado ao domínio [0..W-1],[0..H-1].
    """
    raise NotImplementedError("Implemente: _tri_bbox")

def _amostra_bilinear(img_float, x, y):
    """
    Amostragem bilinear em (x,y) com clamp nas bordas.
    img_float: (H,W,3) float32 [0,1] — retorna vetor (3,).
    """
    raise NotImplementedError("Implemente: _amostra_bilinear")

def gera_frame(A, B, pA, pB, triangles, alfa, beta):
    """
    Gera um frame intermediário por morphing com warping por triângulos.
    - A,B: imagens (H,W,3) float32 em [0,1]
    - pA,pB: (N,2) pontos correspondentes
    - triangles: (M,3) índices de triângulos
    - alfa: controla geometria (0=A, 1=B)
    - beta:  controla mistura de cores (0=A, 1=B)
    Retorna (H,W,3) float32 em [0,1].
    """
    raise NotImplementedError("Implemente: gera_frame")
