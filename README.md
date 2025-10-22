# Trabalho Prático — Warping e Morphing de Imagens (INF394 - Processamento Digital de Imagens)

Este repositório contém o **código base** para o primeiro trabalho prático da disciplina **Processamento Digital de Imagens (INF394)**, referente ao tema de **transformações geométricas**.  
O objetivo é implementar, testar e visualizar um algoritmo de *image morphing* utilizando *triangulação de Delaunay* e interpolação ponto a ponto entre duas imagens.

---

## Estrutura do Projeto

```bash
MorphingGUI/
├── input # Pasta com exemplos de arquivos de entrada de dados
├── lib # Módulos Python personalizados
│   ├── __init__.py # Arquivo auxiliar
│   ├── check_points.py # Módulo auxiliar de checagem de pontos
│   ├── morph.py # Arquivo auxiliar
│   ├── morph_core.py # Funções fornecidas prontas (infraestrutura)
│   └── morph_student.py # Funções a serem implementadas pelos alunos
├── morphingGui.py # Arquivo principal do aplicativo (GUI em Tkinter)
├── output # Pasta para arquivos de saída
└── requirements.txt # Requisitos Python para rodar o projeto
```


---

## O que deve ser implementado pelos estudantes

Apenas o arquivo `lib/morph_student.py` deve ser modificado.  
Todas as funções já possuem **assinaturas e docstrings** explicando o que se espera.  
As demais partes (interface, carregamento de imagens, geração de animações, etc.) já estão prontas.

Ao executar o aplicativo (`python morphingGui.py`), pode-se:

- Carregar duas imagens.
- Anotar pontos correspondentes entre elas.
- Visualizar e checar a consistência dos pontos.
- Gerar animações de *morphing*, com ou sem exibição da malha de triangulação.

---

## Requisitos de Software

- **Python 3.9 ou superior**
- Sistema operacional compatível com Tkinter (Linux, macOS, Windows)

Os pacotes necessários estão listados em `requirements.txt`.

---

## Instalação e Configuração do Ambiente

### 1. Clonar o repositório

```bash
git clone https://github.com/mhfribeiroufv/MorphingGUI.git
cd MorphingGUI
```

### 2. Criar ambiente virtual (recomendado)

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependências

Antes de instalar, garanta que o pip está atualizado:

```bash
python -m pip install --upgrade pip
```

Em seguida, instale os pacotes necessários:

```bash
pip install -r requirements.txt
```

### 4. Execução do aplicativo

Depois de ativar o ambiente virtual e instalar as dependências, execute:

```bash
python morphingGui.py
```

O aplicativo abrirá uma interface gráfica com três abas:

1. Anotação de Pontos — marque pontos correspondentes entre as imagens.
2. Morphing — defina parâmetros de interpolação e gere animações.
3. Remoção de Plano de Fundo — ferramenta auxiliar para preparar imagens autorais.

---

## Créditos e Licença

Desenvolvido por **Prof. Marcos H. F. Ribeiro**
Departamento de Informática — Universidade Federal de Viçosa (UFV)
Distribuído apenas para fins didáticos no contexto da disciplina INF394 – Processamento Digital de Imagens.
