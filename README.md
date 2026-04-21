# Interactive HPO

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)

![test](./demo.png)

---

## Quick start

Python 3.11+ is required.

---

### Option 1: Install from GitHub

Installs the `ihpo` command and all dependencies.

```bash
pip install git+https://github.com/angrinord/InteractiveHPO.git
ihpo
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

### Option 2: Run from source

First, get the code:

```bash
git clone https://github.com/angrinord/InteractiveHPO.git
cd InteractiveHPO
```

Then install dependencies and launch using whichever environment manager you prefer:

**pip**
```bash
# Recommended to use venv
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run run.py
```

**conda**
```bash
conda create -n ihpo python=3.12
conda activate ihpo
pip install -r requirements.txt
streamlit run run.py
```

> **Note:** `smac` and `hypershap` are not available on conda-forge, so the `pip install` step is required even when using a conda environment.

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

### Option 3: Docker

```bash
docker build -t interactivehpo .
docker run -p 8501:8501 interactivehpo
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

///////////////////////////////////////////////////

---

///////////////////////////////////////////////////

### Adding a model

///////////////////////////////////////////////////

---

## Design decisions

///////////////////////////////////////////////////
