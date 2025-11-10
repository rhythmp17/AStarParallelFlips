

## 🧭 **README.md**
# Flip Distance Solver using A* Search with Parallel Flips

This project implements an **A\* search algorithm** to compute the **flip distance** between two triangulations of the same point set.  
It supports **parallel flips** — where multiple non-conflicting edges (no shared triangles) can be flipped simultaneously.

The code is completely **self-contained**, requiring no external geometry libraries.  
It works directly on the triangulation edge data provided in the **CGSHOP2026 JSON format**.

---

## 🚀 Features
- **Pure Python** — no external geometry libraries.
- **Accurate flip logic** — identifies flippable edges as diagonals of quadrilaterals.
- **Parallel flip capability** — flips up to `k` non-conflicting edges per step.
- **A\* search** with a simple heuristic: number of edges not present in the target triangulation.
- **Modular** — clean separation of input parsing (`inputparser.py`) and search (`a_star_parallel.py`).

---

## 📂 Project Structure
```

BTP/
├── venv/                     # Virtual environment (created locally)
├── inputparser.py            # JSON instance parser
├── a_star_parallel.py        # A* implementation for flip distance
├── instance.json             # Example 5-point instance file
├── README.md                 # This file
└── requirements.txt          # Optional, if additional dependencies are added later

````

## 🧰 Environment Setup

### 1️⃣ Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # On Linux/macOS
# venv\Scripts\activate       # On Windows
````

### 2️⃣ Upgrade `pip`

```bash
pip install --upgrade pip
```

### 3️⃣ Install dependencies

This implementation is pure Python, so there are **no external package requirements**.
If you plan to visualize triangulations or later use `cgshop2026_pyutils`, you can optionally install:

```bash
pip install matplotlib cgshop2026-pyutils
```

---

## 📦 Input Format

Input files must follow the **CGSHOP2026 instance schema**, containing:

* `points_x` and `points_y`: coordinate lists
* `triangulations`: list of triangulations (each is a list of edges)

Example (`instance.json`):

```json
{
  "content_type": "CGSHOP2026_Instance",
  "instance_uid": "demo_5_points_flip",
  "points_x": [0, 2, 4, 3, 1],
  "points_y": [0, 0, 1, 3, 3],
  "triangulations": [
    [
      [0,1], [1,2], [2,3], [3,4], [4,0],
      [0,3], [0,2]
    ],
    [
      [0,1], [1,2], [2,3], [3,4], [4,0],
      [2,4], [0,2]
    ]
  ]
}
```

This instance requires just **one flip** (edge `(0,3)` → `(2,4)`) to convert triangulation 0 to triangulation 1.

---

## ⚙️ Running the Solver

After activating your virtual environment, run:

```bash
python3 a_star_parallel.py instance.json --start 0 --target 1 -k 1
```

### Command-line Arguments:

| Flag       | Description                              | Default |
| ---------- | ---------------------------------------- | ------- |
| `instance` | Path to the input JSON file              | —       |
| `--start`  | Index of the starting triangulation      | `0`     |
| `--target` | Index of the target triangulation        | `1`     |
| `-k`       | Maximum number of parallel flips allowed | `1`     |

### Example Output:

```
✅ Flip distance (k=1): 1
Path length: 2
```

---

## 🧑‍💻 Example Debug Run

To see debug logs (by uncommenting lines inside `run()`):

```bash
# Inside run()
print(f"Expanding {cur}, flippable = {flippable}")
```

Then run:

```bash
python3 a_star_parallel.py instance.json --start 0 --target 1 -k 1
```

---
