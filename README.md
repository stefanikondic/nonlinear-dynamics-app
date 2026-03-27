# Nonlinear Dynamics App

Interactive web application for exploring **2D nonlinear dynamical systems** through phase portraits, trajectories, nullclines, and fixed-point analysis.

---

## ✨ What this app does

This tool allows you to define a dynamical system of the form:

[
\dot{x} = f(x, y), \quad \dot{y} = g(x, y)
]

and interactively explore its behavior.

### Features

* 📈 Phase portraits (vector fields and streamlines)
* 🔁 Trajectories from arbitrary initial conditions
* 📍 Automatic detection of fixed points
* 🔬 Linear stability analysis (Jacobian + eigenvalues)
* ➖ Nullclines visualization
* ⚙️ Automatic detection of parameters in equations
* 🎛️ Manual parameter input

---

## 🧠 Intended use

* learning nonlinear dynamics
* building intuition for differential equations
* quick qualitative analysis
* teaching demonstrations

---

## 🚀 Getting started (using Conda)

This project uses a **Conda environment** defined in `env.yml`.

---

### 1. Install Conda (if needed)

Download Miniconda (recommended):
https://docs.conda.io/en/latest/miniconda.html

---

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nonlinear-dynamics-app.git
cd nonlinear-dynamics-app
```

---

### 3. Create environment from file

```bash
conda env create -f env.yml
```

---

### 4. Activate environment

```bash
conda activate nonlinear-dynamics
```

*(Environment name depends on your `env.yml` — adjust if needed.)*

---

### 5. Run the app

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧪 Example systems

### Linear oscillator

```
dx/dt = y
dy/dt = -x
```

### Nonlinear system

```
dx/dt = y - y^3
dy/dt = -x - y^2
```

### With parameter

```
dx/dt = y
dy/dt = r - x^2
```

The app will automatically detect `r`.

---

## ⚠️ Notes

* Use Python-like syntax:

  * `x^2`, `sinx`, `e^x`, `sqrtx`
* Any symbol except `x`, `y` → parameter
* Some functions have limited domains (e.g. `sqrt(x)`)

---

## 🏗️ Project status

🚧 **Work in progress**

This repository is being made public while still under active development.

Planned features:

* parameter sweeps and bifurcation diagrams
* live updates (no "PLOT" button)
* improved numerical robustness
* UI/UX improvements

---

## 🤝 Contributing

Feedback, issues, and suggestions are welcome.

---

## 👤 Author

Stefani Tovilović
Medical physicist

---

## 💡 Philosophy

The goal is not only solving systems —
but **seeing them**.

This app is designed to build geometric and physical intuition
for nonlinear dynamics.
