> This repository contains my solution for the ML4SCI (QMLHEP) GSoC 2026 evaluation task.

# 🧠 Agentic Quantum Neural Network Pipeline for Closed-Loop Circuit Optimization — ML4SCI GSoC 2026 (QMLHEP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zvQnLKAgY-2C_t4qwyLI9Y_jKEw8YhH2?usp=sharing)

> **GSoC 2026 Evaluation Task | ML4SCI Organization | QMLHEP Track**  
> A closed-loop, agent-driven pipeline for quantum circuit design, training, and optimization of quantum neural networks.

---

## Overview

This repository implements an **agentic quantum neural network (QNN) pipeline** as part of the GSoC 2026 evaluation under the [ML4SCI](https://ml4sci.org/) organization's **QMLHEP** (Quantum Machine Learning for High Energy Physics) track.

An orchestrator-agent architecture inspired by Orchestral AI, where a central controller coordinates specialized tools to explore Hilbert spaces, train parameterized quantum circuits, evaluate performance, and iteratively refine hyperparameters. This architecture is a direct proof-of-concept for **closed-loop LLM-guided quantum circuit design**, where an intelligent agent autonomously drives the full quantum ML lifecycle without manual intervention.

This approach reduces manual experimentation in quantum circuit design and demonstrates how intelligent agents can efficiently navigate complex quantum search spaces.

---

## Features

- **Tool-based Agent Architecture** — Modular tools defined via `@define_tool` decorators, enabling composable and extensible quantum workflows.
- **Hilbert Space Exploration** — Systematic search over quantum embedding strategies and ansatz configurations.
- **QNN Training Module** — PennyLane variational circuits integrated with PyTorch autograd for hybrid quantum-classical optimization.
- **Hyperparameter Optimization Loop** — Agentic closed-loop search over circuit depth, learning rate, and encoding strategy.
- **Binary MNIST Classification** — End-to-end evaluation on a 2-class MNIST subset (digits 0 vs. 1).
- **Closed-Loop Evaluation** — `generate → train → evaluate → improve` feedback cycle with no manual tuning between iterations.

---

## Architecture / Workflow

```
┌─────────────────────────────────────────┐
│              Orchestrator Agent         │
│  (dispatches tools, tracks state)       │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌──────────────┐
│Hilbert │  │  QNN   │  │Hyperparameter│
│ Space  │  │Training│  │ Optimization │
│Explorer│  │ Module │  │    Loop      │
└────────┘  └────────┘  └──────────────┘
    │            │            │
    └────────────┴────────────┘
                 │
         ┌───────────────┐
         │  Evaluation & │
         │    Feedback   │
         └───────────────┘
```

**Execution flow:**
1. **Hilbert Space Exploration** — Agent selects quantum feature maps and ansatz topologies.
2. **QNN Training** — Parameterized quantum circuit trained via gradient descent (Adam optimizer).
3. **Evaluation** — Accuracy and loss computed on held-out test set.
4. **Refinement** — Agent updates hyperparameters based on evaluation signal and re-runs the loop.

---

## Results

| Metric              | Value         |
|---------------------|---------------|
| Task                | Binary MNIST (0 vs. 1) |
| Best Test Accuracy  | **~74%**      |
| Optimizer           | Adam          |
| Circuit Type        | Variational QNN (PennyLane) |
| Backend             | `default.qubit` simulator |
| Training Paradigm   | Hybrid Quantum-Classical |

> The pipeline autonomously converges to competitive accuracy without manual hyperparameter tuning, demonstrating the viability of agentic control over quantum ML workflows.

---

## How to Run

### Prerequisites

```bash
pip install pennylane torch torchvision numpy matplotlib
```

### Quick Run (Colab)

Click the **Open in Colab** badge at the top and run all cells — no setup required.

### Local Run

Open `QMLHEP_GSoC_Evaluation.ipynb` in Jupyter and run all cells sequentially.

```bash
jupyter notebook QMLHEP_GSoC_Evaluation.ipynb
```

> **Note:** GPU is not required. The PennyLane `default.qubit` device runs efficiently on CPU for the binary MNIST experiments included here.

### Key Entry Points

| Component                    | Description                              |
|-----------------------------|------------------------------------------|
| `@define_tool`              | Decorator used to register agent tools   |
| `hilbert_space_explorer()`  | Tool for circuit architecture search     |
| `train_qnn()`               | Hybrid training loop (PennyLane + PyTorch) |
| `optimize_hyperparams()`    | Agentic hyperparameter refinement loop   |

---

## Tech Stack

| Layer               | Technology                        |
|---------------------|-----------------------------------|
| Quantum Framework   | [PennyLane](https://pennylane.ai) |
| Classical ML        | [PyTorch](https://pytorch.org)    |
| Agent Architecture  | Custom Orchestral-style dispatcher |
| Dataset             | Binary MNIST (via `torchvision`)  |
| Environment         | Google Colab / Jupyter Notebook   |
| Language            | Python 3.10+                      |

---

## Notebook

📓 **[View Notebook on Google Colab](https://colab.research.google.com/drive/1zvQnLKAgY-2C_t4qwyLI9Y_jKEw8YhH2?usp=sharing)**

The notebook contains all experiments, training curves, circuit diagrams, and reproducible results for the GSoC 2026 QMLHEP evaluation task.

---

## Connection to LLM-Guided Quantum Circuit Design

This implementation demonstrates a **closed-loop, agentic control paradigm** for quantum circuit optimization — a foundational step toward fully **LLM-guided quantum circuit design**, where a language model acts as the orchestrator, reasoning over circuit performance metrics to autonomously propose, evaluate, and refine quantum architectures in a continuous feedback loop.

---

## Author

**Kunal Sanga**  
GSoC 2026 Applicant | ML4SCI — QMLHEP Track  

---

## License

This project is submitted as part of the GSoC 2026 evaluation process for ML4SCI. All code is original work by the author.
