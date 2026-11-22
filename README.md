# AISR Laboratory Report — Experiments 1 to 8

**CSAI4014 · Algorithms for Intelligent Systems and Robotics**
University of Petroleum and Energy Studies (UPES), Dehradun · Autumn Semester 2026–27

| | |
|---|---|
| **Student** | Himanshu Pokhriyal |
| **Roll No.** | 500121200 |
| **Batch** | B.Tech CSE (AI & ML), Semester VII |
| **Course Teacher** | Dr. Resham Raj Shivwanshi |

## 📄 Main document

**[AISR_Lab_Report_500121200.pdf](AISR_Lab_Report_500121200.pdf)** is the complete submitted report (43 pages).

It contains the title page, declaration, contents, list of figures and list of tables. Each of the eight experiments starts on a new page and has the eleven sub-headings from Aim to Conclusion. The report ends with the IEEE references and Appendices A–C. Each experiment's Questions section includes the scanned handwritten answers for that experiment.

The LaTeX source is [`report_final.tex`](report_final.tex).

## Experiments

| # | Experiment | Folder | Run |
|---|---|---|---|
| 1 | Time and frequency domain, and the pendulum | [`exp01-signals`](exp01-signals) | `python exp01-signals/exp01.py` |
| 2 | The eight queens problem | [`exp02-queens`](exp02-queens) | `python exp02-queens/exp02.py` |
| 3 | Breadth-first and depth-first search | [`exp03-search`](exp03-search) | `python exp03-search/exp03.py` |
| 4 | Hill climbing | [`exp04-hillclimb`](exp04-hillclimb) | `python exp04-hillclimb/exp04.py` |
| 5 | Reinforcement learning (Q-learning) | [`exp05-rl`](exp05-rl) | `python exp05-rl/exp05.py` |
| 6 | A simple neural network (XOR) | [`exp06-neural`](exp06-neural) | `python exp06-neural/exp06.py` |
| 7 | Webots: reactive swarm and lidar navigation | [`exp07-webots`](exp07-webots) | see [`exp07-webots/code/README.md`](exp07-webots/code/README.md) |
| 8 | Cross-embodiment transfer in PyBullet | [`exp08-xembody`](exp08-xembody) | see [`exp08-xembody/code/README.md`](exp08-xembody/code/README.md) |

Each experiment folder has its script and a `figs/` folder (Experiment 8 uses `code/results/`) holding the figures used in the report.

## Requirements

- Experiments 1–6: Python 3.10+ with NumPy, SciPy and Matplotlib
- Experiment 7: Webots R2025a and Python 3
- Experiment 8: Python 3.10/3.11 with PyBullet, NumPy, Matplotlib and scikit-learn (see `exp08-xembody/code/requirements.txt`)

## Repository layout

```
aisr-lab-500121200/
├── AISR_Lab_Report_500121200.pdf   # main report (submission)
├── report_final.tex                # LaTeX source of the report
├── handwritten_p1-4.png            # full scanned answer sheets (Appendix C)
├── handwritten_crops/              # answers cropped per experiment (exp01–exp08)
├── exp01-signals/ … exp06-neural/  # script + figs/ for each experiment
├── exp07-webots/                   # Webots worlds, controllers, analysis
└── exp08-xembody/                  # PyBullet cross-embodiment code and results
```

## Rebuilding the PDF

```bash
xelatex report_final.tex   # run three times to update contents, figures and tables lists
```
