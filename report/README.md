# Report

- Template: official **RLC 2026** LaTeX template (from the RLC submission instructions), **cover page disabled**. Keep fonts, margins, spacing.
- Limit: **8 pages main content** (figures, tables and pre-reference appendices included). References and post-reference supplementary material excluded.
- Write in Overleaf, export `main.tex` + `figures/` here before submission.

Suggested structure and owners:

| Section | Owner | ~pages |
|---|---|---|
| Abstract, Introduction, research question | M1 | 0.75 |
| Problem formulation (MDP: state/history, actions, reward, termination), adapter | M1 | 1 |
| Background: DQN / PPO (+ related ARC-AGI-3 work) | M2 / M3 | 1 |
| Experimental setup (games, split, budgets, seeds, hardware) + shared action-space method | M1 | 1 |
| DQN: baseline → imp1 → imp2 | M2 | 1.5 |
| PPO: baseline → imp1 → imp2 | M3 | 1.5 |
| DQN vs PPO comparison, ablations, held-out, failure cases | M1 | 1 |
| Limitations & conclusion | all | 0.5 |

Quality reference given in the brief: *Less is More: Recursive Reasoning with Tiny Networks*.

Drafts already written: `sections/01_introduction.tex`, `02_problem.tex`, `04_setup.tex`, `07_comparison.tex` (Member 1), `references.bib`. Define `\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}` in the preamble.
