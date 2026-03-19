# 🤖 Generative AI Usage Documentation — WorkPulse

## Overview

This document describes how Generative AI tools were used throughout the WorkPulse capstone project, in compliance with Step 9 requirements.

## Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| **Claude (Anthropic)** | Claude Opus | Code generation, architecture design, documentation |

## Usage by Project Step

### Step 1: Problem Framing
- **Used for:** Refining the problem statement, identifying relevant success metrics, and structuring the business context narrative.
- **Human contribution:** Domain selection, business KPI definition, final framing decisions.

### Step 2: Data Collection
- **Used for:** Generating self-contained data loading code, creating the data dictionary template, and writing dataset profiling summaries.
- **Human contribution:** Dataset selection, source verification, quality assessment decisions.

### Step 3: EDA & Feature Engineering
- **Used for:** Writing preprocessing pipeline code (null handling, outlier treatment, encoding), generating EDA visualisation code, and engineering domain-specific features.
- **Human contribution:** Domain knowledge for feature design rationale, interpreting EDA findings, threshold decisions.

### Step 4: Model Implementation
- **Used for:** Implementing the 11-model training pipeline, hyperparameter search configuration, evaluation framework, and model comparison infrastructure.
- **Human contribution:** Model selection strategy, interpreting results, final model choice validation in Colab.

### Step 5: Ethical AI & Bias Auditing
- **Used for:** SHAP/LIME/PDP implementation code, fairness metrics computation, mitigation strategy code, and limitations analysis structure.
- **Human contribution:** Ethical framing, fairness threshold decisions, mitigation strategy selection.

### Step 6: Presentations
- **Used for:** Generating both presentation decks (PowerPoint + LaTeX Beamer), slide content, and visual design.
- **Human contribution:** Narrative flow, audience-appropriate messaging, final review.

### Step 7: GitHub Repository
- **Used for:** Repository structure, README, requirements.txt, src/ scripts, CI configuration, and documentation.
- **Human contribution:** Repository naming, final review, git operations.

## Key Principles

1. **Human-in-the-loop:** All AI-generated code was reviewed, tested in Google Colab, and validated by the author before inclusion.
2. **Transparency:** This document explicitly lists all AI usage — nothing is hidden.
3. **Critical thinking:** AI suggestions were evaluated critically; several were modified or rejected (e.g., the scoring formula was redesigned after identifying a normalisation flaw).
4. **Reproducibility:** All code is deterministic (fixed random seeds) and self-contained.

## Examples of AI-Assisted Outputs

### Auto-Generated EDA Summary (Step 3)
The AI generated initial EDA code blocks including univariate distributions, correlation heatmaps, and bivariate analyses. The author then interpreted the outputs and added domain-specific commentary.

### Data Dictionary Generation (Step 2)
The AI produced the initial data dictionary structure from dataset schemas. The author verified all entries against source documentation and added domain justifications.

### Code Review & Bug Fixing
The AI identified and fixed several issues during development:
- `PartialDependenceDisplay` reshape bug with binary features (replaced with manual PDP computation)
- Model selection scoring formula flaw (replaced min-max with rank-based scoring)
- Directory creation order for Colab compatibility
