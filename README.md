Welcome to the official repository for our AIES 2025 paper: **On the Misalignment Between Legal Notions and Statistical Metrics of Intersectional Fairness**.

## Abstract
This paper examines whether current statistical metrics for intersectional fairness truly capture the non-additive discrimination emphasized in legal and social theory. Using controlled synthetic data, we isolate single-axis, additive, and interaction-based biases. We find that existing metrics respond more to additive disparities than to uniquely intersectional effects, revealing a misalignment with the theoretical foundations of intersectionality. We call for fairness metrics that explicitly account for structural, non-additive harms.

## 🗂️ Repository Structure

- `main.py` — Generate synthetic data for all scenarios and seeds
- `config/` — YAML configuration files for each discrimination scenario  
- `src/` — Core source code for experiments and analyses
  - `data_generator.py` — Synthetic data generation logic
  - `metrics.py` — Fairness metric implementations
  - `model_fitting.py` — Classifier training and evaluation
  - `run_analysis.py` — Main analysis pipeline
  - `helper.py` — Utility functions and data processing
  - `plots.py` — Visualization functions
- `Notebooks/` — Interactive Jupyter notebooks for reproducing results
  - `fairness_analyis.ipynb` — Main fairness analysis across multiple seeds
  - `exploratory.ipynb` — Single-seed data exploration
- `generated_data/` — Generated synthetic datasets (created by main.py)
- `images/` — Generated plots and figures

## 🚀 Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/KANUBALAD/On-the-Misalignment-Between-Legal-Notions-and-Statistical-Metrics-of-Intersectional-Fairness.git
    cd On-the-Misalignment-Between-Legal-Notions-and-Statistical-Metrics-of-Intersectional-Fairness
    ```

2. **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pyyaml jupyter
    ```

3. **Generate synthetic data:**
    ```bash
    python main.py
    ```

4. **Run fairness analysis:**
    ```bash
    jupyter notebook Notebooks/fairness_analyis.ipynb
    ```

5. **Explore individual scenarios:**
    ```bash
    jupyter notebook Notebooks/exploratory.ipynb
    ```


## 📊 Key Experiments and Results

The repository implements five key discrimination scenarios:

1. **No Bias**: Baseline with no discrimination
2. **Single-Axis Bias**: Discrimination against one protected attribute (e.g., gender only)
3. **Multiple Bias**: Additive discrimination against multiple attributes
4. **Intersectional Bias**: Non-additive interaction effects between attributes
5. **Compounded Bias**: Complex combination of all discrimination types

### Implemented Fairness Metrics

- **Demographic Parity Ratio (DPR)**
- **ε-lift (elift)**: Multiplicative fairness metric
- **s-lift (slift)**: Additive fairness metric  
- **Subgroup Unfairness**: Maximum disparity across intersectional subgroups


## 📚 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{kanubala2025intersectional,
  title={On the Misalignment Between Legal Notions and Statistical Metrics of Intersectional Fairness},
  author={Kanubala, Deborah D. and Valera, Isabel},
  booktitle={Proceedings of the 2025 AAAI/ACM Conference on AI, Ethics, and Society (AIES)},
  year={2025},
  organization={ACM}
}
```


## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Report bugs or issues
2. Suggest improvements
3. Submit pull requests
4. Ask questions about the implementation

## 📞 Contact

For questions, collaborations, or support:

- **Deborah D. Kanubala**: [kanubala@cs.uni-saarland.de](mailto:kanubala@cs.uni-saarland.de)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

