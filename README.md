

# On the Misalignment Between Legal Notions and Statistical Metrics of Intersectional Fairness

Welcome to the official repository for our AIES 2025 paper: **On the Misalignment Between Legal Notions and Statistical Metrics of Intersectional Fairness**.

## Abstract

Intersectional (un)fairness, as conceptualized in legal and social theory, emphasizes the non-additive and structurally complex nature of discrimination against individuals at the intersection of multiple sensitive attributes (such as race, gender, etc). Recent works have proposed statistical metrics for intersectional fairness by estimating disparities across groups of individuals sharing two or more sensitive attributes. However, it is unclear if these metrics detect uniquely intersectional discrimination. We therefore pose the following question, Do current statistical intersectional metrics detect the non-additive discrimination highlighted by intersectionality theory? More specifically, to answer this, we run controlled synthetic data experiments that explicitly allow us to control for single, multiple, intersectional, and compounded forms of discrimination. Our analyses show that current statistical metrics for intersectional fairness behave more like multi-attribute disparity measures. Specifically, they respond more strongly to additive or compounded biases than to non-additive interaction effects. While they effectively capture disparities across multiple sensitive attributes, they often fail to detect uniquely intersectional discrimination. These findings reveal a fundamental misalignment between existing intersectional fairness metrics and the legal and theoretical foundations of intersectionality. We argue that if intersectional fairness metrics are to be deemed truly intersectional, they must be explicitly designed to account for the structural, non-additive nature of intersectional discrimination.

## Contents

- `src/` — Source code for experiments and analyses
- `notebooks/` — Interactive Jupyter notebooks for reproducing results
- `results/` — Outputs and figures from our experiments

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/KANUBALAD/On-the-Misalignment-Between-Legal-Notions-and-Statistical-Metrics-of-Intersectional-Fairness.git
    cd On-the-Misalignment-Between-Legal-Notions-and-Statistical-Metrics-of-Intersectional-Fairness
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run experiments:**
    See instructions in the `notebooks/` directory for step-by-step guides.

## Citation

If you use this code or data in your research, please cite our paper:

```
@inproceedings{kanubala2025aies,
  title={On the Misalignment Between Legal Notions and Statistical Metrics of Intersectional Fairness},
  author={Deborah D. Kanubala, Isabel Valera},
  booktitle={AIES 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Contact
For questions or collaborations, please contact: [kanubala@cs.uni-saarland.de](mailto:kanubala@cs.uni-saarland.de).

