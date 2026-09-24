Then please generate a README for me according to the interpretation and my clarification, and the Overview of AGENTS.md. Limit the scope to 

---

Read the pyproject.toml.

---

Overview of AGENTS.md:

```
Cocoonet evaluates machine learning models using an explicitly configured root-local worker, independent heterogeneous
online workers, or both. The root (user) node always owns coordination; it does not automatically act as a worker.
It prepares dataset access, supplies models and evaluation directives, and receives serialized reports. Online
workers use synchronized datasets. Scientific auditability, transparency, and reproducibility are core requirements.
```

---

My clarifications:

```
1. The major purpose of the project is to establish a powerful machine learning evaluation pipeline for the scientific statistical evaluation and optimization of various model structures, hyperparameter combinations and data processing techniques.
2. Different from the current modeling optimization pipelines, this pipeline facilitates statistical evaluation, instead of optimization based on one metric with pruning, which is the major approach in model optimization currently.
3. The `CocooNet` project / module acts as the basic part of this purpose, i.e. generating evaluation reports for single models, so that the reports can be later summarized and analyzed across the model structures, hyperparameter combinations and data processing techniques for a statistical significant scientific optimization.
4. The project support classic machine learning models supported by multiprocessor CPU computation as well as neural networks supported by GPU computation.
5. The size of the models in this project is expected to be small to moderate, not large.
6. However, factorial analysis of various model structures, hyperparameter combinations and data processing techniques need significant computation power, so that this project / module take advantage of distributed computational power (workers) with low communication requirements for this task.
7. Model is independently evaluated by workers and evaluation reports are collected (by this module) for further analysis (not the scope of `CocooNet`).
8. Users are expected to be able to add their own workers, as well as using public workers, shared by others.
9. This is meaningful because this type of optimization work exists for industrial needs, instead of large models in the field of computer science or AI industry, because specific application industries like biology and engineering often use small to moderate machine learning models instead of the current popular large models, for performance, stability, determinism, interpretability, auditability, and limited sample size in industrial data for the modeling.
10. This is also meaningful for this type of tasks requiring very limited communication between workers, which allow a cost-efficient construction of the computation worker clusters.
11. This is not expected to be designed for consumer grade products, but expected to be designed for scientific researching purposes including both academic researches and application researches.
12. `CocooNet` is one of the infrastructure for this purpose, and one of the most important infrustructures.
```
