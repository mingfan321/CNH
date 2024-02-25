# CNH

Code for TNNLS Paper: Conditional Neural Heuristic for Multi-objective Vehicle Routing Problems
It contains the implementation codes and testing dataset for three multi-objective vehicle routing problems:

- Bi-objective traveling salesman problem(BiTSP).

- Bi-objective capability vehicle routing problem(BiCVRP).

- Tri-objective traveling salesman problem(TriTSP).

This code is heavily based on the [POMO repository](https://github.com/yd-kwon/POMO) and [PMOCO repository](https://github.com/Xi-L/PMOCO).

#### Quick Start

- To train a model, such as BiTSP with 20 nodes, run train_motsp_n20.py in the corresponding folder.

- To test a model, such as BiTSP with 20 nodes, run test_motsp_n20.py in the corresponding folder.

- To test a model using CAS, such as BiTSP with 20 nodes, run test_active_search_CAS.py in the corresponding folder.

- Pretrained models for each problem can be found in the result folder.

- The testing dataset used in our paper can be found in the test_data folder.

#### Reference

If our work is helpful for your research, please cite our paper:

```

```
