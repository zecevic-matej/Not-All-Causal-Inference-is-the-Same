### Not All Causal Inference is the Same [Accepted to TMLR 2023]

Official code repository for reproducing the empirical section of the paper: "7.1 ‘Bonus:’ An Easy Solution to Speeding Up Mechanism Inference in SCM".

![Thumbnail of Figure 3 from Paper](media/thumbnail.png)

---

**Code Structure:**

* `aux` contains helper functions
* `models` contains base functions like neural nets, sum-product networks but also the actual (T)NCM
* `expX.py` reproduces an experiment as found in the paper (e.g. `exp1.py` reproduces the results from Figure 6 of the main paper)