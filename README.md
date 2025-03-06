# SANE
This is the notebook for Strategic(S) Autonomous(A) Non-Smooth(S) Exploration(E)


we develop a *Strategic Autonomous Non-Smooth Exploration (SANE)* framework, which demonstrates advancements in exploring multidimensional parameter spaces with multi-modal and non-differentiable black box functions for materials and physics discovery. Traditional BO methods, while powerful, often lead to the risk of over-focusing on singular optimal conditions and the potential for becoming trapped in noisy/fake regions or local optima. SANE integrates a cost-driven probabilistic acquisition function with a probablislic gate constraint via minor human intervention, a more robust and exploratory approach, to address these limitations in autonomous experimentation. SANE actively seeks out multiple global and local optima, ensuring a more comprehensive exploration of the parameter space. The application of the SANE framework in two complex material systems, i.e., Sm-doped BiFeO3 combinatorial library and PbTiO3 ferroelastic/ferroelectric thin films, has demonstrated its efficacy. The full results are provided in Analysis folder. 

![image](https://github.com/user-attachments/assets/092314a2-8203-4cc9-8d7b-2ec1bdb88fea)


## Examples

*   Multi-modal Synthetic function: https://github.com/arpanbiswas52/SANE/blob/main/Analysis/SANE_synthetic2D(Notebook_Github).ipynb

*    Multi-modal Synthetic function with added noise: [https://github.com/arpanbiswas52/SANE/blob/main/Analysis/SANE_synthetic2D(Notebook_Github).ipynb
](https://github.com/arpanbiswas52/SANE/blob/main/Analysis/SANE_noisysynthetic2D_(Notebook_Github).ipynb)

*   Sm-doped BiFeO3 combinatorial library: https://github.com/arpanbiswas52/SANE/blob/main/Analysis/SANE_Sm_BFO(Notebook).ipynb

*   PbTiO3 ferroelastic/ferroelectric thin films: https://github.com/arpanbiswas52/SANE/blob/main/Analysis/SANE_BEPS(Notebook_Github).ipynb

## Data Availability

*   Sm-doped BiFeO3 combinatorial library [1]:
  1. https://drive.google.com/uc?id=1rKE6d4T-JHlNOT0K4HExOM83PTHPdy-2
  2. https://drive.google.com/uc?id=1C1UUH_ZEfDxrGtZgqumfyprNu7HGTjab

*   PbTiO3 ferroelastic/ferroelectric thin films [2]:
  1. https://drive.google.com/uc?id=1D5P4pxGEyk_R09k6Iwx8pTCC-z5T2SfX

## Installation

#### Requirements
Please see the list here: https://github.com/arpanbiswas52/SANE/blob/main/requirement.txt

## Demonstration. Full workflows are provided in the notebooks

* Initilize training data via Latin Hypercube Sampling. 
```
n = 30
lb = 0
ub = 4225
bounds = [lb, ub]
#np.random.seed(1000)
sampler = scipy.stats.qmc.LatinHypercube(d=1, scramble=True, seed=1)
train_idx = bounds[0] + (bounds[1] - bounds[0])  * sampler.random(n=n)
#train_x = bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(n, 2)
train_idx = np.round(train_idx)
train_idx = train_idx.astype(int)
train_x = np.zeros((len(train_idx), X.shape[-1]))
train_y = np.zeros((len(train_x)))
for i in range(len(train_x)):
    train_x[i, :] = X[train_idx[i], :]
    train_y[i] = y[train_idx[i]]

train_x.shape, train_y
```
* User assessment of initial experimental data.
```
idx_good, idx_bad = votes_initialdata(args)
idx_good, idx_bad
```
* Converting votes to constraint evalution metric
```
y_vote = np.zeros(len(y_measured))
for i in range(0, len(y_vote)):
    y_good = 0
    y_bad = 0
    for j1 in range(0, len(idx_good)):
      y_good = y_good + np.absolute(indices_measured[i, 0] -  indices_all[idx_good[j1], 0]) + np.absolute(indices_measured[i, 1] -  indices_all[idx_good[j1], 1])
    y_good = y_good/len(idx_good)
    #print(y_good)

    for j2 in range(0, len(idx_bad)):
      y_bad = y_bad + np.absolute(indices_measured[i, 0] -  indices_all[idx_bad[j2], 0]) + np.absolute(indices_measured[i, 1] -  indices_all[idx_bad[j2], 1])
    y_bad = y_bad/len(idx_bad)
    #print(y_bad)

    y_vote[i] = y_bad - y_good # Turning into maximization problem
```
* Setting the switching parameter between strategic exploitation and exploration
```
k = np.zeros(100)
k[60:] = k[60:] + 1
plt.plot(k)
```

* Checking criteria wheather we find a potential region of interest to exploit-- can be a local or global optima
```
if np.isin(exp_x, int(np.argmax(y_measured))).any()== True: #New potential region of interest found with low function values

    y_train_norm = (y_measured-np.min(y_measured))/(np.max(y_measured)-np.min(y_measured))
    best_y_norm = np.max(y_train_norm)
    # Need to provide normalized value of X and Y
    print("Checking for ROI with potntial local optima")
    exp_x = check_ROI(indices_measured_norm, y_train_norm, best_x_norm, best_y_norm, exp_x)
    best_x = indices_measured[exp_x[-1]]
    best_x_norm = indices_measured_norm[exp_x[-1]]
else:
    exp_x = np.hstack((exp_x, int(np.argmax(y_measured)))) #New potential ROI found with higher function values
    best_x = indices_measured[exp_x[-1]]
    best_x_norm = indices_measured_norm[exp_x[-1]]
    print("New ROI with potential global optima found")
```
* Run prediction model for constraint and function evaluation
```
cGP, constraint_mean, _= run_constraindkl(X_measured, y_vote, X_unmeasured)

gpbo, obj, y_pred, y_var = run_statdKL(X_measured, y_measured, indices_measured, test_data, constraint = constraint_mean, pen=1000, cost_params= cost_params)
```

* Approximated Constraint evaluation of new sample with no further human intervention
```
# Update distance metric for new X
for j1 in range(0, len(idx_good)):
  y_good = y_good + np.absolute(indices_measured[-1, 0] -  indices_all[idx_good[j1], 0]) + np.absolute(indices_measured[-1, 1] -  indices_all[idx_good[j1], 1])
y_good = y_good/len(idx_good)
for j2 in range(0, len(idx_bad)):
  y_bad = y_bad + np.absolute(indices_measured[-1, 0] -  indices_all[idx_bad[j2], 0]) + np.absolute(indices_measured[-1, 1] -  indices_all[idx_bad[j2], 1])
y_bad = y_bad/len(idx_bad)

y_vote = np.hstack((y_vote, (y_bad - y_good))) # Maximizing the target constraint
```

### Support

The authors acknowledge the use of facilities and instrumentation at the UT Knoxville Institute for Advanced Materials and Manufacturing (IAMM) and the Shull Wollan Center (SWC) supported in part by the National Science Foundation Materials Research Science and Engineering Center program through the UT Knoxville Center for Advanced Materials and Manufacturing [DMR-2309083](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2309083&HistoricalAwards=false). AFM measurements were performed at the Center for Nanophase Materials Sciences (CNMS), which is a US Department of Energy, Office of Science User Facility at ORNL.

<img width="400px" src="https://mrsec.org/sites/default/files/MRSEC%20logo_clear%20background.png">


## References
[1] Fujino, S.; Murakami, M.; Anbusathaiah, V.; Lim, S.-H.; Nagarajan, V.; Fennie, C. J.; Wuttig, M.; Salamanca-Riba, L.; Takeuchi, I. Combinatorial Discovery of a Lead-Free Morphotropic Phase Boundary in a Thin-Film Piezoelectric Perovskite. Appl. Phys. Lett. 2008, 92 (20), 202904. https://doi.org/10.1063/1.2931706.

[2] Pratiush, U.; Funakubo, H.; Vasudevan, R.; Kalinin, S. V.; Liu, Y. Scientific Exploration with Expert Knowledge (SEEK) in Autonomous Scanning Probe Microscopy with Active Learning. arXiv August 4, 2024. https://doi.org/10.48550/arXiv.2408.02071

# Citation
Please cite our paper if you use the model for your experiments. Please refer to *arpanbiswas52@gmail.com* for any question, bugs and feedbacks

Arpan Biswas, Rama Vasudevan, Rohit Pant, Ichiro Takeuchi, Hiroshi Funakubo, Yongtao Liu. SANE: Strategic Autonomous Non-Smooth Exploration for Multiple Optima Discovery in Multi-modal and Non-differentiable Black-box Functions. Digital Discovery, 2025, Advance Article, DOI: 10.1039/D4DD00299G. 
