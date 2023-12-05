# GLP1 Project

## Overview
Glucagon-like peptide-1 (GLP-1) receptor agonists, widely prescribed for weight loss, offer a non-invasive solution for obesity. 
Social media users who discuss GLP-1 drugs share their experiences, including adverse side effects (ASE). 
%Recent studies highlight unofficial GLP-1 ASE, such as suicidal thoughts. 
Exploring the connection between co-occurring ASEs is crucial due to individual differences often missed in small-scale experiments.
We collected 11,700 posts from $\mathbb{X}$, 489,529 Reddit posts, and 14,502 PubMed manuscripts related to GLP-1 receptor agonists.
Utilizing a Natural Language Processing (NLP) technique of named entity recognition (NER), we extracted adverse side effects (ASEs) to form a co-mention ASE-ASE network. 
Employing network analysis techniques such as clustering and graph neural network (GNN) classification, we identified groups of ASEs and revealed frequencies of unknown ASEs.
Analyzing social media data and biomedical studies on GLP-1, we identified GLP-1 ASEs not found in official manufactures' ASE lists (e.g., 
nightmare,
insomnia, and
irritable).
Detecting unknown ASE frequencies is critical in assessing the risk-benefit of drugs. 
Currently, these frequencies are found via clinical trials. 
Analyzing the ASE-ASE network using machine learning, we successfully identified (F1-score 0.83) frequent side effects.
Our model can be applied to any drug discussed online to identify unknown ASEs and predict their frequencies.

# Repository Contents
## Papaer LateX Documents

## Running the Code

`requirements.txt` - required python packages to run the code.

`.env.example` - variable and credentials needed to run the code.

## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to alon.bartal@biu.ac.il.



## Citing our work
If you find this code useful for your research, please consider citing us:
```
@article{Bartal2023GLP1,
  title     = {Integrating Online Biomedical Research and Social Media Analyses in Tracking and Understanding The GLP-1 Agonist Adverse Effects},
  author    = {Bartal, Alon and Jagodnik, Kathleen M. and Pliskin, N. and Avi, Seidmann},
  journal   = {},
  volume    = {},
  number    = {},
  pages     = {from page– to page},
  year      = {2023}
}
