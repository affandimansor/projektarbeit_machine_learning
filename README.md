# Machine Learning Project

This guide provides step-by-step instructions to set up the environment locally and run the analysis scripts.

## Prerequisites

Ensure you have the following installed on your machine:
* A local copy of this repository
* **Python**: 3.9 <= version <= 3.12
* **pip**: For package installation
* **venv**: To create a virtual environment

---

## Set up a virtual environment
On the root directory run the following commands to:
1. create a virtual environment:
``` bash
    python3.12 -m venv venv
```
2. activate the virtual environment: 
``` bash
    source venv/bin/activate
```
3. install the required Python packages:
``` bash
    pip install -r requirements.txt
```

---

## Analysis of the algorithms
### Analysis workflow
For every analysis there is a workflow in png format.
This file can be found at \<Analysis Name\>/workflow/workflow.png.

### Run an analysis script
From the root directory, do the following:
1. change to the specific src folder, e.g. Decision_Tree:
``` bash
    cd Decision_Tree/src
```
2. run the script in the activated Python virtual environment:
``` bash
    python decision_tree_analysis.py
```
3. The results will be displayed on the terminal.

### Project report
[A project report is available under] (Documentation/MuhammadAffandi_BinMansor-Projektarbeit_Machine_Learning.pdf)