# DRECT: An automatic approach for Developer REcommendation for Crowdsourcing software development Tasks.


## A multi-objective search-based approach, named DRECT, to recommend the set of developers for a given task on crowdsourcing software development(CSD). 
## Project for the submitted paper for Empirical Software Engineering Journal.

## Project Overview
The purpose of this project is twofold:

1. Data Collection: Collecting data from the crowdsourcing software development(CSD) i.e. topcoder and get features of tasks and develpoers.

2. DRECT: Apply a multi-objective search-based approach, named DRECT, to recommend the set of developers for a given task on crowdsourcing software development(CSD).

## Running Experiment

## Stack
- Windows 10
- VSCode 1.45.1
- PowerShell 7.0.1
- Python 3.8.3
- JavaScript 18.12.1

</ul>
<p><strong>Install python environment</strong></p>
<p>We develop the tool using python, so we recommend you to install an VSCode 1.45.1 Python 3.8.3 environment at: https://code.visualstudio.com/
</p>

<p><strong>Install JavaScript</strong></p>
<p>
We apply our script that implemented in JavaScript 18.12.1 to crawler data from topcoder project at: https://www.topcoder.com/.
</p>

#### Required python packages

- Pymoo: Multi-objective optimization in python! at: [Pymoo](https://pymoo.org/) 
- Data preprocessing: numpy, pandas, networkx.
- Algorithms required for the project NSGA2, NSGA3, UNSGA3, MOEAD, AGEMOEA.

  
#### The order for running the code are as follows:
1. [Data Collection](DataCollection/)
2. Data Preparation
   - [Cosine_Similarity](Cosine_Similarity/)
   - [Developer_Social_Network](Developer_Social_Network/)
3. [Developer Recommendation](DRECT.py.py/)


## Note to rerun experiment
- The "Dataset" directory is needed to be placed in the same directory as the project. The script that used to collect the data is aviliable here [Data Collection](DataCollection/). If you are eager to use our data, contact us here or via email.
- Run the developer recommendation code by run the file DRECT.py.

## Support
If you have any questions on this project or get stuck during code execution, feel free to create issue on this repository. We will be able to fix your issue.

