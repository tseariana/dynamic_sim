# Active Rouse Model Dynamic Simulator
## Design Document

### Overview: 
The goal for this BIODS253 project is to develop the Spakowitz Lab’s existing dynamics simulator code to become compatible with the Chromo-Analysis code base
and to improve the documentation and modularity of the simulation code to establish it as a more permanent and lasting program to be used by future students of the 
lab.  Various changes will need to be made including establishment of a virtual environment, increased commenting, modularization through function delineation.  

### Background: 
The existing Rouse model dynamic simulator is the product of iterative editing from multiple students, including myself, and Dr. Andrew Spakowitz. The code written by me was written at very early stages of my code-writing career, resulting in overall poor quality and implementation. This code will need to be cleaned up. In addition to general code housekeeping and documentation, the code will need to be edited to ensure that the output of the dynamic simulator is compatible with existing analysis tool contained in Chromo-Analysis. While Chromo-Analysis is used for analysis of configurations coming from a Monte Carlo simulator known as Chromo (as opposed to a dynamic simulator), many programs for analyzing configurations can be used for analyzing configurations generated from dynamic simulations such as end-to-end distance. Because of this, we want to ensure that the output files is structured similarly to the files outputted from Chromo.
Lastly, active polymers have been a significant interest in the polymer physics community. Therefore, we want the dynamic simulator to adopt a framework that allows easy manipulation of active forces. That is, we want the ability to specify and call different active force descriptions (for example: temporally-correlated, spatially-correlated, Gaussian-distributed, etc.) While not all of these descriptions will be developed for this project, we will ensure that to active force code is designed to be modular so that any force description can be added with ease. 

### Goals:
•	Encapsulate dependencies in virtual environment
•	Make output compatible with Chromo-Analysis
•	Modularity of dynamics controller (forward-integrations, force calculations etc.)
•	Increased commenting and documentations
•	Establish active force code framework

### Non-goals:
•	Defining a multitude of active force descriptions
•	Speeding up simulation code
•	Developing new features (For example: worm-like chain model)
•	Sphinx documentation (Goal eventually, just not for this class project)
•	Using objects and classes (Goal eventually! Especially using Chromo class definitions! Unfortunately I don’t know how to use classes just yet, so we will not be doing that for this project. However, the code framework will be designed so that it’s easy to pass through a polymer object through once we’re ready to implement object-oriented programming.

### Future Goals:
•	Sphinx documentation
•	Object oriented programming

### Detailed Design:


### Python environment: 
We will include a conda environment 'requirements.txt' file

### Third Party Dependencies: 
This code utilizes Python 3.8 and NumPy. No other dependencies.

### Work estimate: 
This project should not take longer than 20 hours. Minimal code will need to be written. The work will mostly consist of code restructuring and documentation.

### Alternative approaches: 
There are not any alternative approaches. 

### Related work: 
Spakowitz Lab codebases Chromo and Chromo-Analysis, LAMMPS, NAMD, simDNA
