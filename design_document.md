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
*	Encapsulate dependencies in virtual environment
*	Make output compatible with Chromo-Analysis
*	Modularity of dynamics controller (forward-integrations, force calculations etc.)
*	Increased commenting and documentations
*	Establish active force code framework

### Non-goals:
*	Defining a multitude of active force descriptions
*	Speeding up simulation code
*	Developing new features (For example: worm-like chain model)
*	Sphinx documentation (Goal eventually, just not for this class project)
*	Using objects and classes (Goal eventually! Especially using Chromo class definitions! Unfortunately I don’t know how to use classes just yet, so we will not be doing that for this project. However, the code framework will be designed so that it’s easy to pass through a polymer object through once we’re ready to implement object-oriented programming.

### Future Goals:
*	Sphinx documentation
*	Object oriented programming

### Detailed Design:
The code input file is found in the input/ folder and is called sim_input. Within this file, you can specify parameters such as the number of chains, number of beads per polymer, spherical confinement size, active force magnitude, number of saved points/duration of simulation, etc.

Code necessary for the function of the simulator is in the twlcsim directory. The main file is twlc_sim.py, which calls all the necessary modules for running a simulation. These modules include scripts which initialize the simulated polymers in various configuration types and scripts that calculate various forces. The bd/ directory contains all of the force calculations and forward-integrations for running the simulations. The util/ directory contains functions for initializations and conformation specific characterizations independent of the function of the simulation.

In general, dynamic simulators work by calculating change of position of a bead based on the applied force. This simulator works by initializing a polymer chain of a desired length in Kuhn lengths (a polymer rigidity metric) with a specified number of beads. We are also able to specify the number of polymers of those specification running per simulation. Note: these polymers do NOT interact with each other. We are also able to specify whether or not it is within a spherical confinement and the size of that confinement. We can also specify whether or not the ends of the chains are tethered to the confinement. We can also specify the active force parameters. Current implementation utilizes the active force description as specified by Ghosh et al. (2022). This active force is characterized by its temporal-correlation relaxation time and its scale. Based off these specifications, the chain(s) will be initialized as a collection of coordinates specifying the position of each bead in the simulation. At each time step, the necessary forces will be calculated (Brownian, internal forces, active forces, confinement forcesc). Then a forward integration occurs, which calculates change in position based off the calculated forces. Once a new set of coordinates and bead directions is calculated, it is saved as a file. This continues until the simulation is done (as specified by the "Number of BD conformations saved" parameter in the sim_input file), resulting in an output directory containing a sequence of position vectors and active force vectors for each time point.

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
