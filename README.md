# dynamic_sim

This is a dynamic simulator for an active Rouse polymer physics model. 
For details of the project, please see the design document in `design_document.md`.

### Setup

1. Run `git clone git@github.com:tseariana/dynamic_sim.git` to download a copy of this repository.

2. Enter the repository with `cd dynamic_sim`.

3. Create a conda environment using the 'requirements.txt' file using the line 'conda create --name dynamic_sim --file requirements.txt'
  
4. Run `conda activate dynamic_sim` to activate the newly generated environment.

### Running a simulation

1. Enter the input/ directory with 'cd input'

2. Open the 'sim_input file'. Specify desired parameters. Note: If loading from previous configuration, position vector file (pos_file_#) and its corresponding active force file (fa_file_#) must be in the input folder. Save any changes.

3. Exit input folder with 'cd ..' and enter twlcsim folder with 'cd twlcsim'

4. Run simulation using 'python3 twlcsim.py'

5. All output files will be stored in the output folder. Check to see if this is the case by exiting the twlcsim folder with 'cd ..' and entering the opening the output folder with 'cd output'
