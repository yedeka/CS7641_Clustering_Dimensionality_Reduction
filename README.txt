CS7641 - Machine Learning 
Yogesh Edekar (GTID - yedekar3)
Assignment 3 - Clustering and Dimensionality Reduction

Source Code git repo - https://github.com/yedeka/CS7641_Clustering_Dimensionality_Reduction.git
Data - Available in the same git repo

environment setup - 
1] To setup Conda environment please download the source code and locate environment.yml file in the root folder.
2] If anaconda is not already installed on the system please install anaconda. 
3] On conda prompt please execute following commands for setting up the environment. 
	conda env create --file <Path_to_environment.yml>
	conda activate ml
	pip install yellowbrick
	conda install -c conda-forge kneed
4] Please install pycharm. 
5] Create a new project by importing the source code from git repo.
6] Selected existing pythin interpreter from the conda environment to be used within pycharm so that all the dependencies for the project will be automatically made available.
7] run the main.py file to run both the clustering and DR and obtain the charts as well as the observations for the experiments.