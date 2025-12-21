#codebase 
move the UV-Tunisia.csv into the data folder, rename it to UV-Tunisia.csv
create a list of various models

create the following files :

preprocess : contains all sorts of pre processing functions 
plot : all visualization code (same used for all models)

ts : contains all time series models related functions
ml : all machine learning models functions
dl : all deep learning models functions
boosters : all boosters related models

all models should have the same clean code structure, and visualize in the same manner

create the following folders :
models/ all models should be saved into this folder with ha clear name for future tests
gen/ contains all the plot images (save them with clear names)

all code should have minimal clear straight to point comments, and all sorts of arguments
create a main.py that will be interactive, and allow me to chose what model / models to train 