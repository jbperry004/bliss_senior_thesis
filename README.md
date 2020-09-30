# Bliss's Senior Thesis

Repository containing notebooks for Bliss's senior thesis

## Setup

### Installing Conda

First, ensure that you have an up-to-date installation of conda, a package manager which can create isolated Python environments for managing libraries on your machine:

Follow the directions at https://www.anaconda.com/products/individual, and verify your installation by opening the "Terminal" application and typing:

```
conda info
```

### Creating an Environment

First, assuming a recent installation of git, clone this repo at the appropriate place in your file system with the following command:

```
git clone https://github.com/jbperry004/bliss_senior_thesis.git
cd bliss_senior_thesis
```

Next, create a conda environment named `thesis` with the following commands:

```
conda env create -f bliss_senior_thesis_setup.yml
conda activate thesis
```

In general, before checking out and running code in the repository, you'll have to activate the environment with `conda activate thesis`.

### Creating a new kernel

To create a new Python kernel in this environment, please run:

```
python -m ipykernel install --user --name thesis
```

### Exploring on Jupyter

Run the following command to enter Jupyter Notebook:

```
jupyter notebook
```

A tab will pop up in your browser, and you can view any one of the notebooks in the repo from here. Once in a specific notebook, it will be necessary to change the kernel created above by going to "Kernel > Change Kernel > `thesis`".

Now, you should be all set! You can run individual code sells with Shift+Enter, or navigate to "Cell > Run All" to run every cell sequentially.
