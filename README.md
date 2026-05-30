The project is organised as such:


QPINN_Analysis/
│
├── src/
│   ├── model.py   : model classes
│   ├──pde.py      : PDE classes
│   ├──loss.py     : all loss functions
│   ├──sampler.py  : point sampler
│   ├──train.py    : training loop
│   ├──eval.py     : generates model output
│   ├──plotter.py  : makes figures
│   └── utils.py    : utility functions
│
├── notebooks/
│   └── contains all test notebooks
│
├── figures/
│   └── one folder per notebook, each folder contains figures
│
└── README.md

Workflow :

0) Each test you perform is its own notebook and generates figures within its own subfolder in figures/
   Create your notebook and explain what it does in the title

1) Initalize a model (from model.py) and a PDE (from pde.py)
2) Train the model on the pde using train (from train.py), loss history is outputted
3) Get model output for any spacetime interval using evaluate (from eval.py)
4) Plot your data using functions from plotter.py


All of these methods take default kwarguments, if you want to change/iterate over arguments input them manually in the notebook
If you want to use another model/loss/pde/sampler/... you can add your own class and call it
#TODO : currently train takes only one kind of sampler and loss cant deal with PDEs that have u_tt in the residual