# heterogeneous-transmission

Code associated with the paper: **Heterogeneous transmission in groups induces a superlinear force of infection**

# Requirements

The scripts make use of 3 different submodules:
- [hgcm](https://github.com/gstonge/hgcm) for the approximate master equations framework
- [horgg](https://github.com/gstonge/horgg) for the synthetic generation of hypergraphs
- [schon](https://github.com/gstonge/schon) for the simulation of contagions on hypergraphs
Please refer to each submodule for their own requirements.

We also make use of standard python packages `numpy`, `scipy`, `matplotlib`,
and many others. See the `requirement.txt` file.

# Installation

First clone the repository and the submodules using

```
git clone --recurse-submodules git@github.com:gstonge/heterogeneous-transmission.git
```

or

```
git clone --recurse-submodules https://github.com/gstonge/heterogeneous-transmission.git
```

We suggest to start a new python virtual environment and install all requirements using
```
pip install -r requirements.txt
```

Depending on the modules you want to use, you might need to install `hgcm`, `horgg`, and/or `schon`.
Please refer to each submodule installation instructions.
The following command should work if the requirements were installed properly:
```
pip install hgcm horgg schon
```

# Usage

Each jupyter notebook that create the figures need some data to be generated. Go to the appropriate
directory and follow instructions.
