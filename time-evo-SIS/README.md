Here, we first need to generate a **configuration file**.
For instance:
```
python configgen_exp1.py
```

This creates a file `dat/conf_exp1.json`.

Then we get the solutions from the AME using
```
python datagen_hgcm.py ./dat/conf_exp1.json
```

Similarly, we generate simulations using
```
python simulation.py ./dat/conf_exp1.json
```

