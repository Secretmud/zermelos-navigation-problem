# The project

To run this project I reccomend setting up a Python virtual environment. You can do this by running the following commands in your terminal:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Jupyter notebook 

To run the Jupyter notebook, you can use the following command in your terminal:

```bash
jupyter lab
```

This will let you interact with the yves_method.ipynb notebook file. Which imports all the necessary functions from the library.


# Classical computer requirements

As you can see in the table below, to simply run a 11 qutrit system would for a float32 precision require roughly 117 GiB or RAM. Most computers nowadays are between 16 GB and 32GB. So doing a 10 qutrit simulation might our best option, altough it might be computationally costly in regards to time. 

| N | 3^N Dimension | (3^N)^2 Elements | float32 GB | float64 GB |
|---|---------------|------------------|-----------:|-----------:|
| 8  | 6,561        | 43,046,721        | 0.172      | 0.344 |
| 9  | 19,683       | 387,420,489       | 1.550      | 3.099 |
| 10 | 59,049       | 3,486,784,401     | 13.947     | 27.894 |
| 11 | 177,147      | 31,381,059,609    | 125.524    | 251.048 |
| 12 | 531,441      | 282,429,536,481   | 1,129.718  | 2,259.436 |
| 13 | 1,594,323    | 2,541,865,828,329 | 10,167.463 | 20,334.927 |
| 14 | 4,782,969    | 22,876,792,454,961| 91,507.170 | 183,014.340 |
