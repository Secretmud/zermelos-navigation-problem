# Classical computer requirements

As you can see in the table below, to simply run a 11 qutrit system would for a float32 precision require roughly 117 GB or RAM. Most computers nowadays are between 16 GB and 32GB. So doing a 10 qutrit simulation might our best option, altough it might be computationally costly in regards to time. 

| N | 3^N Dimension | (3^N)^2 Elements | float32 Bytes | float32 GiB | float64 Bytes | float64 GiB |
|---|---------------|------------------|---------------|-------------|---------------|-------------|
| 8  | 6,561        | 43,046,721        | 172,186,884      | 0.160 | 344,373,768       | 0.321 |
| 9  | 19,683       | 387,420,489       | 1,549,681,956    | 1.444 | 3,099,363,912     | 2.887 |
| 10 | 59,049       | 3,486,784,401     | 13,947,137,604   | 12.986 | 27,894,275,208    | 25.970 |
| 11 | 177,147      | 31,381,059,609    | 125,524,238,436  | 116.900 | 251,048,476,872   | 233.800 |
| 12 | 531,441      | 282,429,536,481   | 1,129,718,145,924| 1,027.600 | 2,259,436,291,848 | 2,055.300 |
| 13 | 1,594,323    | 2,541,865,828,329 | 10,167,463,313,316| 9,246.800 | 20,334,926,626,632| 18,493.600 |
| 14 | 4,782,969    | 22,876,792,454,961| 91,507,169,819,844| 83,471.200 | 183,014,339,639,688| 166,942.300 |
