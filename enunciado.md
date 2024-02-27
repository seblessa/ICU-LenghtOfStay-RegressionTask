## This assignment consists of running a full machine learning (ML) pipeline using some python libraries for big data.

### The 4 main objectives are:

- identify performance bottlenecks when using a specific library

- identify syntactic differences among the different libraries

- identify operations that are best suited to one particular library

- get acquainted with these libraries and knowing what is supported from pandas, scikit-learn and numpy

Your first task is to repeat experiments found at: https://databricks.com/blog/2021/04/07/benchmark-koalas-pyspark-and-dask.html. In these experiments, authors compare koalas[pyspark] and dask on various database-like operations.

You may use GCP to create a cluster similar to i3.4xlarge AWS and a machine similar to i3.16xlarge, up to what credits allow.

After repeating that experiment, your next task is to modify them to integrate code that uses Modin, JobLib and RapidsAI.

Besides running your experiments with the NYC taxi driver dataset, choose two other datasets: one smaller and one larger than the NYC taxi. They can be samples of the taxi dataset or other datasets.

For the taxi dataset the machine learning  task is to build a model to predict the target variable "fare_amount".

Suggested ML models: XGBRegressor and LogisticRegression (both will perform predictions, but the second one will perform classification. In that case, you need to discretize the "fare_amount" variable).

A full ML pipeline consists of: 

    reading the data
    preprocessing (that may include cleaning, filtering, feature selection etc)
    training and validation (use cross-validation and tune parameters)
    testing

Define scoring metrics to evaluate the models (accuracy, precision, recall, f-measure, error etc).

Suggested structure for the report: (it can be a notebook with comments)

1. Brief background on PySpark, Dask, Modin, JobLib, Rapids and Koalas

2. Materials and methods

      2.1 Machines used and their characteristics

      2.2 Datasets description

3. Experiment #1: repeat NYC taxi driver dataset study

      - report comparisons of execution times for each operation defined in the blog

4. Experiment #2

     - Run all datasets using Dask+Modin, Dask+Rapids, Dask+Modin+Rapids and Koalas

     - You may need to use cProfile or yappi to profile your codes (pycallgraph may not work because the code is somewhat complex)

5. Discussion and conclusions

 