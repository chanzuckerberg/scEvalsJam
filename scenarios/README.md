### Environment information

The current environment used is in poetry. The `pyproject.toml` file contains the necessary dependencies.

```
poetry install
```

### Usage

This code contains two major classes - MethodTransform and Scenarios. 

MethodTransform needs to be used and called in a method-specific manner before initializing the Scenarios class.

The Scenarios class initializes scenario-specific classes (e.g. Scenario1 - only one currently coded in), which is 
passed as a parameter in the Scenarios class, as well as the train and test anndata objects. 

A followable example with dummy data can be found in `splits/examples/test_train_split_example.ipynb`