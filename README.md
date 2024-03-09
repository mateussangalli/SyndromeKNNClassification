To plot the tSNE graph and obtain the data distribution do
```
python data_exploration.py --data_path [path/to/data]
```

For the main evaluation do
```
python cross_validation.py --data_path [path/to/data]
```

The results will be saved in the current work directory in a folder ```results/```

The tests for the KNNClassifier can be run using pytest.
