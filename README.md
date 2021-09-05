# Challenge #1 Drug activity

## Prepare the environment

```
python3.8 -m venv venv
source venv/bin/activate
pip3.8 install -r requirements.txt
```

## Other requirements

- Install torch for cuda
- Install tensorflow 

## Ideas and methods

1) run `./preprocssing.py` before anything else - it will save the train csv files under `./data`
2) three methods are implemented here:
    1. `approch#1_baseline.ipynb`
    2. `approach#2_zoo_sklearn,py`
    3. `approach#3_multitaskingNN.py`
3) The prediction csv file for submission is saved under `./data`   


#### The `approch#1_baseline.ipynb` was chosen for the server submission with 0.57 accuracy.

`approch#2` output fingerprints:
Task: 0 | RF: 0.9129 | SGD: 0.9104 | MLP: 0.7979 | OneVsRest: 0.7979 | SVC: 0.0033 | DT: 0.9100 | extra: 0.9092| KN: 0.9146 | perceptron: 0.9129 | Gauss: 0.9125 | GB: 0.9100 | Bern: 0.9042 | naiveg: 0.9146
best model for task 0 is KN
Task: 1 | RF: 0.9487 | SGD: 0.9096 | MLP: 0.7837 | OneVsRest: 0.7837 | SVC: 0.0396 | DT: 0.9379 | extra: 0.9404| KN: 0.9137 | perceptron: 0.9096 | Gauss: 0.9229 | GB: 0.9475 | Bern: 0.8279 | naiveg: 0.9137
best model for task 1 is random forest
Task: 2 | RF: 0.8908 | SGD: 0.8904 | MLP: 0.7762 | OneVsRest: 0.7762 | SVC: 0.0800 | DT: 0.8567 | extra: 0.8508| KN: 0.8946 | perceptron: 0.8904 | Gauss: 0.8800 | GB: 0.8988 | Bern: 0.8229 | naiveg: 0.8946
best model for task 2 is GB
Task: 3 | RF: 0.9150 | SGD: 0.9129 | MLP: 0.6246 | OneVsRest: 0.6246 | SVC: 0.0079 | DT: 0.9167 | extra: 0.9100| KN: 0.9171 | perceptron: 0.9154 | Gauss: 0.9154 | GB: 0.9137 | Bern: 0.8467 | naiveg: 0.9171
best model for task 3 is KN
Task: 4 | RF: 0.9363 | SGD: 0.9442 | MLP: 0.8387 | OneVsRest: 0.8387 | SVC: 0.0288 | DT: 0.9300 | extra: 0.9217| KN: 0.9471 | perceptron: 0.9442 | Gauss: 0.9304 | GB: 0.9433 | Bern: 0.8500 | naiveg: 0.9471
best model for task 4 is KN
Task: 5 | RF: 0.9425 | SGD: 0.9437 | MLP: 0.8063 | OneVsRest: 0.8063 | SVC: 0.0075 | DT: 0.9400 | extra: 0.9346| KN: 0.9471 | perceptron: 0.9454 | Gauss: 0.9408 | GB: 0.9396 | Bern: 0.9375 | naiveg: 0.9471
best model for task 5 is KN
Task: 6 | RF: 0.9517 | SGD: 0.9463 | MLP: 0.9471 | OneVsRest: 0.9471 | SVC: 0.0050 | DT: 0.9504 | extra: 0.9475| KN: 0.9504 | perceptron: 0.9504 | Gauss: 0.9504 | GB: 0.9508 | Bern: 0.9479 | naiveg: 0.9504
best model for task 6 is random forest
Task: 7 | RF: 0.6808 | SGD: 0.6771 | MLP: 0.6808 | OneVsRest: 0.6808 | SVC: 0.0042 | DT: 0.6787 | extra: 0.6754| KN: 0.6808 | perceptron: 0.6796 | Gauss: 0.6808 | GB: 0.6775 | Bern: 0.6229 | naiveg: 0.6808
best model for task 7 is random forest
Task: 8 | RF: 0.6863 | SGD: 0.6875 | MLP: 0.5667 | OneVsRest: 0.5667 | SVC: 0.0467 | DT: 0.6658 | extra: 0.6654| KN: 0.6917 | perceptron: 0.6875 | Gauss: 0.6842 | GB: 0.6925 | Bern: 0.6142 | naiveg: 0.6917
best model for task 8 is GB
Task: 9 | RF: 0.6304 | SGD: 0.6292 | MLP: 0.4713 | OneVsRest: 0.4713 | SVC: 0.0267 | DT: 0.6250 | extra: 0.6167| KN: 0.6333 | perceptron: 0.6292 | Gauss: 0.6283 | GB: 0.6329 | Bern: 0.5763 | naiveg: 0.6333
best model for task 9 is KN
Task: 10 | RF: 0.9129 | SGD: 0.0458 | MLP: 0.8171 | OneVsRest: 0.8171 | SVC: 0.9279 | DT: 0.8921 | extra: 0.8846| KN: 0.9279 | perceptron: 0.9279 | Gauss: 0.8983 | GB: 0.9283 | Bern: 0.8900 | naiveg: 0.9279
best model for task 10 is GB
Final accuracy with the stacked method: 0.86