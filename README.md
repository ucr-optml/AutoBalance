# AutoBalance

## Training of Search Phase:

### cifar10 example
```
python loss_search.py --config configs/cifar10/dyly_no_init/1.yaml
```

## Retraining Phase:

### cifar10 example

You can use the config.yaml as the config in your result folder (it is defined in the search phase config file).

```
python retraining.py --config results/cifar10/dyly_no_init/config.yaml
```