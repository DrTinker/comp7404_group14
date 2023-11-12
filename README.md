This is comp7404 group14 project

# Paper Selection
[Hashing based Efficient Inference for Image-Text Matching](https://aclanthology.org/2021.findings-acl.66.pdf)

# Dataset
Flickr30K is used in the implementation. download here: [SCAN Faster R-CNN Image Features](https://www.kaggle.com/datasets/kuanghueilee/scan-features).
`/data/f30k_precomp` and `/vocab` are needed.

# Train and Validation
**step 1. train BFAN model**
replace *$DATA_PATH* and *$VOCAB_PATH* to your own path
```shell
python BFAN_train.py --data_path "$DATA_PATH" --vocab_path "$VOCAB_PATH" --data_split train
```

**step 2. train HEI model**
replace *$DATA_PATH* and *$VOCAB_PATH* to your own path
```shell
python HEI_train.py --data_split train
```

**step 2. validate or test**
replace $SPLIT to dev(for validation) or test(for test)
```shell
python test.py --data_split "$SPLIT"
```

## Environment
Recommended:
* Python 3.7
* PyTorch 2.1.0
* Numpy(>1.12.1)
* TensorBoard
* torchvision
* nltk
