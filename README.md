Code for NeurIPS'20 paper Trading Personalization for Accuracy: Data Debugging in Collaborative Filtering

**Requirements**
```
python3
numpy
scipy
```

**Usage**

`python main.py --mode=test`

Run the above command to get the result on the movielens dataset.

To run the debug process, configure other parameters in the command, where the _mode_ parameter is set to _debug_.

Here is the code fragment about the settings of hyperparameters.

```
parser.add_argument("--dataset", type=str, default="movielens", help="dataset")
parser.add_argument("--delim", type=str, default="::", help="delimiter of each line in the dataset file")
parser.add_argument("--fold", type=int, default=4, help="# of fold to split the data")
parser.add_argument("--factor", type=int, default=10, help="# of dimension parameter of the CF model")
parser.add_argument("--lambda_u", type=float, default=0.1, help="regularization parameter lambda_u of the CF model")
parser.add_argument("--lambda_v", type=float, default=0.1, help="regularization parameter lambda_v of the CF model")
parser.add_argument("--als_iter", type=int, default=15, help="# of iterations for ALS training")
parser.add_argument("--debug_iter", type=int, default=20, help="# of iterations in the debugging stage")
parser.add_argument("--debug_lr", type=float, default=0.05, help="learning rate in the debugging stage")
parser.add_argument("--retrain", type=str, default="full", help="the retraining mode in the debugging stage: full/inc")
parser.add_argument("--process", type=int, default=4, help="# of processes in the debugging stage")
parser.add_argument("--mode", type=str, default="debug", help="debug/test")
```