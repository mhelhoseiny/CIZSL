# CIZSL
[Creativity Inspired Zero-Shot Learning](https://arxiv.org/pdf/1904.01109.pdf)

Please contact us at m.elfeki11@gmail.com and mohamed.elhoseiny@gmail.com for questions.  The initial version of the code was written by Mohamed Elhoseiny and then it was followed by several improvements written by Mohamed Elfeki. 

Mohamed Elhoseiny, Mohamed Elfeki, Creativity Inspired Zero Shot Learning, Thirty-sixth International Conference on Computer Vision  (ICCV), 2019



# Requirements
Python 3.5

Pytorch 0.4.1

sklearn, scipy, matplotlib, numpy, random, copy


# How to run?

Data: download the [dataset CUBird and NABird](https://www.dropbox.com/s/9qovr86kgogkl6r/CUB_NAB_Data.zip

Please put the uncompressed data to the folder "data"

Run "run.sh" as follows: ./run.sh

<!-- Set the experiment parameters as parser arguments: Dataset, Splitmode, model-number, and the **main_dir** in "train_CIZSL.py"

Set the value of creativity_weight with the corresponding value to the dataset/splitmode found in the main of "train_CIZSL.py". If the value of the weight is unknown, set 'validate' to 1, and it will perform cross-validation to obtain the optimal weight, then use it to infer the ZSL.

P.S. We obtained those values by cross-validation, which can be found in "return_best_creativity_weight_validation" function


Run "train_CIZSL.py" as follows


With Cross Validation (Slower)
--------------------------------------------------------------------------------
`python train_CIZSL.py --dataset CUB --splitmode easy --validate 1`

`python train_CIZSL.py --dataset CUB --splitmode hard --validate 1`

`python train_CIZSL.py --dataset NAB --splitmode easy --validate 1`

`python train_CIZSL.py --dataset CUB --splitmode hard --validate 1`


Without Cross Validation (Faster), creativity_weight selected by cross validation. 
------------------------------------------------
`python train_CIZSL.py --dataset CUB --splitmode easy --creativity_weight 0.0001`

`python train_CIZSL.py --dataset CUB --splitmode hard --creativity_weight 0.1`

`python train_CIZSL.py --dataset NAB --splitmode easy --creativity_weight 1`

`python train_CIZSL.py --dataset NAB --splitmode hard --creativity_weight 0.1`

-->

