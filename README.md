# News Recommendation
Implementation of several news recommendation methods in Pytorch.



### Requirements

- torch
- numpy
- pandas
- nltk
- scikit-learn
- matplotlib



### Usage

- **Clone this repository**	

  ```bash
  git clone https://github.com/yflyl613/NewsRecommendation.git
  cd NewsRecommendation
  mkdir dataset
  ```

- **Prepare dataset**

  - Dowload GloVe pre-trained word embedding (https://nlp.stanford.edu/data/glove.840B.300d.zip)

  - Unzip and put it `NewsRecommendation/dataset`

  - Create a new directory `NewsRecommendation/dataset/MINDsmall`

  - Download MIND-small dataset

    - Training Set: https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
    - Validation Set: https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip

  - Unzip and put them in `NewsRecommendation/dataset/MINDsmall`

  - After those operations above, the sturcture of `NewsRecommendation/dataset` should look like:

    ```
    |--- NewsRecommendation/
    	|--- dataset/
    		|--- MINDsmall/
    			|--- MINDsmall_dev
    			|--- MINDsmall_train
    		|--- glove.840B.300d.txt
    ```

- **Start training**

  ```bash
  cd NewsRecommendation/src
  python main.py  # add command line arguments behind, see `option.py` for details
  # eg. python main.py --model NRMS
  ```

  

### Result

| Model |  AUC  |  MRR  | nDCG@5 | nDCG@10 |           Configuration           |
| :---: | :---: | :---: | :----: | :-----: | :-------------------------------: |
| NAML  | 67.89 | 33.01 | 36.69  |  42.75  | batch size 64, training 2 epochs  |
| NRMS  | 65.93 | 31.40 | 34.51  |  40.83  | batch size 192, training 5 epochs |



### Citation

[1] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. **MIND: A Large-scale Dataset for News Recommendation.** ACL 2020.

