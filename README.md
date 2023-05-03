## 1. clone


## 2. env


```
conda create -n colink-unifed-fedlearner python=3.7
conda activate colink-unifed-fedlearner
```

+ #export PATH=/home/xiaojunxu/anaconda3/envs/colink-unifed-fedlearner/bin:$PATH  # my server is weird that it does not automatically switch to the anaconda python version

## 3. install

```pip install -e .
```

## 4. fedlearner

```sh install_fedlearner.sh
```
+ #my server does not have mysql installed. To accomodate to this, need to remove remove mysqlclient and sqlalchemy in fedlearner/requirement.txt, and remove import MySQLClient (L21) in fedlearner/fedlearner/common/db_client.py

##5. test
```python test/test_all_config.py
```
