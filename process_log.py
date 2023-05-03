import json
import sys
import os

config = json.load(open(sys.argv[1], 'r'))

if config['algorithm'] == 'split_learning':
    with open('./log/t0.log', 'r') as f:
        train0 = f.readlines()
    with open('./log/0.log', 'r') as f:
        test0 = f.readlines()
    log0 = train0[:-1] + test0[1:2] + test0[-2:]
    with open('./log/0.log', 'w') as f:
        f.writelines(log0)
    os.remove('./log/t0.log')

if config['model'] == 'gbdt':
    for ii in [0, 1]:
        with open('./log/{}.log'.format(ii), 'r') as f:
            lines = f.readlines()
        p = -1
        for i in range(len(lines)):
            log = json.loads(lines[i])
            if log.get('flbenchmark') is not None and log['flbenchmark'] == 'end':
                p = i
        new_log = lines[:p+1]
        with open('./log/{}.log'.format(ii), 'w') as f:
            f.writelines(new_log)
