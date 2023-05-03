import json
import sys

config = json.load(open(sys.argv[1], 'r'))

AUC = ['breast_horizontal', 'default_credit_horizontal', 'give_credit_horizontal',
       'breast_vertical', 'default_credit_vertical', 'give_credit_vertical', ]
ACC = ['vehicle_scale_horizontal', 'vehicle_scale_vertical', 'femnist', 'reddit']
ERR = ['student_horizontal', 'motor_vertical', 'dvisits_vertical',  'student_vertical']


config = json.load(open(sys.argv[1], 'r'))

if config['dataset'] in AUC:
    metrics = 'auc'
elif config['dataset'] in ACC:
    metrics = 'accuracy'
elif config['dataset'] in ERR:
    metrics = 'mse'
else:
    raise NotImplementedError('Dataset {} is not supported.'.format(config['dataset']))


with open('./log/0.log', 'r') as f:
    lines = f.readlines()
out = []
for line in lines:
    log = json.loads(line)
    if log.get('event') is not None and log['event'] == 'model_evaluation' and log['action'] == 'end':
        try:
            real_metrics = {metrics: log['metrics']['target_metric']}
        except:
            real_metrics = {metrics: log['metrics'][metrics]}
        log['metrics'] = real_metrics
        out.append(json.dumps(log)+'\n')
    else:
        out.append(line)
with open('./log/0.log', 'w') as f:
    f.writelines(out)
