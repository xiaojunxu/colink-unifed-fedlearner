import glob
import json
import pytest
import copy
import os
import time

import flbenchmark.datasets
import flbenchmark.logging

import colink as CL

def simulate_with_config(config_file_path):
    case_name = config_file_path.split("/")[-1].split(".")[0]
    with open(config_file_path, "r") as cf:
        config = json.load(cf)
    fedlearner_config = copy.deepcopy(config)
    fedlearner_config["training_param"] = fedlearner_config["training"]
    fedlearner_config.pop("training")
    fedlearner_config["bench_param"] = fedlearner_config["deployment"]
    with open("config.json", "w") as cf:
        json.dump(fedlearner_config, cf)

    # Process Dataset
    FATE_DATASETS_HORIZONTAL = ['breast_horizontal', 'default_credit_horizontal', 'give_credit_horizontal', 'student_horizontal', 'vehicle_scale_horizontal']
    FATE_DATASETS_VERTICAL = ['motor_vertical', 'breast_vertical', 'default_credit_vertical', 'dvisits_vertical', 'give_credit_vertical', 'student_vertical', 'vehicle_scale_vertical']
    LEAF_DATASETS = ['celeba', 'femnist', 'reddit', 'sent140', 'shakespeare', 'synthetic']

    if config['dataset'] in FATE_DATASETS_HORIZONTAL:
        test_type = 'horizontal'
    elif config['dataset'] in FATE_DATASETS_VERTICAL:
        if config['model'] == 'gbdt' and config['dataset'] != 'motor_vertical' and config['dataset'] != 'dvisits_vertical' and config['dataset'] != 'student_vertical':
            test_type = 'tree'
        else:
            test_type = 'vertical'
    elif config['dataset'] in LEAF_DATASETS:
        test_type = 'leaf'
    else:
        raise NotImplementedError('Dataset {} is not supported.'.format(config['dataset']))

    # Do simulation

    # use instant server for simulation
    ###ir = CL.InstantRegistry()
    #### TODO: confirm the format of `participants``
    ###config_participants = config["deployment"]["participants"]
    ###cls = []
    ###participants = []
    ###for _, role in config_participants:  # given user_ids are omitted and we generate new ones here
    ###    cl = CL.InstantServer().get_colink().switch_to_generated_user()
    ###    pop.run_attach(cl)
    ###    participants.append(CL.Participant(user_id=cl.get_user_id(), role=role))
    ###    cls.append(cl)
    ###task_id = cls[0].run_task("unifed.flower", json.dumps(config), participants, True)
    ###results = {}
    ###def G(key):
    ###    r = cl.read_entry(f"{UNIFED_TASK_DIR}:{task_id}:{key}")
    ###    if r is not None:
    ###        if key == "log":
    ###            return [json.loads(l) for l in r.decode().split("\n") if l != ""]
    ###        return r.decode() if key != "return" else json.loads(r)
    ###for cl in cls:
    ###    cl.wait_task(task_id)
    ###    results[cl.get_user_id()] = {
    ###        "output": G("output"),
    ###        "log": G("log"),
    ###        "return": G("return"),
    ###        "error": G("error"),
    ###    }
    ###return case_name, results
    from unifed.frameworks.fedlearner.protocol import pop, UNIFED_TASK_DIR
    if test_type == 'horizontal' or test_type == 'leaf':
        print ("HORIZONTAL")
        # use CL to run evaluate_horizontal --- change the test_train threads
        cl = CL.InstantServer().get_colink().switch_to_generated_user()
        pop.run_attach(cl)
        participants = [CL.Participant(user_id=cl.get_user_id(), role='horizontal')]
        task_id = cl.run_task("unifed.fedlearner", json.dumps(config), participants, True)
        results = {}
        def G(key):
            r = cl.read_entry(f"{UNIFED_TASK_DIR}:{task_id}:{key}")
            if r is not None:
                if key == "log":
                    return [json.loads(l) for l in r.decode().split("\n") if l != ""]
                return r.decode() if key != "return" else json.loads(r)
        cl.wait_task(task_id)
        results[cl.get_user_id()] = {
            "output": G("output"),
            "log": G("log"),
            "return": G("return"),
            "error": G("error"),
        }

        os.system('python process_log.py config.json')
    elif test_type == 'vertical':
        # make training data
        os.system('python make_data.py config.json train')

        # run leader/follower training
        ir = CL.InstantRegistry()
        config_participants = config["deployment"]["participants"]
        cls = []
        participants = []
        assert len(config_participants) == 2
        for _, role in config_participants:
            print (role)
            assert role == 'leader' or role == 'follower', "Unrecognized role: %s"%role
            cl = CL.InstantServer().get_colink().switch_to_generated_user()
            pop.run_attach(cl)
            participants.append(CL.Participant(user_id=cl.get_user_id(), role=role))
            cls.append(cl)
            print (cl, participants[-1])
        task_id = cls[0].run_task("unifed.fedlearner", json.dumps(config), participants, True)
        results = {}
        def G(key):
            r = cl.read_entry(f"{UNIFED_TASK_DIR}:{task_id}:{key}")
            if r is not None:
                if key == "log":
                    return [json.loads(l) for l in r.decode().split("\n") if l != ""]
                return r.decode() if key != "return" else json.loads(r)
        for cl in cls:
            cl.wait_task(task_id)
            results[cl.get_user_id()] = {
                "output": G("output"),
                "log": G("log"),
                "return": G("return"),
                "error": G("error"),
            }

        # move log
        os.system('mv -f log/0.log log/t0.log')
        os.system('mv -f log/1.log log/t1.log')
        time.sleep(1)

        # make testing data
        os.system('python make_data.py config.json test')

        # run leader/follower testing
        config_participants = config["deployment"]["participants"]
        cls = []
        participants = []
        assert len(config_participants) == 2
        for _, role in config_participants:
            print (role)
            assert role == 'leader' or role == 'follower', "Unrecognized role: %s"%role
            cl = CL.InstantServer().get_colink().switch_to_generated_user()
            pop.run_attach(cl)
            participants.append(CL.Participant(user_id=cl.get_user_id(), role=role+'eval'))
            cls.append(cl)
            print (cl, participants[-1])
        task_id = cls[0].run_task("unifed.fedlearner", json.dumps(config), participants, True)
        def G(key):
            r = cl.read_entry(f"{UNIFED_TASK_DIR}:{task_id}:{key}")
            if r is not None:
                if key == "log":
                    return [json.loads(l) for l in r.decode().split("\n") if l != ""]
                return r.decode() if key != "return" else json.loads(r)
        for cl in cls:
            cl.wait_task(task_id)
            results[cl.get_user_id()] = {
                "output": G("output"),
                "log": G("log"),
                "return": G("return"),
                "error": G("error"),
            }
        # move log
        os.system('mv -f log/t1.log log/1.log')
        os.system('rm -f log/t0.log')
        os.system('python process_log.py config.json')


    elif test_type == 'tree':
        # use CL to run leader & follower
        # make training data
        os.system('python make_data.py config.json train')

        # run leader/follower training
        ir = CL.InstantRegistry()
        config_participants = config["deployment"]["participants"]
        cls = []
        participants = []
        assert len(config_participants) == 2
        for _, role in config_participants:
            print (role)
            assert role == 'leader' or role == 'follower', "Unrecognized role: %s"%role
            cl = CL.InstantServer().get_colink().switch_to_generated_user()
            pop.run_attach(cl)
            print ("aaa", "tree"+role)
            participants.append(CL.Participant(user_id=cl.get_user_id(), role='tree'+role))
            cls.append(cl)
            print (cl, participants[-1])
        task_id = cls[0].run_task("unifed.fedlearner", json.dumps(config), participants, True)
        results = {}
        def G(key):
            r = cl.read_entry(f"{UNIFED_TASK_DIR}:{task_id}:{key}")
            if r is not None:
                if key == "log":
                    return [json.loads(l) for l in r.decode().split("\n") if l != ""]
                return r.decode() if key != "return" else json.loads(r)
        for cl in cls:
            cl.wait_task(task_id)
            results[cl.get_user_id()] = {
                "output": G("output"),
                "log": G("log"),
                "return": G("return"),
                "error": G("error"),
            }
        os.system('python process_log.py config.json')
    else:
        raise NotImplementedError(test_type)
    return case_name, results


def convert_metric(config):
    AUC = ['breast_horizontal', 'default_credit_horizontal', 'give_credit_horizontal',
        'breast_vertical', 'default_credit_vertical', 'give_credit_vertical']
    ACC = ['vehicle_scale_horizontal', 'vehicle_scale_vertical', 'femnist', 'celeba', 'reddit']
    ERR = ['student_horizontal', 'motor_vertical', 'dvisits_vertical',  'student_vertical']
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

if __name__ == "__main__":
    nw = time.time()
    target_case = "test/configs/case_0.json"
    # print(json.dumps(simulate_with_config(target_case), indent=2))
    results = simulate_with_config(target_case)
    for r in results[1].values():
        print(r["return"]["stderr"])
        print(r["return"]["stdout"])
    convert_metric(json.load(open("config.json", 'r')))
    flbenchmark.logging.get_report('./log')
    print("Time elapsed:", time.time() - nw)

