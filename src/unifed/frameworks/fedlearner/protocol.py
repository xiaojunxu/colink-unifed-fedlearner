import os
import json
import sys
import subprocess
import tempfile
from typing import List

import colink as CL

from unifed.frameworks.fedlearner.util import store_error, store_return, GetTempFileName, get_local_ip

pop = CL.ProtocolOperator(__name__)
UNIFED_TASK_DIR = "unifed:task"

def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "fedlearner"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config

def run_external_process_and_collect_result(cl: CL.CoLink, participant_id,  role: str, n_epoch: int, tree_lr:float=0.0, tree_bins:int=0, tree_depth:int=0):#, server_ip: str):
    with GetTempFileName() as temp_log_filename, \
        GetTempFileName() as temp_output_filename:
        # note that here, you don't have to create temp files to receive output and log
        # you can also expect the target process to generate files and then read them

        # start training procedure
        if role == 'horizontal':
            new_env = os.environ.copy()
            new_env["PYTHONPATH"] = "./fedlearner:"+new_env.get("PYTHONPATH","")
            process = subprocess.Popen(
                [
                    "python",  
                    "evaluate_horizontal.py",
                    "config.json"
                ],
                env=new_env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        elif role == 'leader':
            new_env = os.environ.copy()
            new_env["PYTHONPATH"] = "./fedlearner:"+new_env.get("PYTHONPATH","")
            process = subprocess.Popen(
                [
                    "python",  
                    "leader.py",
                    "--local-addr=localhost:50051",
                    "--peer-addr=localhost:50052",
                    "--data-path=data/leader",
                    "--checkpoint-path=log/checkpoint/leader",
                    "--save-checkpoint-steps=10",
                    "--summary-save-steps=10",
                    "--export-path=model/leader/saved_model",
                    f"--epoch-num={n_epoch}",
                ],
                env=new_env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        elif role == 'follower':
            new_env = os.environ.copy()
            new_env["PYTHONPATH"] = "./fedlearner:"+new_env.get("PYTHONPATH","")
            process = subprocess.Popen(
                [
                    "python",  
                    "follower.py",
                    "--local-addr=localhost:50052",
                    "--peer-addr=localhost:50051",
                    "--data-path=data/follower",
                    "--checkpoint-path=log/checkpoint/follower",
                    "--save-checkpoint-steps=10",
                    "--summary-save-steps=10",
                    "--export-path=model/follower/saved_model",
                    f"--epoch-num={n_epoch}",
                ],
                env=new_env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        elif role == 'leadereval':
            new_env = os.environ.copy()
            new_env["PYTHONPATH"] = "./fedlearner:"+new_env.get("PYTHONPATH","")
            process = subprocess.Popen(
                [
                    "python",  
                    "leader.py",
                    "--local-addr=localhost:50051",
                    "--peer-addr=localhost:50052",
                    "--data-path=data/leader",
                    "--load-checkpoint-path=log/checkpoint/leader",
                    "--mode=eval"
                ],
                env=new_env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        elif role == 'followereval':
            new_env = os.environ.copy()
            new_env["PYTHONPATH"] = "./fedlearner:"+new_env.get("PYTHONPATH","")
            process = subprocess.Popen(
                [
                    "python",  
                    "follower.py",
                    "--local-addr=localhost:50052",
                    "--peer-addr=localhost:50051",
                    "--data-path=data/follower",
                    "--load-checkpoint-path=log/checkpoint/follower",
                    "--mode=eval"
                ],
                env=new_env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        elif role == 'treeleader':
            new_env = os.environ.copy()
            new_env["PYTHONPATH"] = "./fedlearner:"+new_env.get("PYTHONPATH","")
            process = subprocess.Popen(
                [
                    #"python -m fedlearner.model.tree.trainer",
                    "python",
                    "-m",
                    "fedlearner.model.tree.trainer",
                    "leader",
                    "--verbosity=1",
                    "--local-addr=localhost:50051",
                    "--peer-addr=localhost:50052",
                    "--file-type=tfrecord",
                    "--data-path=data/leader/leader_train.tfrecord",
                    "--validation-data-path=data/leader/leader_test.tfrecord",
                    "--l2-regularization=0.1",
                    "--checkpoint-path=exp/leader_checkpoints",
                    "--output-path=exp/leader_train_output.output",
                    f"--max-iters={n_epoch}",
                    f"--learning-rate={tree_lr}",
                    f"--max-bins={tree_bins}",
                    f"--max-depth={tree_depth}"
                ],
                env=new_env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        elif role == 'treefollower':
            new_env = os.environ.copy()
            new_env["PYTHONPATH"] = "./fedlearner:"+new_env.get("PYTHONPATH","")
            process = subprocess.Popen(
                [
                    #"python -m fedlearner.model.tree.trainer",
                    "python",
                    "-m",
                    "fedlearner.model.tree.trainer",
                    "follower",
                    "--verbosity=1",
                    "--local-addr=localhost:50052",
                    "--peer-addr=localhost:50051",
                    "--file-type=tfrecord",
                    "--data-path=data/follower/follower_train.tfrecord",
                    "--validation-data-path=data/follower/follower_test.tfrecord",
                    "--l2-regularization=0.1",
                    "--checkpoint-path=exp/follower_checkpoints",
                    "--output-path=exp/follower_train_output.output",
                    f"--max-iters={n_epoch}",
                    f"--learning-rate={tree_lr}",
                    f"--max-bins={tree_bins}",
                    f"--max-depth={tree_depth}"
                ],
                env=new_env,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        else:
            raise NotImplementedError()
        # gather result
        stdout, stderr = process.communicate()
        returncode = process.returncode
        with open(temp_output_filename, "rb") as f:
            output = f.read()
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
        with open(temp_log_filename, "rb") as f:
            log = f.read()
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)
        return json.dumps({
            #"server_ip": server_ip,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": returncode,
        })

@pop.handle("unifed.fedlearner:horizontal")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_horizontal(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    ## for certain frameworks, clients need to learn the ip of the server
    ## in that case, we get the ip of the current machine and send it to the clients
    #server_ip = get_local_ip()
    #cl.send_variable("leader_ip", server_ip, [p for p in participants if p.role == "follower"])
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "horizontal", unifed_config['training']['epochs'])#, leader_ip)

@pop.handle("unifed.fedlearner:leader")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_leader(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    ## for certain frameworks, clients need to learn the ip of the server
    ## in that case, we get the ip of the current machine and send it to the clients
    #server_ip = get_local_ip()
    #cl.send_variable("leader_ip", server_ip, [p for p in participants if p.role == "follower"])
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "leader", unifed_config['training']['epochs'])#, leader_ip)

@pop.handle("unifed.fedlearner:follower")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_follower(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    ## get the ip of the server
    #server_in_list = [p for p in participants if p.role == "server"]
    #assert len(server_in_list) == 1
    #p_server = server_in_list[0]
    #server_ip = cl.recv_variable("server_ip", p_server).decode()
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "follower", unifed_config['training']['epochs'])#, server_ip)

@pop.handle("unifed.fedlearner:leadereval")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_leadereval(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    ## for certain frameworks, clients need to learn the ip of the server
    ## in that case, we get the ip of the current machine and send it to the clients
    #server_ip = get_local_ip()
    #cl.send_variable("leader_ip", server_ip, [p for p in participants if p.role == "follower"])
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "leadereval", unifed_config['training']['epochs'])#, leader_ip)

@pop.handle("unifed.fedlearner:followereval")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_followereval(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    ## get the ip of the server
    #server_in_list = [p for p in participants if p.role == "server"]
    #assert len(server_in_list) == 1
    #p_server = server_in_list[0]
    #server_ip = cl.recv_variable("server_ip", p_server).decode()
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "followereval", unifed_config['training']['epochs'])#, server_ip)

@pop.handle("unifed.fedlearner:treeleader")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_treeleader(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    ## for certain frameworks, clients need to learn the ip of the server
    ## in that case, we get the ip of the current machine and send it to the clients
    #server_ip = get_local_ip()
    #cl.send_variable("leader_ip", server_ip, [p for p in participants if p.role == "follower"])
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "treeleader", unifed_config['training']['epochs'], tree_lr=unifed_config['training']['learning_rate'], tree_bins=unifed_config['training']['tree_param']['max_bins'], tree_depth=unifed_config['training']['tree_param']['max_depth'])#, leader_ip)

@pop.handle("unifed.fedlearner:treefollower")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_treefollower(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    ## get the ip of the server
    #server_in_list = [p for p in participants if p.role == "server"]
    #assert len(server_in_list) == 1
    #p_server = server_in_list[0]
    #server_ip = cl.recv_variable("server_ip", p_server).decode()
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "treefollower", unifed_config['training']['epochs'], tree_lr=unifed_config['training']['learning_rate'], tree_bins=unifed_config['training']['tree_param']['max_bins'], tree_depth=unifed_config['training']['tree_param']['max_depth'])#, server_ip)

