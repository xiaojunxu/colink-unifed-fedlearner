import sys

from unifed.frameworks.fedlearner import protocol
from unifed.frameworks.fedlearner.workload_sim import *


def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here

