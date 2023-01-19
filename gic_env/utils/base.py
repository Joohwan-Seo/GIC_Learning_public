from enum import Enum

import numpy as np


class PrimitiveStatus(Enum):
    FAIL = 0
    EXECUTING = 1
    SUCCESS = 2


class Primitive(object):
    """Base class for primitives, whose purpose is to generate the setpoint
    to the low-level controller to execute a motion pattern.

    :param learn_seq.controller.RobotState robot_state: access simulation data
    :param learn_seq.controller.TaskController controller: controller used
    :param float timeout : maximum execution time
    """
    def __init__(self, robot_state, controller, timeout=2., **kwargs):
        self.robot_state = robot_state
        self.controller = controller

        # upper velocity limit
        self.xdot_max = np.array([0.5] * 3 + [1.] * 3)

        # common parameters
        self.timeout = timeout
        self._reset_time()

    def _reset_time(self):
        self.dt = self.robot_state.get_timestep()
        self.t_exec = 0.
        self.timeout_count = self.timeout

    def plan(self):
        """Plan motion parameters before execution."""
        self._reset_time()

    def run(self, tau_cmd, viewer=None):
        """Execute until the stop condition is achieved. Return the execution
        time (the time as if the primitive is executed in the real world),
        and the status of termination (SUCCESS or FAIL)"""
        self.plan()
        status = PrimitiveStatus.EXECUTING
        t_start = self.robot_state.get_sim_time()
        while status is PrimitiveStatus.EXECUTING:
            # update state
            self.robot_state.update()
            # compute tau_cmd and set
            # tau_cmd, status = self.step()
            tau_cmd = np.zeros((self.robot_state.N))
            self.robot_state.set_control_torque(tau_cmd)
            # forward dynamic
            self.robot_state.update_dynamic()
            if viewer is not None:
                viewer.render()
        if status is PrimitiveStatus.FAIL:
            rew = -2
        else:
            rew = 0
        return self.robot_state.get_sim_time() - t_start, rew

    def _status(self):
        raise NotImplementedError

    def step(self):
        """Short summary.

        :return: The command torque and status of the primitive
        :rtype: tuple(np.array(7), PrimitiveStatus)

        """

        raise NotImplementedError
