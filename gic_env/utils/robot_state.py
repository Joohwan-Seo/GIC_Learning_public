import numpy as np
from mujoco_py import functions

from learn_seq.utils.filter import ButterLowPass
from gic_env.utils.mujoco import (MJ_SITE_OBJ, get_contact_force,
                                    inverse_frame, pose_transform,
                                    transform_spatial, get_contact_force_mine)

from ctypes import *


class RobotState:
    """Wrapper to the mujoco sim to store robot state and perform
    simulation operations (step, forward dynamic, ...).

    :param mujoco_py.MjSim sim: mujoco sim
    :param str ee_site_name: name of the end-effector site in mujoco xml model.
    :attr mujoco_py.MjData data: Description of parameter `data`.
    :attr mujoco_py.MjModel model: Description of parameter `model`.
    """

    def __init__(self, sim, ee_site_name, robot_name):
        self.data = sim.data
        self.model = sim.model
        self.robot_name = robot_name

        if self.robot_name == 'ur5e':
            self.N = 6
        elif self.robot_name == 'fanuc':
            self.N = 6
        else:
            self.N = 7
        self.ee_site_idx = functions.mj_name2id(
            self.model, MJ_SITE_OBJ, ee_site_name)
        self.isUpdated = False
        # low pass filter
        dt = sim.model.opt.timestep
        # print('dt:', dt)
        fs = 1 / dt
        cutoff = 50 ## Default value is 50
        self.fe = np.zeros(6)
        self.lp_filter = ButterLowPass(cutoff, fs, order=5)

        self.load_dll()

    def load_dll(self):
        dir = "/home/joohwan/deeprl/research/GIC_CQL/"
        if self.robot_name == 'ur5e':
            self.c = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/UR5e/funCori_UR5e_sq.so")
            self.g = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/UR5e/funGrav_UR5e_sq.so")
            self.m = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/UR5e/funMass_UR5e_sq.so")
            self.g_st = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/UR5e/g_st.so")
            self.Jb = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/UR5e/Jb.so")
            self.Je = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/UR5e/Je.so")
        elif self.robot_name == 'fanuc':
            self.c = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/fanuc/funCori_Fanuc.so")
            self.g = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/fanuc/funGrav_Fanuc.so")
            self.m = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/fanuc/funMass_Fanuc.so")
            self.g_st = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/fanuc/g_st.so")
            self.Jb = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/fanuc/Jb.so")
            self.Jb_dot = cdll.LoadLibrary(dir + "./gic_env/dynamic_models/fanuc/Jb_dot_fun.so")
            self.Je = cdll.LoadLibrary("./gic_env/dynamic_models/fanuc/Je.so")
        else:
            print('!!!WARNING!!!  That robot type is not implemented yet! (and maybe never ever)')
            quit()

    def update(self):
        """Update the internal simulation state (kinematics, external force, ...).
        Should be called before perform any setters or getters"""
        # update position-dependent state (kinematics, jacobian, ...)
        functions.mj_step1(self.model, self.data)
        # udpate the external force internally
        functions.mj_rnePostConstraint(self.model, self.data)
        # filter ee force
        self.update_ee_force()
        self.isUpdated = True

    def update_dynamic(self):
        """Update dynamic state (forward dynamic). The control torque should be
        set between self.update() and self.update_dynamic()"""
        functions.mj_step2(self.model, self.data)
        self.isUpdated = False

    def is_update(self):
        return self.isUpdated

    def update_ee_force(self):
        """calculate and filter the external end-effector force
        """
        p, q = self.get_pose()
        fe = get_contact_force(self.model, self.data, "peg", p, q)
        self.fe = self.lp_filter(fe.reshape((-1, 6)))[0, :]

    def reset_filter_state(self):
        self.lp_filter.reset_state()

    def get_pose(self, frame_pos=None, frame_quat=None):
        """Get current pose of the end-effector with respect to a particular frame

        :param np.array(3) frame_pos: if None then frame origin is coincide with
                                      the base frame
        :param np.array(4) frame_quat: if None then frame axis coincide with the
                                      base frame
        :return: position, quaternion
        :rtype: tuple(np.array(3), np.array(4))

        """
        p = self.data.site_xpos[self.ee_site_idx].copy()    # pos
        R = self.data.site_xmat[self.ee_site_idx].copy()    # rotation matrix

        q = np.zeros(4)
        functions.mju_mat2Quat(q, R)
        # print('current quat:', q)

        if frame_pos is None:
            frame_pos = np.zeros(3)
        if frame_quat is None:
            frame_quat = np.array([1., 0, 0, 0])
        # inverse frame T_0t -> T_t0
        inv_pos, inv_quat = inverse_frame(frame_pos, frame_quat)
        pf, qf = pose_transform(p, q, inv_pos, inv_quat)

        return pf, qf

    def get_pose2(self):
        p = self.data.site_xpos[self.ee_site_idx].copy()    # pos
        R = self.data.site_xmat[self.ee_site_idx].copy()    # rotation matrix

        return p, R.reshape((3,3))

    def get_pose_mine(self):
        q = self.get_joint_pose()
        pyarr = [0.]* 4**2

        g_st_mat = (c_double * len(pyarr))(*pyarr)
        self.g_st.g_st(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                    c_double(q[3]),c_double(q[4]),c_double(q[5]), g_st_mat)

        g_st = np.zeros((4*4))
        for i in range(4 * 4):
            g_st[i] = float(g_st_mat[i])

        g_st = g_st.reshape((4,4)).T

        return g_st[0:3,3], g_st[0:3,0:3]

    def forward_kinematics(self, q):
        pyarr = [0.]* 4**2

        g_st_mat = (c_double * len(pyarr))(*pyarr)
        self.g_st.g_st(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                    c_double(q[3]),c_double(q[4]),c_double(q[5]), g_st_mat)

        g_st = np.zeros((4*4))
        for i in range(4 * 4):
            g_st[i] = float(g_st_mat[i])

        g_st = g_st.reshape((4,4)).T

        return g_st[0:3,3], g_st[0:3,0:3]
    
    def get_ee_force_mine(self, frame_quat = None):
        fe = get_contact_force_mine(self.model, self.data, "peg")
        if frame_quat is None:
            return fe
        
        p, q = self.get_pose()
        qf0 = np.zeros(4)
        functions.mju_negQuat(qf0, frame_quat)
        qfe = np.zeros(4)
        functions.mju_mulQuat(qfe, qf0, q)  # qfe = qf0 * q0e

        ff = transform_spatial(fe, qfe)
        # desired = (np.zeros((3,)), np.eye(3))
        # ff = self.transform_adj(fe, desired)
        return ff

    def get_ee_force(self, frame_quat=None):
        """Get current force torque acting on the end-effector,
        with respect to a particular frame

        :param np.array(3) frame_pos: if None then frame origin is coincide with
                                      the ee frame
        :param np.array(4) frame_quat: if None then frame axis coincide with the
                                      ee frame
        :return: force:torque format
        :rtype: np.array(6)

        """
        # force acting on the ee, relative to the ee frame
        if frame_quat is None:
            return self.fe
        p, q = self.get_pose()
        qf0 = np.zeros(4)
        functions.mju_negQuat(qf0, frame_quat)
        qfe = np.zeros(4)
        functions.mju_mulQuat(qfe, qf0, q)  # qfe = qf0 * q0e

        # print(qfe)
        desired = (np.zeros(3,), np.eye(3))
        # transform to target frame
        ff = transform_spatial(self.fe, qfe)
        ff = self.transform_rot(self.fe, desired)
        # ff = self.transform_adj(self.fe, desired)
        # lowpass filter
        return ff
    
    def transform_rot(self, fe, desired):
        pe, Re = self.get_pose_mine()
        ps, Rs = desired

        R12 = Rs.T @ Re
        Mat = np.block([[R12, np.zeros((3, 3))], [np.zeros((3, 3)), R12]])

        return Mat.dot(fe)

    def get_jacobian(self):
        """Get 6x7 geometric jacobian matrix."""
        dtype = self.data.qpos.dtype
        jac = np.zeros((6, self.N), dtype=dtype)
        jac_pos = np.zeros((3 * self.N), dtype=dtype)
        jac_rot = np.zeros((3 * self.N), dtype=dtype)
        functions.mj_jacSite(
            self.model, self.data,
            jac_pos, jac_rot, self.ee_site_idx)
        jac[3:] = jac_rot.reshape((3, self.N))
        jac[:3] = jac_pos.reshape((3, self.N))
        # only return first 7 dofs
        return jac[:, :self.N].copy()
    
    def get_jacobian_mine(self):
        pyarr = [0.]* self.N**2
        q = self.get_joint_pose()

        J_mat = (c_double * len(pyarr))(*pyarr)
        self.Je.Je(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                    c_double(q[3]),c_double(q[4]),c_double(q[5]), J_mat)

        Je = np.zeros((self.N * 6))
        for i in range(self.N * 6):
            Je[i] = float(J_mat[i])

        return Je.reshape((6, self.N)).T
    
    def get_body_jacobian_dot(self):
        pyarr = pyarr = [0.]* self.N**2
        q = self.get_joint_pose()
        dq = self.get_joint_velocity()

        J_dot_mat = (c_double * len(pyarr))(*pyarr)
        self.Jb_dot.Jb_dot_fun(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                    c_double(q[3]),c_double(q[4]),c_double(q[5]), 
                    c_double(dq[0]),c_double(dq[1]),c_double(dq[2]),
                    c_double(dq[3]),c_double(dq[4]),c_double(dq[5]), 
                    J_dot_mat)

        Jb_dot = np.zeros((self.N * 6))
        for i in range(self.N * 6):
            Jb_dot[i] = float(J_dot_mat[i])

        return Jb_dot.reshape((6, self.N)).T   

    def get_body_jacobian(self):
        pyarr = [0.]* self.N**2
        q = self.get_joint_pose()
        

        J_mat = (c_double * len(pyarr))(*pyarr)
        self.Jb.Jb(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                    c_double(q[3]),c_double(q[4]),c_double(q[5]), J_mat)

        Jb = np.zeros((self.N * 6))
        for i in range(self.N * 6):
            Jb[i] = float(J_mat[i])

        return Jb.reshape((6, self.N)).T

    def get_body_ee_velocity(self):
        Jb = self.get_body_jacobian()
        dq = self.get_joint_velocity()
        Vb = Jb@dq.reshape((-1,1))

        return Vb
    
    def get_spatial_ee_velocity(self):
        Js = self.get_jacobian()
        dq = self.get_joint_velocity()

        Vs = Js@dq.reshape((-1,1))

        return Vs

    def get_dynamic_matrices(self):
        pyarr = [0.]* self.N**2
        pyarr_g = [0.] * self.N

        q = self.get_joint_pose()
        dq = self.get_joint_velocity()

        C_mat = (c_double * len(pyarr))(*pyarr)
        M_mat = (c_double * len(pyarr))(*pyarr)
        G_vec = (c_double * len(pyarr_g))(*pyarr_g)

        if self.robot_name == 'ur5e':
            self.c.funCori_UR5e_sq(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                                c_double(q[3]),c_double(q[4]),c_double(q[5]), 
                                c_double(dq[0]),c_double(dq[1]),c_double(dq[2]),
                                c_double(dq[3]),c_double(dq[4]),c_double(dq[5]), 
                                C_mat)
            self.m.funMass_UR5e_sq(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                                    c_double(q[3]),c_double(q[4]),c_double(q[5]), M_mat)
            self.g.funGrav_UR5e_sq(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                                    c_double(q[3]),c_double(q[4]),c_double(q[5]), G_vec)
        
        elif self.robot_name == 'fanuc':
            self.c.funCori_Fanuc(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                                 c_double(q[3]),c_double(q[4]),c_double(q[5]), 
                                 c_double(dq[0]),c_double(dq[1]),c_double(dq[2]),
                                 c_double(dq[3]),c_double(dq[4]),c_double(dq[5]), 
                                C_mat)
            self.m.funMass_Fanuc(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                                 c_double(q[3]),c_double(q[4]),c_double(q[5]), M_mat)
            self.g.funGrav_Fanuc(c_double(q[0]),c_double(q[1]),c_double(q[2]),
                                 c_double(q[3]),c_double(q[4]),c_double(q[5]), G_vec)

        C = np.zeros((self.N**2))
        M = np.zeros((self.N**2))
        G = np.zeros((self.N))

        for i in range(self.N**2):
            C[i] = float(C_mat[i])
            M[i] = float(M_mat[i])
        for i in range(self.N):
            G[i] = float(G_vec[i])

        return M.reshape((self.N,self.N)).T, C.reshape((self.N,self.N)).T, G.reshape((-1,1))

    def get_ee_velocity(self):
        """Get ee velocity w.s.t base frame"""
        dq = self.get_joint_velocity()
        jac = self.get_jacobian()
        return jac.dot(dq[:self.N])

    def get_joint_pose(self):
        return self.data.qpos.copy()

    def get_joint_velocity(self):
        return self.data.qvel.copy()

    def get_bias_torque(self):
        """Get the gravity and Coriolis, centrifugal torque """
        return self.data.qfrc_bias[:self.N].copy()

    def get_timestep(self):
        """Timestep of the simulator is timestep of controller."""
        return self.model.opt.timestep

    def get_sim_time(self):
        return self.data.time

    def set_control_torque(self, tau):
        """Set control torque to robot actuators."""
        assert tau.shape[0] == self.N
        # self.data.ctrl[:] = np.hstack((tau, [0, 0]))
        # if self.robot_name == 'ur5e':
        #     self.data.ctrl[:] = tau
        # else:
        #     self.data.ctrl[:] = np.hstack((tau, [0, 0]))

        self.data.ctrl[:] = tau
