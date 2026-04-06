import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
try:
    import termios
    import tty
except ImportError:
    import msvcrt
import rospy
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import os
import sys
import threading
from piper_control import PIPER
from piper_msgs.msg import PosCmd

piper_control = PIPER()

exit_flag = False

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        last_path = os.path.dirname(current_dir)
        last_path = os.path.dirname(last_path)
        last_path = os.path.dirname(last_path)
        urdf_dir = os.path.join(last_path, 'piper_description/urdf/piper_description.urdf')
        # urdf_path = '/home/agilex/piper_ws/src/piper_description/urdf/piper_description.urdf'
        urdf_path = urdf_dir
    
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        self.mixed_jointsToLockIDs = ["joint7",
                                      "joint8"
                                      ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        # q = quaternion_from_euler(0, -1.57, -1.57)
        q = quaternion_from_euler(0, 0, 0)
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          # pin.Quaternion(1, 0, 0, 0),
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([0.0, 0.0, 0.0]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # # Initialize the Meshcat visualizer  for visualization
        self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        self.vis.displayFrames(True, frame_ids=[113, 114], axis_length=0.15, axis_width=5)
        self.vis.display(pin.neutral(self.reduced_robot.model))

        # Enable the display of end effector target frames with short axis lengths and greater width.
        frame_viz_names = ['ee_target']
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                      [0, 1, 0], [0.6, 1, 0],
                      [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 10
        for frame_viz_name in frame_viz_names:
            self.vis.viewer[frame_viz_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=axis_width,
                        vertexColors=True,
                    ),
                )
            )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # # Get the hand joint ID and define the error function
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # for smooth

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # print("self.reduced_robot.model.lowerPositionLimit:", self.reduced_robot.model.lowerPositionLimit)
        # print("self.reduced_robot.model.upperPositionLimit:", self.reduced_robot.model.upperPositionLimit)
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper/2.0, -gripper/2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.vis.viewer['ee_target'].set_transform(target_pose)     # for visualization

        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 30.0/180.0*3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            self.vis.display(sol_q)  # for visualization

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))

            is_collision = self.check_self_collision(sol_q, gripper)

            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return sol_q, '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        # print("collision:", collision)
        return collision

    def get_ik_solution(self, x,y,z,roll,pitch,yaw):
        
        q = quaternion_from_euler(roll, pitch, yaw, axes='rzyx')
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        print(target)
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous,0)
        # print("result:", sol_q)
        
        if get_result :
            piper_control.joint_control_piper(sol_q[0],sol_q[1],sol_q[2],sol_q[3],sol_q[4],sol_q[5],0)
        else :
            print("collision!!!")

    
class C_PiperIK():
    def __init__(self):
        rospy.init_node('inverse_solution_node', anonymous=True)
        # 创建Arm_IK实例
        self.arm_ik = Arm_IK()
        
        # 启动订阅线程
        sub_pos_th = threading.Thread(target=self.SubPosThread, daemon=True)
        sub_pos_th.daemon = True
        sub_pos_th.start()
    
    def SubPosThread(self):
        # 创建订阅者，监听PosCmd类型的消息
        rospy.Subscriber('pin_pos_cmd', PosCmd, self.pos_cmd_callback)
        rospy.spin()

    def pos_cmd_callback(self, msg):
        # 获取PosCmd类型消息中的数据
        x = msg.x
        y = msg.y
        z = msg.z
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw

        # 调用Arm_IK类的逆解函数
        self.arm_ik.get_ik_solution(x, y, z, roll, pitch, yaw)

def key_listener():
    global exit_flag
    if os.name == 'nt':
        while True:
            if msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    exit_flag = True
                    print("exit...")
                    break
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                if sys.stdin.read(1).lower() == 'q':
                    exit_flag = True
                    print("exit...")
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def eef_pos_to_qpos(eef_pos: np.ndarray) -> np.ndarray:
    """
    Convert 14-dimensional end-effector positions to joint positions (6 DOF + 1 gripper for each arm).
    
    Args:
        eef_pos: numpy array of shape (14,) containing:
            - eef_pos[0:6]: left end-effector pose (x, y, z, roll, pitch, yaw in meters/degrees)
            - eef_pos[6]: left gripper position (normalized 0-1)
            - eef_pos[7:13]: right end-effector pose (x, y, z, roll, pitch, yaw in meters/degrees)
            - eef_pos[13]: right gripper position (normalized 0-1)
    
    Returns:
        numpy array of shape (14,) containing:
            - qpos[0:6]: left arm joint angles (6 DOF)
            - qpos[6]: left gripper position
            - qpos[7:13]: right arm joint angles (6 DOF) 
            - qpos[13]: right gripper position
    """
    if eef_pos.shape != (14,):
        raise ValueError(f"Expected eef_pos to have shape (14,), got {eef_pos.shape}")
    
    # Initialize inverse kinematics for both arms
    ik_left = Arm_IK()
    ik_right = Arm_IK()
    
    # Extract end-effector poses for each arm
    left_eef_pose = eef_pos[0:6]  # [x, y, z, roll, pitch, yaw]
    right_eef_pose = eef_pos[7:13]  # [x, y, z, roll, pitch, yaw]
    
    # Extract gripper positions
    left_gripper = eef_pos[6]
    right_gripper = eef_pos[13]
    
    # Convert positions from meters to millimeters for IK
    left_eef_pose_mm = left_eef_pose.copy()
    left_eef_pose_mm[0:3] = left_eef_pose_mm[0:3] * 1000  # Convert to mm
    left_eef_pose_mm[3:6] = left_eef_pose_mm[3:6] / np.pi * 180  # Convert roll, pitch, yaw to degrees
    right_eef_pose_mm = right_eef_pose.copy()
    right_eef_pose_mm[0:3] = right_eef_pose_mm[0:3] * 1000  # Convert to mm
    right_eef_pose_mm[3:6] = right_eef_pose_mm[3:6] / np.pi * 180  # Convert roll, pitch, yaw to degrees
    
    # Calculate inverse kinematics for both arms
    left_joints, _, left_success = ik_left.ik_fun(
        target_pose=create_target_pose(left_eef_pose_mm),
        gripper=left_gripper
    )
    right_joints, _, right_success = ik_right.ik_fun(
        target_pose=create_target_pose(right_eef_pose_mm),
        gripper=right_gripper
    )
    
    # Combine into final output
    qpos = np.zeros(14)
    qpos[0:6] = left_joints  # left arm joint angles
    qpos[6] = left_gripper   # left gripper position
    qpos[7:13] = right_joints  # right arm joint angles
    qpos[13] = right_gripper   # right gripper position
    
    return qpos


def batch_eef_pos_to_qpos(eef_pos_batch: np.ndarray) -> np.ndarray:
    """
    Convert a batch of 14-dimensional end-effector positions to joint positions.
    
    Args:
        eef_pos_batch: numpy array of shape (batch_size, 14) or (batch_size, timesteps, 14)
    
    Returns:
        numpy array of same shape as input containing joint positions
    """
    if len(eef_pos_batch.shape) == 2:
        # Single batch dimension
        batch_size = eef_pos_batch.shape[0]
        qpos_batch = np.zeros_like(eef_pos_batch)
        
        for i in range(batch_size):
            qpos_batch[i] = eef_pos_to_qpos(eef_pos_batch[i])
            
    elif len(eef_pos_batch.shape) == 3:
        # Batch and time dimensions
        batch_size, timesteps = eef_pos_batch.shape[0], eef_pos_batch.shape[1]
        qpos_batch = np.zeros_like(eef_pos_batch)
        
        for i in range(batch_size):
            for t in range(timesteps):
                qpos_batch[i, t] = eef_pos_to_qpos(eef_pos_batch[i, t])

    elif len(eef_pos_batch.shape) == 1:
        # same as eef_pos_to_qpos
        qpos_batch = eef_pos_to_qpos(eef_pos_batch)
        
    else:
        raise ValueError(f"Expected eef_pos_batch to have 1, 2 or 3 dimensions, got {len(eef_pos_batch.shape)}")
    
    return qpos_batch


def create_target_pose(eef_pose: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix from end-effector pose.
    
    Args:
        eef_pose: numpy array of shape (6,) containing [x, y, z, roll, pitch, yaw]
                 where x, y, z are in mm and roll, pitch, yaw are in degrees
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    from tf.transformations import quaternion_from_euler
    
    # Extract position and orientation
    x, y, z = eef_pose[0:3]
    roll, pitch, yaw = np.radians(eef_pose[3:6])  # Convert degrees to radians
    
    # Create quaternion from Euler angles
    quat = quaternion_from_euler(roll, pitch, yaw, axes='rzyx')
    
    # Create SE3 transformation
    target = pin.SE3(
        pin.Quaternion(quat[3], quat[0], quat[1], quat[2]),
        np.array([x, y, z])
    )
    
    return target.homogeneous

if __name__ == "__main__":
    piper_ik = C_PiperIK()
    print("Press 'q' to quit")
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()
    while not exit_flag:
        time.sleep(0.1)
