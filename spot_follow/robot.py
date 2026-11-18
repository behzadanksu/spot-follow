"""Thin wrapper around the Spot SDK: connect, lease, E-Stop, power, images, motion."""
import logging
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_odom_tform_body
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_sit,
                                         blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

from .cameras import CAMERA_SOURCES, yaw_of

log = logging.getLogger(__name__)


class EstopNotReady(RuntimeError):
    pass


class SpotInterface:
    """Owns every SDK client. Use as a context manager for the motion session:

        spot = SpotInterface(cfg); spot.connect()
        with spot.motion_session():   # lease + power on + stand ... sit + power off
            spot.walk_to(...)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = None
        self._lease_keepalive = None
        self._estop_keepalive = None
        self._mobility_params = None

    # ------------------------------------------------------------------ connection
    def connect(self):
        bosdyn.client.util.setup_logging(verbose=False)
        sdk = bosdyn.client.create_standard_sdk('SpotFollowClient')
        self.robot = sdk.create_robot(self.cfg.robot.hostname)
        bosdyn.client.util.authenticate(self.robot)   # BOSDYN_CLIENT_USERNAME/PASSWORD or prompt
        self.robot.time_sync.wait_for_sync()
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        self._mobility_params = self._build_mobility_params()
        log.info('Connected to %s', self.cfg.robot.hostname)
        return self

    def list_image_sources(self):
        return [s.name for s in self.image_client.list_image_sources()]

    def battery_percent(self):
        st = self.state_client.get_robot_state()
        if st.battery_states:
            return st.battery_states[0].charge_percentage.value
        return None

    # ------------------------------------------------------------------ E-Stop
    def estop_status(self):
        status = self.estop_client.get_status()
        n_endpoints = len(status.endpoints)
        return status.stop_level, n_endpoints

    def _ensure_estop(self):
        if self.cfg.robot.own_estop:
            log.warning('Registering THIS program as the only E-Stop. Ctrl-C twice = cut motor power.')
            endpoint = EstopEndpoint(self.estop_client, 'spot_follow', self.cfg.robot.estop_timeout_s)
            endpoint.force_simple_setup()
            self._estop_keepalive = EstopKeepAlive(endpoint)
            time.sleep(0.5)
        level, n = self.estop_status()
        if n == 0:
            raise EstopNotReady('No E-Stop is configured. Keep the tablet app open (its E-Stop stays active '
                                'after you release control), run the SDK estop_gui example, or set '
                                'robot.own_estop: true.')
        if level != estop_pb2.ESTOP_LEVEL_NONE:
            raise EstopNotReady('Robot is E-Stopped. Release the E-Stop first.')

    def cut_power_now(self):
        """Emergency path: E-Stop via our own endpoint if we have one, else power off."""
        if self._estop_keepalive is not None:
            self._estop_keepalive.settle_then_cut()
        else:
            self.robot.power_off(cut_immediately=True, timeout_sec=5)

    # ------------------------------------------------------------------ session
    def motion_session(self):
        return _MotionSession(self)

    def _acquire_lease(self):
        if self.cfg.robot.take_lease:
            log.warning('Taking the body lease from whoever holds it')
            self.lease_client.take()
            self._lease_keepalive = LeaseKeepAlive(self.lease_client, return_at_exit=True)
        else:
            self._lease_keepalive = LeaseKeepAlive(self.lease_client, must_acquire=True,
                                                   return_at_exit=True)

    def power_on_and_stand(self):
        log.info('Powering on motors...')
        self.robot.power_on(timeout_sec=30)
        log.info('Standing...')
        blocking_stand(self.command_client, timeout_sec=15)

    def sit_and_power_off(self):
        try:
            self.stop()
            blocking_sit(self.command_client, timeout_sec=15)
            self.robot.power_off(cut_immediately=False, timeout_sec=20)
            log.info('Sat down and powered off')
        except Exception as exc:  # noqa: BLE001 - best effort during shutdown
            log.error('Shutdown problem: %s', exc)

    def release(self):
        if self._lease_keepalive is not None:
            self._lease_keepalive.shutdown()
            self._lease_keepalive = None
        if self._estop_keepalive is not None:
            self._estop_keepalive.shutdown()
            self._estop_keepalive = None

    # ------------------------------------------------------------------ perception
    def image_requests(self, cameras, with_depth=True, quality=70):
        reqs = []
        for cam in cameras:
            vis, depth, _ = CAMERA_SOURCES[cam]
            reqs.append(build_image_request(vis, quality_percent=quality))
            if with_depth:
                reqs.append(build_image_request(depth))   # depth defaults to raw uint16
        return reqs

    def get_images(self, requests):
        return self.image_client.get_image(requests)

    def odom_pose(self):
        """(x, y, yaw) of the body in the odom frame, from the latest robot state."""
        snap = self.state_client.get_robot_state().kinematic_state.transforms_snapshot
        odom_T_body = get_odom_tform_body(snap)
        return odom_T_body.x, odom_T_body.y, yaw_of(odom_T_body)

    # ------------------------------------------------------------------ motion
    def _build_mobility_params(self):
        f = self.cfg.follow
        v, w = float(f.max_lin_vel), float(f.max_ang_vel)
        limit = geometry_pb2.SE2VelocityLimit(
            max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=v, y=v), angular=w),
            min_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=-v, y=-v), angular=-w))
        obstacles = spot_command_pb2.ObstacleParams(
            obstacle_avoidance_padding=float(f.obstacle_padding_m))
        return spot_command_pb2.MobilityParams(
            vel_limit=limit, obstacle_params=obstacles,
            locomotion_hint=spot_command_pb2.HINT_AUTO)

    def walk_to(self, x, y, heading):
        """Walk to (x, y, heading) in the odom frame. Expires after command_duration_s."""
        cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=x, goal_y=y, goal_heading=heading, frame_name=ODOM_FRAME_NAME,
            params=self._mobility_params, body_height=self.cfg.follow.body_height)
        self.command_client.robot_command(cmd, end_time_secs=time.time() + self.cfg.follow.command_duration_s)

    def turn(self, v_rot, duration_s=None):
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=0.0, v_y=0.0, v_rot=v_rot, params=self._mobility_params,
            body_height=self.cfg.follow.body_height)
        dur = duration_s if duration_s is not None else self.cfg.follow.command_duration_s
        self.command_client.robot_command(cmd, end_time_secs=time.time() + dur)

    def stop(self):
        self.command_client.robot_command(RobotCommandBuilder.stop_command())


class _MotionSession:

    def __init__(self, spot: SpotInterface):
        self.spot = spot

    def __enter__(self):
        s = self.spot
        s._ensure_estop()
        s._acquire_lease()
        try:
            s.power_on_and_stand()
        except Exception:
            s.release()
            raise
        return s

    def __exit__(self, exc_type, exc, tb):
        s = self.spot
        try:
            if s.cfg.robot.sit_and_power_off_at_exit:
                s.sit_and_power_off()
            else:
                s.stop()
        finally:
            s.release()
        return False
