#!/usr/bin/env python

import rospy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
NO_TRAFFIC_LIGHT = -1

def get_car_xy_from_global_xy(car_x, car_y, yaw_rad, global_x, global_y):
    # Translate global point by car's position
    xg_trans_c = global_x - car_x
    yg_trans_c = global_y - car_y
    # Perform rotation to finish mapping
    # from global coords to car coords
    x = xg_trans_c * math.cos(0 - yaw_rad) - yg_trans_c * math.sin(0 - yaw_rad)
    y = xg_trans_c * math.sin(0 - yaw_rad) + yg_trans_c * math.cos(0 - yaw_rad)
    return (x, y)

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.base_waypoints_count = None

        self.current_pose = None

        # Loop Rate
        self.loop_frequency = 2 # Hz

        # Max velocity
        self.max_velocity = 10 # mph

        #Traffic light index
        self.traffic_light_index = NO_TRAFFIC_LIGHT # no traffic light.

        self.loop()

        rospy.spin()


    def loop(self):

        rate = rospy.Rate(self.loop_frequency)
        while not rospy.is_shutdown():
            if self.current_pose != None and self.base_waypoints != None:
                xyz_position = self.current_pose.position
                quaternion_orientation = self.current_pose.orientation

                p = xyz_position
                qo = quaternion_orientation

                p_list = [p.x, p.y, p.z]
                qo_list = [qo.x, qo.y, qo.z, qo.w]
                euler = euler_from_quaternion(qo_list)
                yaw_rad = euler[2]

                closest_waypoint_idx = None
                closest_waypoint_dist = None
                for idx in range(len(self.base_waypoints)):
                    wcx, wcy = self.get_on_car_waypoint_x_y(p, yaw_rad, idx)
                    if closest_waypoint_idx is None:
                        closest_waypoint_idx = idx
                        closest_waypoint_dist = math.sqrt(wcx**2 + wcy**2)
                    else:
                        curr_waypoint_dist = math.sqrt(wcx**2 + wcy**2)
                        if curr_waypoint_dist < closest_waypoint_dist: #
                            closest_waypoint_idx = idx
                            closest_waypoint_dist = curr_waypoint_dist



                wcx, wcy = self.get_on_car_waypoint_x_y(p, yaw_rad, closest_waypoint_idx)
                while wcx < 0.:
                    closest_waypoint_idx = (closest_waypoint_idx + 1) % self.base_waypoints_count
                    wcx, wcy = self.get_on_car_waypoint_x_y(p, yaw_rad, closest_waypoint_idx)

                next_waypoints = []
                for loop_idx in range(LOOKAHEAD_WPS):
                    wp_idx = (loop_idx + closest_waypoint_idx) % self.base_waypoints_count
                    next_waypoints.append(self.get_waypoint_to_sent(wp_idx))

                rospy.loginfo('INDEX {} wc [{:.3f},{:.3f}]'.format(closest_waypoint_idx, wcx, wcy))
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)
                lane.waypoints = self.adjust_velocity_to_stop(next_waypoints, closest_waypoint_idx)
                self.final_waypoints_pub.publish(lane)


            rate.sleep()

    def adjust_velocity_to_stop(self, waypoints, closest_waypoint_idx):
        traffic_index = self.traffic_light_index
        if traffic_index == NO_TRAFFIC_LIGHT:
            rospy.loginfo('TRAFFIC no traffic lights ahead')
            return waypoints

        # Map traffic index to current list
        if traffic_index < closest_waypoint_idx:
            traffic_index = self.base_waypoints_count - closest_waypoint_idx + traffic_index
        else:
            traffic_index = traffic_index - closest_waypoint_idx

        if traffic_index > LOOKAHEAD_WPS:
            rospy.loginfo('TRAFFIC no traffic lights before LOOKAHEAD_WPS {}'.format(traffic_index))
            return waypoints

        stop_buffer = 6
        if traffic_index > stop_buffer:
            traffic_index = traffic_index - stop_buffer

        points_before = 30
        m = self.max_velocity / float(points_before)
        for index in range(points_before):
            self.set_waypoint_velocity(waypoints, traffic_index - points_before + index , self.max_velocity - m * ( index + 1 ))

        for index in range(traffic_index, LOOKAHEAD_WPS):
            self.set_waypoint_velocity(waypoints, index , 0)
        rospy.loginfo('TRAFFIC traffic lights stop at {}, waypoints with zero {}'.format(traffic_index, LOOKAHEAD_WPS - traffic_index))
        return waypoints

    def get_on_car_waypoint_x_y(self, current_possition, yaw_rad, index):
        wgx, wgy = self.get_waypoint_x_y(index)
        return get_car_xy_from_global_xy(current_possition.x, current_possition.y, yaw_rad, wgx, wgy)

    def get_waypoint_x_y(self, index):
        waypoint = self.base_waypoints[index]
        x = waypoint.pose.pose.position.x
        y = waypoint.pose.pose.position.y
        return x, y

    def get_waypoint_to_sent(self, wp_idx):
        self.set_waypoint_velocity(self.base_waypoints, wp_idx, self.max_velocity)
        return self.base_waypoints[wp_idx]


    def pose_cb(self, msg):
        self.current_pose = msg.pose

    def waypoints_cb(self, lane):
        self.base_waypoints = lane.waypoints
        self.base_waypoints_count = len(self.base_waypoints)

    def traffic_cb(self, msg):
        self.traffic_light_index = msg.data;
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
