import cv2 as cv
import numpy as np
from blob_follow_ros2.utils.blob_functions import center_lane
from blob_follow_ros2.utils.functions import Vec, cols, rows

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rclpy.parameter_service import SetParametersResult
from cv_bridge import CvBridge


class BlobFollow(Node):

    config: dict
    bridge = CvBridge()
    debug_image_raw : cv.Mat = None

    def __init__(self):
        super().__init__("blob_follow")
        self.config = {}

        # node parameters
        params = [
            ("speed", 0),
            ("turn_mult", 0),
            ("input_mask_topic", "/mask"),
            ("output_twist_topic", "/control/cmd_vel"),
            ("input_debug_image_topic", "/routecam/image_raw"),
            ("canny_lower_thresh", 700),
            ("canny_upper_thresh", 2000),
            ("line_dilation_size", 5),
            ("top_crop_pct", 0.5),
            ("hough_rho", 2),
            ("hough_line_thresh", 85),
            ("hough_min_line_len", 70),
            ("hough_max_line_gap", 10),
            ("line_min_slope", 0.5),
            ("line_max_slope", 1.7),
        ]

        self.declare_parameters(namespace="", parameters=params)
        param_values = self.get_parameters([x[0] for x in params])
        for param in param_values:
            self.config[param.name] = param.value
        self.add_on_set_parameters_callback(self.params_cb)

        # subscribers
        self.create_subscription(
            Image, self.config["input_mask_topic"], self.image_cb, 1)
        self.create_subscription(
            Image, self.config["input_debug_image_topic"], self.debug_image_cb, 1)

        # publishers
        self.twist_pub = self.create_publisher(
            Twist, self.config["output_twist_topic"], 1)
        self.debug_lines_pub = self.create_publisher(Image, "blobs/lines", 1)

    def params_cb(self, params):
        for param in params:
            self.config[param.name] = param.value
        return SetParametersResult(successful=True)

    def debug_image_cb(self, ros_image):
        image_raw = self.bridge.imgmsg_to_cv2(ros_image)
        r = rows(image_raw)
        c = cols(image_raw)
        self.debug_image_raw = image_raw[int(self.config["top_crop_pct"] * r):r, 0:c]

    def image_cb(self, ros_mask):
        debug_image = None
        if self.debug_image_raw is not None:
            debug_image = self.debug_image_raw.copy()
            
        mask = self.bridge.imgmsg_to_cv2(ros_mask)
        lines = self.find_lines(mask, debug_image=debug_image)
        
        p0 = Vec(cols(lines)/2, rows(lines) - rows(lines)/10)
        force = center_lane(lines, p0, self.debug_image_raw)

        twist = Twist()
        twist.linear.x = float(self.config["speed"])
        twist.angular.z = float(self.config["turn_mult"] * force.x)

        
        self.twist_pub.publish(twist)
        
        if debug_image is not None:
            self.debug_lines_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))


    def find_lines(self, mask: cv.Mat, debug_image: cv.Mat = None):

        # crop image from top
        r = rows(mask)
        c = cols(mask)
        mask = mask[int(r * self.config["top_crop_pct"]):r, 0:c]

        # canny edge detection
        canny_image = cv.Canny(
            mask, self.config["canny_lower_thresh"], self.config["canny_upper_thresh"], apertureSize=5)

        # dilate edges
        dilation_size = (
            2 * self.config["line_dilation_size"] + 1, 2 * self.config["line_dilation_size"] + 1)
        dilation_anchor = (
            self.config["line_dilation_size"], self.config["line_dilation_size"])
        dilate_element = cv.getStructuringElement(
            cv.MORPH_RECT, dilation_size, dilation_anchor)
        dilated_image = cv.dilate(canny_image, dilate_element)

        line_mat = np.zeros_like(dilated_image)

        # find hough lines
        lines = cv.HoughLinesP(dilated_image,
                               rho=self.config["hough_rho"],
                               theta=0.01745329251,
                               threshold=self.config["hough_line_thresh"],
                               minLineLength=self.config["hough_min_line_len"],
                               maxLineGap=self.config["hough_max_line_gap"])

        # plot lines
        if lines is not None:
            for l in lines:
                l = l[0]  # (4,1) => (4,)
                diffx = l[0] - l[2]
                diffy = l[1] - l[3]

                if diffx == 0: # potential div by 0
                    continue

                slope = diffy / diffx

                if abs(slope) < self.config["line_min_slope"] or abs(slope) > self.config["line_max_slope"]:
                    continue

                diffx *= 5
                diffy *= 5

                l[0] -= diffx
                l[1] -= diffy
                l[2] += diffx
                l[3] += diffy

                cv.line(line_mat,
                        (l[0], int(l[1])),
                        (l[2], int(l[3])),
                        255, 5)
                
                if debug_image is not None:
                    cv.line(debug_image,
                        (l[0], int(l[1])),
                        (l[2], int(l[3])),
                        255, 5)

        return line_mat

def main(args=None):
    rclpy.init(args=args)
    node = BlobFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
