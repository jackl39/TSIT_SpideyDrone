#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import websocket
import threading

class Drone2CNN:
    def __init__(self):        
        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Setup ROS Subscriber and Publisher
        self.image_subscriber = rospy.Subscriber("/droneCam/Villain", Image, self.on_image_received)
        self.string_publisher = rospy.Publisher("/Villain", String, queue_size=1)
        
        # Setup WebSocket connection
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp("ws://10.70.139.213:8765",
                                         on_message=self.on_websocket_message,
                                         on_error=self.on_websocket_error,
                                         on_close=self.on_websocket_close)
        self.ws.on_open = self.on_websocket_open
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def on_image_received(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Encode the image as a JPEG
            _, jpeg_image = cv2.imencode('.jpg', cv_image)

            # Ensure the WebSocket connection is open before sending
            if self.ws.sock and self.ws.sock.connected:
                # Convert numpy array to bytes and send
                self.ws.send(jpeg_image.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
            else:
                print("WebSocket connection is closed. Cannot send data.")
        except CvBridgeError as e:
            print(e)
        except Exception as ex:
            print(f"Failed to process or send image: {ex}")

    def on_websocket_message(self, ws, message):
        print("Received message from WebSocket: " + message)

    def on_websocket_error(self, ws, error):
        print("WebSocket error: " + str(error))

    def on_websocket_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed with status {} and message {}".format(close_status_code, close_msg))

    def on_websocket_open(self, ws):
        print("WebSocket connection established")

    def run(self):
        rospy.spin()
        
    def shutdown(self):
        print("Shutting down")
        self.ws.close()
