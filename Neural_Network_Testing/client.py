import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import websocket
import threading

# Initialize CV Bridge
bridge = CvBridge()

def on_image_received(data):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        # Encode the image as a JPEG
        _, jpeg_image = cv2.imencode('.jpg', cv_image)

        # Ensure the WebSocket connection is open before sending
        if ws.sock and ws.sock.connected:
            # Convert numpy array to bytes and send
            ws.send(jpeg_image.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
        else:
            print("WebSocket connection is closed. Cannot send data.")

    except CvBridgeError as e:
        print(e)
    except Exception as ex:
        print(f"Failed to process or send image: {ex}")


def on_websocket_message(ws, message):
    print("Received message from WebSocket: " + message)

def on_websocket_error(ws, error):
    print("WebSocket error: " + str(error))

def on_websocket_close(ws, close_status_code, close_msg):
    print("WebSocket closed with status {} and message {}".format(close_status_code, close_msg))

def on_websocket_open(ws):
    print("WebSocket connection established")

if __name__ == '__main__':
    # Initialize ROS
    rospy.init_node('image_to_websocket', anonymous=True)
    rospy.Subscriber("/tello/camera/image_raw", Image, on_image_received)
    
    # Setup WebSocket connection
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://10.70.139.213:8765",
                                on_message=on_websocket_message,
                                on_error=on_websocket_error,
                                on_close=on_websocket_close)
    
    ws.on_open = on_websocket_open
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        ws.close()