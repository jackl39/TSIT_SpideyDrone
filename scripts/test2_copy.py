import sys
import traceback
from time import sleep
import tellopy
import av
import cv2
import numpy as np
import time
import threading


def handler(event, sender, data, **args):
    global flight_data
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        flight_data = data
        print(flight_data)

def video(container):
    global new_image
    global flight_data
    new_image = None

    try:
        # container = av.open(drone.get_video_stream()) This line issue an av.Error
        # skip first 300 frames
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                new_image = image
                if frame.time_base < 1.0 / 60:
                    time_base = 1.0 / 60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time) / time_base)

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)


def commands(drone):
    drone.takeoff()


def land(drone):
    drone.land()


def main():
    global new_image
    global flight_data
    drone = tellopy.Tello()
    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()

        drone.wait_for_connection(60.0)
        container = av.open(drone.get_video_stream())
        threading.Thread(target=video, args=[container]).start()
        while True:
            if new_image is not None:
                #threading.Thread(target=commands, args=[drone]).start()
                break
        sleep(5)
        while True:
            cv2.imshow('Tello', new_image)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                threading.Thread(target=land, args=[drone]).start()
                break
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()