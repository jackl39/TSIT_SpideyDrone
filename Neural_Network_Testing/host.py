import numpy as np
import asyncio
import websockets
import cv2

async def receiver(websocket, path):
    while True:
        try:
            message = await websocket.recv()
            print(f"Received frame of size {len(message)} bytes")

            image_array = np.frombuffer(message, dtype=np.uint8)
            decoded_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Display the captured frame
            cv2.imshow('Webcam', decoded_img)

            # Wait for a short amount of time to display the frame
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed by client.")
            break

async def main():
    async with websockets.serve(receiver, "0.0.0.0", 8765):
        print("Server started.")
        await asyncio.Future()  # Runs forever

asyncio.run(main())