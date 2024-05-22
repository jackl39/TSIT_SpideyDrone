import numpy as np
import asyncio
import websockets
import cv2
import time
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from CNN_model import SpideyDroneNet
from torchvision.transforms import transforms

# Class names for classification
classNames = ["", "Spider Man", "Venom", "Doctor Octopus", "Green Goblin", "Carnage", "Electro", "Null"]

# Load the trained model
model = SpideyDroneNet(inputShape=1, hiddenUnits=40, outputShape=len(classNames))
model.load_state_dict(torch.load('Trained_Networks/drone_footage_100_epoch_95_pc.pt'))
model.eval()

# Define a function to preprocess the input image
def preprocess_image(image):
    # Convert to PIL image
    image = Image.fromarray(image)
    # Apply transformations to match the format used during training
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # Apply transformations
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

# Predict function
def predict(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Perform prediction
    with torch.no_grad():
        output = model(image)
    # Get the predicted class
    predicted_class = output.argmax(1).item()
    return classNames[predicted_class]

async def receiver(websocket, path):
    frame_count = 0
    start_time = time.time()
    frame_rate = 0  # Initialize frame rate

    while True:
        try:
            message = await websocket.recv()
            frame_count += 1

            image_array = np.frombuffer(message, dtype=np.uint8)
            decoded_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Calculate and display the frame rate every second
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:  # update every second
                frame_rate = frame_count / elapsed_time
                frame_count = 0
                start_time = current_time

            # Display the frame rate on the top right of the video feed
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(decoded_img, f"{frame_rate:.2f} FPS", (decoded_img.shape[1] - 150, 30), 
                        font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the captured frame
            cv2.imshow('Webcam', decoded_img)

            predicted_class = predict(decoded_img)
            print("Predicted class:", predicted_class)

            await websocket.send(predicted_class)

            # Wait for a short amount of time to display the frame
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed by client.")
            break
        except Exception as e:
            print(f"Error processing frame: {e}")
            break

async def main():
    async with websockets.serve(receiver, "0.0.0.0", 8765):
        print("Server started.")
        await asyncio.Future()  # Runs forever

if __name__ == "__main__":
    asyncio.run(main())