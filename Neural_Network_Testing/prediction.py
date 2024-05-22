import torch
from torchvision.transforms import ToTensor
from PIL import Image
from CNN_model import SpideyDroneNet
from torchvision.transforms import transforms
import cv2

classNames = ["", "Doctor Octopus", "Electro", "Green Goblin", "The Lizard", "Venom"]

# Load the trained model
model = SpideyDroneNet(inputShape=1, hiddenUnits=75, outputShape=len(classNames))
model.load_state_dict(torch.load('Trained_Networks/villains_1000_epoch_95pc.pt', map_location=torch.device('cpu')))
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

# Function to capture image from webcam and make prediction
def capture_and_predict():
    frame = cv2.imread("/home/jack/Documents/SpiderManHerosVillainsDataSet/combinedDataSet/venom_00018.jpg")
    predicted_class = predict(frame)    
    print("Predicted class:", predicted_class)

    # # Initialize the webcam
    # webcam = cv2.VideoCapture(0)  # 0 for the default webcam, you can change it if you have multiple webcams

    # # Check if the webcam is opened correctly
    # if not webcam.isOpened():
    #     print("Error: Couldn't open webcam")
    #     return

    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = webcam.read()

    #     # Display the captured frame
    #     cv2.imshow('Webcam', frame)

    #     # Press 'q' to quit capturing
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    #     # Make prediction on the captured frame
    #     predicted_class = predict(frame)
    #     print("Predicted class:", predicted_class)

    # # Release the webcam
    # webcam.release()
    # cv2.destroyAllWindows()

# Call the function to capture image and make predictions
capture_and_predict()