# Import necessary libraries
import os
import requests
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials


# Set up Azure Cognitive Services credentials
key = os.environ["COGNITIVE_SERVICES_KEY"]
endpoint = os.environ["COGNITIVE_SERVICES_ENDPOINT"]
credential = CognitiveServicesCredentials(key)


# Instantiate the Face client
client = FaceClient(endpoint, credential)

# Define the image URL
image_url = "https://www.example.com/image.jpg"


# Analyze the image using Azure Cognitive Services
response = requests.get(image_url)
faces = client.face.detect_with_url(url=image_url, detection_model="detection_03")


# Print out the results
print("Detected {} faces in the image:".format(len(faces)))
for face in faces:
    print("- Face ID: {}".format(face.face_id))
    print("  Gender: {}".format(face.face_attributes.gender))
    print("  Age: {}".format(face.face_attributes.age))
    print("  Emotion: {}".format(face.face_attributes.emotion.happiness))
