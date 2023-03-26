from django.shortcuts import render
from django.http import HttpResponse
from django.urls import path
from django.http import HttpResponseBadRequest
from bson import json_util
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.middleware.csrf import get_token
import json
import joblib
import cv2
import numpy as np
from .connectdb import connectDb

# This function renders the main page
def index(request):
    return render(request, 'MainPage.html')

# This function returns a javascript file
def my_javascript_view(request):
    # create an HTTP response with the content type of javascript
    response = HttpResponse(content_type='application/javascript')
    # write the content of the javascript file to the response
    response.write(open('js/MainPage.js', 'rb').read())
    return response

# This function handles file uploads
def upload_file(request):
    if request.method == 'POST' and request.FILES:
        # get the uploaded file from the request
        uploaded_file = request.FILES['image']
        # pass the request and uploaded file to the handle_uploaded_file function
        return handle_uploaded_file(request, uploaded_file)
    else:
        # return an error response if the request is invalid
        return HttpResponseBadRequest("Invalid request")

# This function handles the uploaded file and returns a prediction result
def handle_uploaded_file(request, uploaded_file):
    # open a new file with the name of the uploaded file in the 'reactapp/build/static/media' directory
    with open('reactapp/build/static/media/' + uploaded_file.name, 'wb+') as destination:
        # write the file data in chunks to the new file
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    # return the prediction result
    return predict_fruit(request)
# This function preprocesses the image
def preprocess_image(image_file):
    data = []
    
    # Read image file
    img = cv2.imread('c:/Users/Ani/ImgRecProj/reactapp/build/static/media/'+ image_file.name)
    
    # Check if image dimensions are valid
    if img.shape[0] > 0 and img.shape[1] > 0:
        
        # Add color features
        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]
        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)
        color_features = [red_mean, green_mean, blue_mean]

        # Resize and convert to grayscale
        img = cv2.resize(img, (100, 100))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract HOG features
        win_size = (100, 100)
        block_size = (20, 20)
        block_stride = (10, 10)
        cell_size = (10, 10)
        nbins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(gray).flatten()

        # Combine color and HOG features
        all_features = np.concatenate((color_features, hog_features))

        # Append the combined features to the data list
        data.append(np.array(all_features))
    
    # Convert data list to a numpy array and return it
    return np.array(data)
# This function predicts the type of the fruit
def predict_fruit(request):
    if request.method == 'POST':
        # Get the uploaded image
        uploaded_file = request.FILES['image']

        # Load the input image and preprocess it
        image = preprocess_image(uploaded_file)

        # Make a prediction using the trained model
        clf = joblib.load('svm_model.joblib')
        prediction = clf.predict(image)

        # Get nutritional data for predicted fruit from MongoDB
        fruit = prediction.tolist()[0]
        collection = connectDb()
        for doc in collection.find({"name": fruit}):
            nutritional_data = json.loads(json.dumps(doc, default=str))

        # Return the prediction and nutritional data as a JSON response
        response_data = {
            'prediction': fruit,
            'nutritional_data': nutritional_data
        }
        return JsonResponse(response_data)

    # If the request method is not POST, return an error response
    return JsonResponse({'error': 'Invalid request method.'}, status=400)

def get_csrf_token(request):
    """
    Returns a JSON response containing the CSRF token for the given request.
    """
    return JsonResponse({'csrfToken': get_token(request)})
