def image_class_integration_test():
    import requests
    print()
    print("Testing image classifier with image upload")
    example_images= ['bird1.jpg','bug1.jpg','bird2.jpg','bird3.jpg','bug2.jpg']
    for image in example_images:
        headers = {
            'accept': 'application/json',
        }
        files = {
            'file': open(image, 'rb'),
        }
        response = requests.post('http://127.0.0.1:8000/upload_image', headers=headers, files=files)
        print("expected: <string>predicted label, response : {}".format(response.json()['predicted_label']))

def audio_class_integration_test():
    print()
    print("Testing audio classifier with file upload")
    import requests
    example_audio = ['Platypleuraplumosa.wav', 'Tettigoniaviridissima.wav']
    for audio in example_audio:
        headers = {
            'accept': 'application/json',
        }
        files = {
            'file': open(audio, 'rb'),
        }
        response = requests.post('http://127.0.0.1:8000/upload_audio', headers=headers, files=files)
        print("expected: <string>predicted label, response : {}".format(response.json()['predicted_label']))

def from_url_image_class_integration_test():
    import requests
    print()
    print("Testing classifier from web url")
    example_urls = ['https://www.americanmeadows.com/media/wysiwyg/echinacea-goldfinch-mobile.jpg',
                     'https://www.protechpestcontrol.com.au/uploaded_files/BL_40045_03052020173143.jpg']
    for url in example_urls:
        headers = {
            'accept': 'application/json',
            'content-type': 'application/x-www-form-urlencoded',
        }
        params = {
            'request': url,
        }
        response = requests.post('http://127.0.0.1:8000/predict/image_model/from_url', params=params, headers=headers)
        print("expected: <string>predicted label, response : {}".format(response.json()['predicted_label'][0]))
try:
    image_class_integration_test()
    audio_class_integration_test()
    from_url_image_class_integration_test()
    print("Passed.")
except:
    print("Error. Test failure or loading problem")