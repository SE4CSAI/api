def image_class_integration_test():
    import requests
    print("Testing image classifier with image upload")
    example_images= ['bird1.jpg','bug1.jpg','bird2.jpg','bird3.jpg','bug2.jpg']
    expected_results = [{'status_code': 200, 'predicted_label': 'bird', 'probs': [0.9985526204109192, 0.0014473225455731153]},
                        {'status_code': 200, 'predicted_label': 'bird',
                         'probs': [0.9999029636383057, 9.700276859803125e-05]},
                        {'status_code': 200, 'predicted_label': 'bird',
                         'probs': [0.9996663331985474, 0.000333732517901808]},
                        {'status_code': 200, 'predicted_label': 'bird',
                         'probs': [0.9996663331985474, 0.000333732517901808]},
                        {'status_code': 200, 'predicted_label': 'insect',
                         'probs': [3.7613301628880436e-06, 0.9999961853027344]}
                        ]
    for i, image in enumerate(example_images):
        headers = {
            'accept': 'application/json',
        }
        files = {
            'file': open(image, 'rb'),
        }
        response = requests.post('http://127.0.0.1:8000/upload_image', headers=headers, files=files)
        try:
            assert expected_results[i] == response.json()
        except AssertionError:
            print('Failed.')

def audio_class_integration_test():
    print("Testing audio classifier with file upload")
    import requests
    example_audio = ['Platypleuraplumosa.wav', 'Tettigoniaviridissima.wav']
    expected_results = [ {'status_code': 200, 'predicted_label': 'Platypleuraplumosa'},
                         {'status_code': 200, 'predicted_label': 'Tettigoniaviridissima'}
                         ]
    for i, audio in enumerate(example_audio):
        headers = {
            'accept': 'application/json',
        }
        files = {
            'file': open(audio, 'rb'),
        }
        response = requests.post('http://127.0.0.1:8000/upload_audio', headers=headers, files=files)
    try:
        assert expected_results[i] == response.json()
    except AssertionError:
        print('Failed.')

def from_url_image_class_integration_test():
    import requests
    print("Testing classifier from web url")
    example_urls = ['https://www.americanmeadows.com/media/wysiwyg/echinacea-goldfinch-mobile.jpg',
                     'https://www.protechpestcontrol.com.au/uploaded_files/BL_40045_03052020173143.jpg',
                    'https://www.protechpestcontrol.com.au/uploaded_files/BL_40045_0305202013143.jpg']
    expected_results = [{'status_code': 200, 'predicted_label': ['bird', {}, {}]},
        {'status_code': 200, 'predicted_label': ['insect', {}, {}]}
        , {'status_code': 400, 'message': 'file could not be opened'}]
    for i, url in enumerate(example_urls):
        headers = {
            'accept': 'application/json',
            'content-type': 'application/x-www-form-urlencoded',
        }
        params = {
            'request': url,
        }
        response = requests.post('http://127.0.0.1:8000/predict/image_model/from_url', params=params, headers=headers)
        #print("expected: {}, response : {}".format(expected_results[i],response.json()))
        try:
            assert expected_results[i] == response.json()
        except AssertionError:
            print('Failed.')

try:
    image_class_integration_test()
    audio_class_integration_test()
    from_url_image_class_integration_test()
    print("Passed.")
except:
    print("Error. Test failure or loading problem")