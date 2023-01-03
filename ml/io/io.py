import requests


def test_get(path):
    addr = 'http://localhost:8080'
    headers = {'content-type': 'test'}
    response = requests.get(f'{addr}{path}', headers=headers)
    return response


if __name__ == "__main__":

    response = test_get('/run-name/embed/10/3')
    print(response.text)
