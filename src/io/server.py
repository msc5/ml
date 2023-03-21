import json
import torch
import cv2
from flask import Flask, make_response

app = Flask(__name__)


@app.route('/<run>/embed')
def get_embeddings(run):

    file = open(f'results/{run}/embeddings.json')
    data = json.load(file)
    embeddings = data['embed']

    response = make_response(json.dumps(embeddings))
    response.headers.set('Content-Type', 'data')
    response.headers.set('Access-Control-Allow-Origin', '*')

    return response


@app.route('/<model>/<env>/obss/<episode>/<step>')
def get_obss(model, env, episode, step):

    data = torch.load(f'data/{model}/{env}/obss.pt')
    obs = (data[int(episode)][int(step)] * 255).permute(1, 2, 0)

    retval, buffer = cv2.imencode('.png', obs.numpy())
    obs_bin = bytes(buffer)

    response = make_response(obs_bin)
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Access-Control-Allow-Origin', '*')

    return response


app.run(host='localhost', port='8081')
