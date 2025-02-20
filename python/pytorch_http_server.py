#!/usr/bin/env python3
from flask import Flask, request, jsonify, Response
from deep_q_network import DeepQNetwork
import argparse
import uuid
import json
import numpy as np
import torch as T

########## Log Level adjsutment ###########
import logging
logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)

########## App setup ##########
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
instances = {}
save_frequency = 100
next_save = {}

########## API route definitions ##########
@app.route('/models', methods=['POST'])
def model_create():
    """
    Create a new model
    """
    return jsonify(modelId = 4711)

@app.route('/instances', methods=['POST'])
def instance_create():
    """
    Create an instance of the given model
    """
    guid = str(uuid.uuid4())
    input = np.zeros((4, 84, 84)).shape
    instances[guid] = DeepQNetwork(0.001, 6, guid, input, 'save_model/')
    print('create_instance')
    instances[guid].save_checkpoint()
    next_save[guid] = save_frequency
    return jsonify(instanceId = guid)
    
@app.route('/instances/<instanceId>', methods=['GET'])
def retrieve_instance(instanceId):
    if not instanceId in instances:
        input = np.zeros((4, 84, 84)).shape
        instances[instanceId] = DeepQNetwork(0.001, 6, instanceId, input, 'save_model/')
        instances[instanceId].load_checkpoint()
        next_save[instanceId] = save_frequency
    return jsonify(instanceId = instanceId)
    
@app.route('/instances/<instanceId>/predictions', methods=['POST'])
def predict(instanceId):
    model = instances[instanceId]
    observation = request.get_json()
    observation = np.array(observation)
    ## observation = np.transpose(observation, (2, 0, 1))
    state = T.tensor([observation],dtype=T.float).to(model.device)
    actions = model.forward(state)
    actions = actions.cpu().tolist()[0]
    return jsonify(actions)
    
    
@app.route('/instances/<instanceId>/trainings', methods=['POST'])
def training(instanceId):
    model = instances[instanceId]
    traingsData = request.get_json()
    model.optimizer.zero_grad()
    observation = [entry["observation"] for entry in traingsData]
    observation = np.array(observation)
    ## observation = np.transpose(observation, (0, 3, 1, 2))
    states = T.tensor(observation, dtype=T.float).to(model.device)
    indices = np.arange(len(traingsData))
    actions = [entry["action"] for entry in traingsData]
    q_pred = model.forward(states)[indices, actions]
    q_target = [entry["output"] for entry in traingsData]
    q_target = np.array(q_target)
    q_target = q_target[indices, actions]
    q_target = T.tensor(q_target, dtype=T.float).to(model.device)
    loss = model.loss(q_target, q_pred).to(model.device)
    print('loss: ', loss)
    loss.backward()
    model.optimizer.step()
    
    if next_save[instanceId] <= 0:
        model.save_checkpoint()
        next_save[instanceId] = save_frequency
    else:
        next_save[instanceId] -= 1
    
    return Response(status = 204)
    
@app.route('/instances/<instanceId>/copies', methods=['PUT'])
def copy_params(instanceId):
    sourceInstanceId = request.get_json().get('instanceId', None)
    target = instances[instanceId]
    source = instances[sourceInstanceId]
    target.load_state_dict(source.state_dict())
    return Response(status = 204)
    
@app.route('/examples/84x84x1', methods=['GET'])    
def example_picture():
    array_3d = np.random.rand(84, 84, 1)
    return jsonify(array_3d.tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a Gym HTTP API server')
    parser.add_argument('-l', '--listen', help='interface to listen to', default='127.0.0.1')
    parser.add_argument('-p', '--port', default=5001, type=int, help='port to bind to')

    args = parser.parse_args()
    print('Server starting at: ' + 'http://{}:{}'.format(args.listen, args.port))
    app.run(host=args.listen, port=args.port)
