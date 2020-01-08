# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-08 04:43
# pip3 install tensorflow-serving-api-gpu
import grpc
import tensorflow as tf
from tensorflow_core.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import hanlp
from hanlp.common.component import KerasComponent

tagger: KerasComponent = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN)
transform = tagger.transform
del tagger

inputs = [['商品', '和', '服务'],
          ['我', '的', '希望', '是', '希望', '和平']]

samples = next(iter(transform.inputs_to_dataset(inputs)))[0]
print(samples)

channel = grpc.insecure_channel('{host}:{port}'.format(host='localhost', port=8500))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'ctb5_pos_rnn_20191229_015325'
request.model_spec.signature_name = 'serving_default'
request.inputs['embedding_input'].CopyFrom(
    tf.make_tensor_proto(samples, dtype=tf.float32))
result = stub.Predict(request, 10.0)  # 10 secs timeout
print(result)
prediction = tensor_util.MakeNdarray(result.outputs['dense'])
print(prediction)

print(list(transform.Y_to_outputs(prediction, inputs=inputs)))
