/dataset : bed, adult, child
/exported_models : 학습 시킨 model
/models/ssd_mobilenet : checkpoint, configuration file
/ssd_mobilenet : pretrained model
/tensorflow1 : tflite 관련 tools
/tools : 학습, 모델 변환시 필요한 파일

# capston (capston)
tensorflow == 1.15.0
python == 3.7.13
numpy == 1.19.5
- model : ssd_mobilenet
- batch_size = 24
- num_steps = 100000
- checkpoint path == /data/siwon/22/capston/models/ssd_mobilenet/model.ckpt
- 소요시간 : about 28 hours

# efficientdet (tf)
tensorflow == 2.8.0
python == 3.8.12
- model : 
- batch_size = 128
- num_steps = 300000
- checkpoint path = 
- 소요시간 : 

fine_tune_checkpoint_type = "detection"
--------------------------------------------------------------------------------------------
pbtotflite.py 
- frozen_inference_graph.pb to tfltie
- input_arrays : 시작 노드 이름
- output_arrays : 끝 노드 이름

pb_tensorboard.py 
--------------------------------------------------------------------------------------------


