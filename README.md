# MU_SC_for_VQA

This repository contains the implementation of the paper "Multi-User Semantic Communication for Visual Question Answering" by authors Zhiwei Xu, Yuxuan Song, Yongfeng Huang, and Shengyang Dai. The paper presents a novel approach to multi-modal classification in wireless communications using semantic communication.

The mac_network model used in the paper above was implemented with PyTorch, but this implementation code was written based on tensorflow 2.8 version. I tried to implement the code for the paper above, but still it doesn't work well. The model could be trained with the dataset but the accuracy was limited to be a certain percentage (~50%). 

이 리포지토리에는 "Multi-User Semantic Communication for Visual Question Answering" 논문의 구현이 포함되어 있습니다. 이 논문은 시맨틱 통신을 사용하는 무선 통신의 multi-modal classification에 대한 새로운 접근 방식을 제시합니다.

논문에 사용된 mac_network 모델은 PyTorch로 구현했으나 본 implementation code는 tensorflow 2.8 버전을 기준으로 작성되었습니다. 위 논문에 대한 코드를 구현했지만 학습이 잘 수행되지 않습니다. 현재, 모델은 데이터 세트로 학습되긴 하지만 정확도는 특정 비율(~50%)로 제한됩니다. 문제점을 구체적으로 파악하지 못한 상황입니다.

## Introduction

The paper addresses the problem of multi-user semantic communication (MU-SC) in the context of visual question answering (VQA). The authors propose a novel approach that leverages the power of deep learning to improve the performance of VQA tasks in multi-user scenarios. The proposed method is based on the idea of semantic communication, which involves transmitting high-level semantic information instead of raw data.

이 논문은 Visual Question Answering(VQA) 맥락에서 multi-user semantic communication(MU-SC)의 문제를 다룹니다. 저자는 multi-user 시나리오에서 VQA 작업의 성능을 향상시키기 위해 딥러닝을 활용하는 접근 방식을 제안합니다. 제안하는 방법은 raw data 대신 높은 수준의 semantic information을 전송하는 semantic communication의 아이디어를 기반으로 합니다.

## Methods

The authors use the CLEVR_v1.0 dataset for training the model. The model architecture consists of three main components: the Semantic Encoder, the Semantic Decoder, and the Semantic Channel. The Semantic Encoder is responsible for encoding the input data into semantic messages. In multi-user scenarios, Semantic Channels serve to mitigate channel effects. Finally, the Semantic Decoder decodes the received semantic messages to produce the final output (answer).

원 논문에서는 모델 학습을 위해 CLEVR_v1.0 데이터 세트를 사용합니다. 모델 아키텍처는 Semantic Encoder, Semantic Decoder 및 Semantic Channel의 세 가지 주요 구성 요소로 구성됩니다. Semantic Encoder는 입력 데이터를 Semantic Message로 인코딩하는 역할을 합니다. Semantic Channel은 multi-user 시나리오에서 채널 영향을 완화시키는 역할을 수행합니다. 마지막으로 Semantic Decoder는 수신된 semantic message를 디코딩하여 최종 출력 (answer)을 생성합니다.

## How To Use

### Setup

Create a virtual environment and install required packages with 'requirements.txt':

create environments
```
python -m venv (env_name)
```

install process
```
pip install -r requirements.txt
```

### Data Preprocessing

Download and extract CLEVR v1.0 dataset from http://cs.stanford.edu/people/jcjohns/clevr/

Run the following command to preprocessing. 

```
python preprocess_data.py --raw_data_dir (path)
```

(path) example: './CLEVR_v1.0/'


#### Model Training

An example command to start a training session:

```
python main.py --data_dir (data_path) --batch_size (num_batch_size) --num_epochs (num_epochs)
```

default
```
python main.py --data_dir data/ --batch_size 32 --num_epochs 10
```

Configurations can be modified in `main.py`. 

### Testing

An example command to start a test session with trained model and weights:

```
python test.py --train_model_dir (model_path) --snr (snr) --channel_type (channel_type)
```

default
```
python test.py --data_dir data/ --train_model_dir ./trained_model/ --snr 10 --channel_type awgn
```
