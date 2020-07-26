# clothing_category_attribute_classification
> 시각장애인을 위한 웹 쇼핑몰 개발 중 이미지 분석 - 의복 종류, 디자인 판단

## Installation

-

## Usage example

이미지분석서버: 입력받은 의복 이미지에 대해 의복 종류 및 디자인을 판단하는 기능 수행

웹 서버: 웹 쇼핑몰을 동작시킴

이미지분석서버와 웹 서버는 TCP/IP 소켓으로 연결함

웹에서 상품 분석을 요청하면 이미지분석서버에서 상품 사진을 다운로드하고 분석을 수행함. 

분석 결과인 의복의 종류 및 색상을 웹에 반환함

- dataset_create_super.py

   - 데이터셋 생성을 위한 수퍼클래스

- dataset_create_category.py

   - 의복 종류에 대한 데이터셋 생성

- config.py

   - 관련 변수 및 모듈 선언

- selective_search.py

   - dataset_create_super.py에서 호출하여 사용함

- model.py

   - Inception Resnet V2에 SENet를 추가한 딥러닝 네트워크 모델

   - 하이퍼패러미터는 학습 후 정확도에 따라 조정해야 함

- train.py

   - 딥러닝 모델의 학습을 위한 코드
   
   - v0.3.0에서는 cifar10 데이터셋으로 테스트하려고 했지만 입력배열 차원의 불일치로 실패. 그냥 Deep Fashion 데이터셋을 사용하도록 함. 아직 경로지정 하지 않음


## Development setup

OS: Ubuntu 16.04

Framwork: 

Tensorflow 1.14.0

Keras 2.3.1

Dataset: Deep Fashion Database

## Release History

* 0.3.0
    * feat: generate deep learning model and proceed train code
    * 딥러닝 네트워크 모델 생성 완료
    * 딥러닝 네트워크 학습을 위한 코드 진행 중

* 0.2.0
    * feat: create category dataset
    * 의복종류(category)에 대한 데이터셋 생성 완료
    
* 0.1.0
    * first commit
    * 의복종류(category)에 대한 데이터셋 생성을 위한 코드

## Meta

김준혁 – wnsgur1198@naver.com

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
