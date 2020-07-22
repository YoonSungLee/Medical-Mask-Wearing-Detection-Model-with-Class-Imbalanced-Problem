# Medical Mask Wearing Detection Model with Class Imbalanced Problem

---

# Abstract

최근 코로나 바이러스 문제로 인해 해당 질병의 전파 속도가 증가하고 있다. 이에 따라 정부는 마스크 착용, 손 씻기, 사회적 거리 두기 등을 권고하고 있는 상황이다. 마스크 착용을 통해 비말 감염을 예방할 수 있기 때문에 마스크를 착용하지 않는 사람은 특정 건물이나 대중교통을 이용하는 데에 제한을 두어 마스크 착용의 의무화를 시행하고 있다. 하지만 그럼에도 불구하고 마스크를 착용하지 않는 사람들을 쉽게 볼 수 있고, 이들을 일일히 관리하기란 힘든 일이다. 이러한 상황에서 마스크 착용에 대한 탐지를 자동화하는 시스템은 인력 낭비를 줄이고 효율적인 관리를 기대할 수 있을 것이다. 따라서 해당 프로젝트는 'Medical Mask Wearing Detection Model' 에 대한 내용을 담고 있다. Dataset 수집부터, Model 선택, Trasfer Learing에 관한 전반적인 내용을 다룬다. 특히 수집된 Dataset은 마스크를 착용한 사람(with_mask)에 비해, 착용하지 않은 사람(without_mask)이나 잘못 착용한 사람(mask_weared_incorrect)의 비율이 작은 문제를 가지고 있다. 따라서 이러한 Class Imbalanced Problem에 대한 고민과 문제를 해결하는 내용 또한 다루고 있다. github code는 아래 링크를 통해 확인할 수 있다.

* https://github.com/YoonSungLee/PyTorch-YOLOv3

---

# 1. Introduction

코로나 바이러스가 더 이상 퍼지지 않도록 기여할 수 있는 우리의 가장 간단하고도 중요한 역할은 '마스크 착용' 이다. 길거리를 돌아다니면 쉽게 마스크 착용에 대한 포스터를 볼 수 있으며, 정부에서도 대중교통을 이용하기 위해 마스크 착용을 의무화하는 정책을 펼치고 있다. 나아가 기업에서도 마스크 착용 의무화를 위해 여러 서비스를 실시하고 있다. 예를 들어 LG CNS의 경우, 서울 마곡 본사 일부 출입 게이트에서 'AI 출입통제 서비스'를 활용하여 마스크를 착용한 직원만 통과시키도록 한다.[1]<br>
이는 서비스명에서도 알 수 있듯이 AI를 활용한 얼굴인식 기술이다. Dataset에서 유의미한 패턴을 찾아내는 Machine Learning Method 중에 하나인 Deep Learning Method는 점점 우리의 삶에 자리매김하고있다. Computer Vison, Natural Language Processing, Reinforcement Learning, GAN 등 다양한 분야에서 활발히 연구가 진행중이다. 특히 이번 프로젝트에서 알아볼 내용은 Computer Vision으로써, Image Dataset에서 어떻게 유의미한 정보를 찾아내는지에 관해 살펴본다.<br>
Deep Learning Method는 각 분야마다 깊숙히 그리고 활발히 연구 중에 있기 때문에, 분야가 세분화되어있다. 특히 이번 프로젝트에서 확인할 Computer Vision에 대한 분야 내에서도 Classification, Object Detection, Segmentation, Instance Segmentation 등 해결하고자 하는 문제에 따라 분류되어 있다고 할 수 있다. 이 중에서 '마스크 착용'을 확인하는 솔루션을 제시하기 위해서 어떤 문제로 분류할 수 있을까? Image 안에 여러 사람이 존재할 수 있기 때문에, 단순히 Classification을 적용하기에는 무리가 있다. 그렇다고 Segmentation이나 Instance Segmentation은 객체 추출을 통한 정교한 위치의 결과값까지 찾아주기 때문에, 해당 문제를 해결하는 데에 불필요한 정보를 얻는다. 결국 해당 문제는 사람에 대한 위치 정보를 Bounding Box를 통해 찾고, 동시에 그 사람이 마스크를 착용했는지에 대한 여부를 분류하면 되기 때문에, Object Detection 문제로 귀결된다고 할 수 있다. 따라서 해당 프로젝트는 마스크 착용 문제를 Object Detection 문제로 정의하고, 그에 따른 Dataset 구성, Model 설정 등의 업무를 수행한다.<br>
Object Detection 문제를 해결하기 위한 모델은 굉장히 많으며 각각의 특성을 가지고 있다. 해당 프로젝트를 수행하기 위해 조사한 바에 따르면, Detection 작동 원리에 따라 크게 두 가지 종류의 모델로 구분할 수 있다. 하나는 One-Stage Detection Model이고, 다른 하나는 Two-Stage Detection Model이다. 해당 프로젝트는 마스크 착용 여부를 실시간으로 파악하기를 원한다. 이에 따라 모델의 정확도 뿐만 아니라 실시간으로 수행할 정도의 속도를 가진 모델을 필요로 한다. 따라서 일반적으로 정확도 면에서 조금 떨어지지만 실시간으로 Detection을 수행할 수 있는 One-Stage Detection Model을 활용한다. 특히, 대중적이며 참고할 자료가 많은 YOLO v3 Model을 Custom Dataset에 Transfer Learning을 통해 활용하기로 한다.<br>
Deep Learning 모델을 학습시키는 데에 있어서, Dataset의 Class Balance 또한 중요한 요소이다. 단순한 예를 들어, 강아지와 고양이를 분류하기 위해 Image를 수집했다고 가정해보자. 강아지 Image가 90장, 고양이 Image가 10장 들어있는 Dataset을 모델이 잘 학습할 수 있을까? 모든 Training Dataset에 대하여 강아지라고만 예측해도 해당 모델의 성능(accuracy)은 90%이기 때문에, 이는 Test Dataset에 대해 강아지라고만 예측할 것이다. 따라서 Class의 비율을 적절히 조절하는 것 또한 모델의 성능을 높이는 요소 중의 하나라고 할 수 있다. 해당 프로젝트를 수행할 때 사용한 Dataset 또한 Class Imbalanced Problem을 가지고 있다. 따라서 이 문제를 어떻게 접근했는지, 그리고 어떻게 해결했는지에 대한 내용도 제시한다.<br>해당 프로젝트는 이전에 같은 주제로 한 번 수행한 바 있다.[2] 이전 프로젝트에서는 Model에 대한 이해와 Transfer Learning을 어떻게 수행하는지를 중점으로 두었다고 한다면, 이번 프로젝트는 더 나아가 Model의 코드를 이해 및 수정하고 더 나은 결과를 얻기 위해 연구를 수행하는 과정이 포함된다. 결론적으로 해당 프로젝트는 위와 같은 고민들을 거쳐, Object Detection의 성능 평가 기준 중의 하나인 mAP를 기준으로 **0.757417**이라는 비교적 높은 결과를 얻어냈다. 이에 대한 고찰과 앞으로 나아가야 할 연구방향에 대해서 언급함으로써 해당 프로젝트에 대한 설명을 마무리하도록 하겠다.

# 2. Related Work

## 2.1. Object Detection

<img src='Image/Object_Detection01.png' width='100%'>

Aritifical Intelligence 중의 하나인 Machine Learning, Machine Learning 중의 하나인 Deep Learning, Deep Learning 중의 하나인 Computer Vision, Computer Vision 중의 하나인 Object Detection이 우리가 해결하고자 하는 문제이다. Object Detection 문제는 Image 내의 하나 또는 여러 객체에 대하여 Bounding Box를 추출하고, 해당 객체의 클래스를 분류하는 것을 목적으로 한다. Computer Vision에 관한 개괄적인 내용이 필요하다면 [Report: Mask-R-CNN](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Report_Mask-R-CNN.md) [3] 편을, Object Detection 문제 해결 원리에 대한 내용이 필요하다면 [Report: Object Detection](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Report_Object-Detection.md) [4] 편을 참고하는 것을 권한다.

## 2.2. YOLO v3

<img src='Image/YOLOv301.png' width='100%'>

Object Detection 모델 중의 하나인 YOLO v3는 One-Stage Detection Model 로써, Two-Stage Detection Model에 비해 빠른 속도를 보인다. 기존 YOLO 버전에 비해 3군데의 resolution에서 좀 더 많은 Bounding Box를 추출하기 때문에, 정확도 향상과 작은 물체에 대한 탐지 능력 향상 등의 효과를 얻을 수 있었다. 해당 모델에 대하여 어떻게 작동하는지에 대한 내용이 필요하다면 [How to Perform Object Detection With YOLOv3 in Keras](https://github.com/YoonSungLee/Detection-Segmentation-Project/blob/master/How_to_Perform_Object_Detection_With_YOLOv3_in_Keras.ipynb) [5] 편을 참고하는 것을 권고한다. 이 외에도 YOLO의 다른 버전에 대한 모델을 이해하고 싶다면, 논문 리뷰인 [Review: You Only Look Once: Unified, Real-Time Object Detection](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Paper_Review_YOLOv1.md) [6]과 [Review: YOLO9000: Better, Faster, Stronger](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Paper_Review_YOLO9000.md) [7]에서 그 내용을 확인할 수 있다.

## 2.3. Image Augmentation

Image Augmentation은 기존 데이터를 부풀리는 기법이다. 이를 통해 모델이 기존 데이터의 패턴 뿐만 아니라 부풀린 데이터에 의해 새로운 패턴을 발견하여, 모델의 성능을 좀 더 robust하게 만드는 효과를 볼 수 있다. Image Augmentation의 방법으로는 좌우 반전, 크롭(자르기), 밝기 조절, 회전, 이동, 확대 및 축소, 랜덤 노이즈 등 생각할 수 있는 모든 방법들이 해당한다. [9]<br>
Image Augmentation은 Class Imbalanced Problem을 해결하는데에도 사용할 수 있다. 앞서 언급한 예를 다시 들자면, 강아지 Image가 90장, 고양이 Image가 10장 포함된 Dataset을 가지고 있다고 해보자. 이 상태에서 모델에 학습시키면(물론 성능이 좋은 모델은 좋은 결과를 내겠지만), Class Imbalanced Problem 때문에 모델을 제대로 학습시킬 수 없다. 특히 Object Detection 모델의 성능 평가 기준 중의 하나인 mAP를 적용해보면, Class별로 AP의 차이가 큰 결과를 얻을 것이다. 이러한 경우에 고양이의 사진 10장을 Image Augmentation을 사용하면 어떻게 될까? 위에서 언급한 Image Augmentation의 방법을 적용하여 고양이의 사진을 10장에서 100장으로 부풀린다면, Class Imbalanced Problem이 해결되어 Class 별 AP가 일정한 결과를 기대할 수 있다.<br>
다만 Image Augmentation을 사용함에 있어서 고민해야 할 것은, hyperparameter setting이다. 위에서 언급한 여러 기법들 외에도 무수히 많은 Image Augmentation 기법들이 존재하며, 특정 기법 내에서도 각도, 거리 등 기법을 적용할 정도에 대한 hyperparameter를 직접 설정해주어야 한다. 따라서 어떤 Augmentation 기법을 사용하면 좋을지, 그리고 어떻게 hyperparameter를 setting해야할지를 주어진 문제에 따라 적용하는 skill이 필요하다.<br>
Image Augmentation을 코드로 확인하고 싶다면 github [Image Augmentation Tool](https://github.com/YoonSungLee/Image_Augmentation_Tool) [8] 을 참고할 수 있다.

## 2.4. Focal Loss

<img src='Image/Focal_Loss01.png' width='100%'>

Class Imbalanced Problem을 해결할 수 있는 또 하나의 방법은 'Focal Loss for Dense Object Detection' 논문을 참고하면서 그 아이디어를 얻을 수 있다.<br>
Multi Class Image를 Classification하기 위해 사용하는 일반적인 loss function은 Cross Entropy Function이다. 이는 각 Class마다 probability의 log를 취해서 나온 결과의 음수값을 loss 값으로 설정함으로써, Class를 잘 맞추면(probability가 1에 가까워지면) loss 값이 작아지고 Class를 못 맞추면(probability가 0에 가까워지면) loss 값이 커지는 역할을 한다.<br>
문제는 Dataset의 Class가 Imabalance 할 때 발생한다. 다시 한 번 강아지와 고양이 Dataset을 예로 들어보겠다. 만약 일반적인 Cross Entropy Function을 사용한다면 어떤 문제가 발생할까? 학습 초기에 강아지 Image 하나에 대한 loss값과 고양이 Image 하나에 대한 loss값은 같을 것이다. 하지만 total loss값은 100장의 Image에 대한 loss값을 모두 합치므로 결국 강아지에 대한 loss값이 고양이에 대한 loss값보다 9배나 커지게 된다. 즉, 모델은 강아지에 대한 loss값을 줄여서 total loss값을 줄이는 방향으로 학습이 진행되기 때문에 결국 강아지는 잘 맞추고 고양이는 못 맞추는 문제가 발생하게 된다. 고양이 Image 하나에 대한 loss값이 아무리 크더라도, 절대적인 loss값들의 수가 작기 때문에 총합으로 봤을 때 밀리는 것이다.<br>
따라서 Focal Loss는 이러한 문제를 해결하기 위해 위의 그림과 같이 기존 Cross Entropy Function에 weight를 곱해준다. 이는 (1 - pt)^gamma의 값으로써, 만약 gamma의 값이 2라고 한다면 특정 Class를 잘 맞출수록(probability가 1에 가까워지면) (1-pt)^gamma의 값이 0에 가까워진 상태로 곱해진다. 즉, 잘 맞춘 Class에 대하여 down weight의 역할을 수행한다. 반대로 특정 Class를 못 맞출수록(probability가 0에 가까워지면) 곱해주는 값의 범위가 (물론 1보다는 작아지지만) 조금밖에 작아지지 않기 때문에, 해당 Class의 loss 값이 상대적으로 커지는 역할을 한다. 이것이 바로 Focal Loss의 핵심이라고 할 수 있다.<br>
해당 내용에 대하여 구체적으로 알기를 원한다면, 논문 리뷰인 [Review: Focal Loss for Dense Object Detection](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Paper_Review_Focal_Loss_for_Dense_Object_Detection.ipynb) [10]을 참고하는 것을 권고한다.

# 3. Experiments

## 3.1. Dataset

License Problem

## 3.2. Transfer Learning

Environment

## 3.3. Evaluation

mAP

10 experiments

# 4. Conclusion

# 5. Discussion

Image Augmentation Guideline
Recall

# 6. Reference

[1] ["마스크 착용한 분만 문 열어 드립니다"](https://blog.lgcns.com/2216), LG CNS<br>
[2] [Mask-Wearing-Detection-Project-with-YOLOv3](https://github.com/YoonSungLee/Mask-Wearing-Detection-Project-with-YOLOv3), YoonSungLee<br>
[3] [Report: Mask-R-CNN](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Report_Mask-R-CNN.md), YoonSungLee<br>
[4] [Report: Object Detection](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Report_Object-Detection.md), YoonSungLee<br>
[5] [How to Perform Object Detection With YOLOv3 in Keras](https://github.com/YoonSungLee/Detection-Segmentation-Project/blob/master/How_to_Perform_Object_Detection_With_YOLOv3_in_Keras.ipynb), YoonSungLee<br>
[6] [Review: You Only Look Once: Unified, Real-Time Object Detection](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Paper_Review_YOLOv1.md), YoonSungLee<br>
[7] [Review: YOLO9000: Better, Faster, Stronger](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Paper_Review_YOLO9000.md), YoonSungLee<br>
[8] [Image Augmentation Tool](https://github.com/YoonSungLee/Image_Augmentation_Tool), YoonSungLee<br>
[9] [Data Preprocessing & Augmentation](https://nittaku.tistory.com/272), nittaku<br>
[10] [Review: Focal Loss for Dense Object Detection](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/Paper_Review_Focal_Loss_for_Dense_Object_Detection.ipynb), YoonSungLee