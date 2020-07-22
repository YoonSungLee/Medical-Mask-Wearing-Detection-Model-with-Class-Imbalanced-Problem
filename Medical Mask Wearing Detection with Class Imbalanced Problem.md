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

<img src='Image/Object_Detection01.PNG' width='100%'>

Aritifical Intelligence 중의 하나인 Machine Learning, Machine Learning 중의 하나인 Deep Learning, Deep Learning 중의 하나인 Computer Vision, Computer Vision 중의 하나인 Object Detection이 우리가 해결하고자 하는 문제이다. Object Detection 문제는 Image 내의 하나 또는 여러 객체에 대하여 Bounding Box를 추출하고, 해당 객체의 클래스를 분류하는 것을 목적으로 한다. Computer Vision에 관한 개괄적인 내용이 필요하다면 [Image Instance Segmentation with Mask R-CNN](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/[Report] Mask R-CNN.md) [3] 편을, Object Detection 문제 해결 원리에 대한 내용이 필요하다면 [Object Detection](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/[Report] Object Detection.md) [4] 편을 참고할 수 있다.

## 2.2. YOLO v3

github link

yolo v1, v2 paper review

## 2.3. Image Augmentation

github link

## 2.4. Focal Loss

github link paper review

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
[3] [Image Instance Segmentation with Mask R-CNN](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/[Report] Mask R-CNN.md), YoonSungLee<br>
[4] [Object Detection](https://github.com/YoonSungLee/Detection-Segmentation-Paper-Reivew-and-Report/blob/master/[Report] Object Detection.md), YoonSungLee