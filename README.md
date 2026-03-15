# CPS-843: Computer Vision - American Sign Language (ASL) Recognition
CPS-843 - Term project (group 69)

# The dataset we used 
https://www.kaggle.com/models/sayannath235/american-sign-language/

# How to run
- open the project preferably in a virtual or conda environment
- install dependies via: pip install -r requirements.txt
- should run after this, may take up to a couple of minutes to load the program

# Individual Contribution
Annabel Chao 
- Camera Implementation & reading frames
- model image processing (224x224, normalization)
- text adding, resolution changing functionality 
- caching optimization
- (attempted) asyncronous multithreading
- code review
- report contribution

Thao Nguyen
- filter preprocessing for model
- prediction confidence percentage scale
- (attempted) moving J and Z model detection (mediapipe)
- (attempted) algorithm speed and prediction optimization
- video demonstration of code
- report contribution

Jessica Zhu
- extra dimension for model processing functionality 
- code error handling
- (attempted) moving J and Z model detection
- code review
- IEEE report 

Julia Khong
- picked out keras ASL model
- model precision testing & application user testing
- drawn box tracking hand movement (mediapipe)
- (attempted) precise hand joint movement (UI)
- Created the Installation guide
- IEEE report



## Contributor Information 

| Annabel Chao  | Thao Nguyen | Jessica Zhu  | Julia Khong |
| ------------- | ------------- | ------------- | ------------- |
| annabel.chao@torontomu.ca  | thao2.nguyen@torontomu.ca  | jessica.zhu@torontomu.ca  | julia.khong@torontomu.ca  |
