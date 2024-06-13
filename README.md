# Soccer Event Detection
In this project we aim to detect important events that have occured in a football match. It is trained on the SEV dataset that can be found here: https://drive.google.com/drive/folders/1jzt7g0KqFNTshEAau95aScPWin55g31E
We use a multi-classifier approach to train our model. Each classifier is an EfficientNetB0 with pre-trained weights that is fine-tuned for our classification. 
1. The First Classifier is used to detect whether the image is simply a random football match image or an event that has occured.
2. If the first classifier detects it as an event, it is fed to the second classifier that classifies it into one of 7 classes [Cards,Corner,Center,Left,Right,Substitute,Free-kick,Penalty,Tackle]
3. Finally, if the second classifier detects a card, it is fed to the third classifier that distinguishes whether it was a Yellow or Red card. 

References
1. https://drive.google.com/file/d/11IvTDSavQrdbyP99LZB2ycUywTgInRkh/view?usp=sharing
2. https://arxiv.org/abs/2102.04331
