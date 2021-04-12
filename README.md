# Dog Breed Prediction

### Table of Contents

1. [Installation](#Installation)
2. [Project Motivation](#Motivation)
3. [File Descriptions](#descriptions)
4. [Instructions](#Instructions)
5. [Results](#Results)


### Installation <a name='Installation'>:
The requierement for this project is as following:- <br>
 1. Python 3.8 .
 2. pandas.
 3. openv-python.
 4. torch.
 5. tensorflow.
 6. tqdm.
 7. PIL.
 
The project root directory must be on pythonpath
 
 
### Project Motivation <a name='Motivation'>:
The idea here is to train a CNN model on images of dogs and predict the breed of a dog by
 giving only the image file. This will help a user to identify the breed of a dog. Fun part
 if it identifies a person as dog it will predict the breed for that too :).
 

### File Descriptions <a name='descriptions'>:
The READEME.md file is a guide through the project.<br>
The dog_app.py file is for model training.<br>
The dog_app.html file is a notebook of the training and visualization, it describes how the model is trained.<br>
The predict.py file is the executable for running the program on unseen images.<br>
The saved_model directory contains the best fitted model on the training data test accuracy.<br>
The LICENSE file is present to specify access.<br>

### Instructions <a name='Instructions'>:
1. Run the following commands in the project's root directory to set up your model and visualize the results in notebook.

    - To train and save the model. You can skip this as trained data is not in the repository.<br>
        `python dog_app.py`
    - To run the webapp run below command.<br>
        `python webapp/run.py>`
        
    - Open http://0.0.0.0:3001 on your computer and upload picture of a dog to predict it's breed.


### Results <a name='Results'>:
In the notebook all the steps are mentioned. The training data and accuracy is given.
We are able to identify breed of a dog with an acceptable accuracy.

### Future Improvement 
The model can be improved with more data.

### Github details
Github Repo:- https://github.com/regarmukesh3g/DogBreedClassifier
Clone Url:- https://github.com/regarmukesh3g/DogBreedClassifier.git
