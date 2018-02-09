# Assignements for Stanford course cs231n Convolutional Neural Networks for Visual Recognition
http://cs231n.stanford.edu/2016/syllabus
##  Notebooks describe tasks, contain the code, references to the code in other modules.

## Assignement 1:
### How to run code
#### 1. Install and activate virtual environment:
```bash
cd assignment1
pip2 install virtualenv
virtualenv .env
source .env/bin/activate
```
#### 2. Install dependencies:
```
pip2 install -r requirements.txt
```
#### 3. Download data:
```bash
cd cs231n/datasets
./get_datasets.sh
cd assignment1/cs231n
```
#### 4. Run jupyter:
```bash
cd assignment1
jupyter notebook
```
### Note: for Mac OS X run ./start_ipython_osx.sh instead of jupyter notebook
### 5. open browser at localhost:8888/
### 6. Run notebooks.

## Assignement 2:
### How to run code
#### 1. Install and activate virtual environment:
```bash
cd assignment2
pip2 install virtualenv
virtualenv .env
source .env/bin/activate
```
#### 2. Install dependencies:
```
pip2 install -r requirements.txt
```
#### 3. Download data:
```bash
cd cs231n/datasets
./get_coco_captioning.sh
./get_tiny_imagenet_a.sh
./get_pretrained_model.sh
```
#### 4. Build Cython modules:
```
cd assignment2/cs231n
python2 setup.py build_ext --inplace # build C code
```
#### 5. Run jupyter:
```bash
cd assignment2
jupyter notebook
```
#### 6. Note: for Mac OS X run ./start_ipython_osx.sh instead of jupyter notebook
#### 7. open browser at localhost:8888/

## Assignement 3:
### How to run code
#### 1. Install and activate virtual environment:
```bash
cd assignment3
pip2 install virtualenv
virtualenv .env
source .env/bin/activate
```
#### 2. Install dependencies:
```
pip2 install -r requirements.txt
```
#### 3. Download data:
```bash
cd cs231n/datasets
./get_datasets.sh                # download dataset
```
#### 4. Build Cython modules:
```
cd assignment3/cs231n
python3 setup.py build_ext --inplace # build C code
```
#### 5. Run jupyter:
```bash
cd assignment3
jupyter notebook
```
#### 6. Note: for Mac OS X run ./start_ipython_osx.sh instead of jupyter notebook
#### 7. open browser at localhost:8888/


