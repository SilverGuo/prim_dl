## This is a python project of deep learning for scene labeling.

----

### Here is the structure of the project,

prim_dl  
├── data  
│   ├── data_stanford  
│   │   ├── images  
│   │   └── labels  
│   └── test  
└── scripts  

- ***data***, data forlder, ignored by git. We train the model by the backgroud dataset from Stanford Lab. Also there are data below test for debug.
- ***scripts***, contain the relative fonctions

----

### Fonctions

***file_system.py***  
- Create new folder  
- Delete the folder with subfolder  

***image_proc.py***  
- Open and convert the image to a special type  
- Show the image  
- Get the laplacien pyramid  
- Lecun local contrast normalization  

***cnn***
- To be done ...

***segmentation***
- To be done ...
