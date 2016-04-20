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

*file_system.py*
1. Create new folder
2. Delete the folder with subfolder

*image_proc.py*
1. Open and convert the image to a special type
2. Show the image
3. Get the laplacien pyramid
4. Lecun local contrast normalization
