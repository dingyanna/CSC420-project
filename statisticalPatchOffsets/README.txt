The code has to use a library called gco_wrapper, which can be installed by:
    pip3 install gco_wrapper
And since I used sift code, opencv-contrib-python also has to be installed. 
    pip3 install opencv-contrib-python
Apart from these, I used the common libraries like numpy, matplotlib, sklearn, opencv

In order to run my code, 
     python3 project.py ./images2/Brick.jpg ./images2/Brick.png [-o | -s]
-o means that the original algorithm is used, 
-s means that my version, the one using sift code will be used. 

after running, the produced image will be named Brick.jpg-filled.png
and saved into the same directory as the source image