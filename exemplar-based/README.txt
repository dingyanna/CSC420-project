Scripts to generate results:
python3 main.py ./images/triangle/triangle.png ./images/triangle/trianglemask.png ./results/triangleresult.png 5
python3 main.py ./images/triangle/triangle.png ./images/triangle/trianglemask.png ./results/badtriangleresult.png 9

python3 main.py ./images/flower1/flower1.jpg ./images/flower1/flower1mask.png ./results/flower1.png 3 
python3 main.py ./images/flower2/flower2.jpg ./images/flower2/flower2mask.png ./results/flower2.png 3  

python3 main.py  ./images/brick/brick.png ./images/brick/brickmask.png ./results/brick.png 9 



Input Requirements:
1. The input "mask" should use black pixel values to specify source region and other pixel values to specify target region.