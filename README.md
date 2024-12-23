# Proyectos de computer vision con fútbol. 

Este es un repositorio para ver temas de computer vision. Algunas fuentes útiles:

Para este proyecto se han realizado varios diferentes ejemplos. Se utilizó un video de Pele y un casi gol contra Uruguay en 1970 que está en este repo. 

Con este video, se realizaron varias acciones y pruebas. 

1. Con OPENCV[[4]](#4) se separó en frames todo el video y se puede ver cada uno de ellos, que luego se guardan en un folder. 
2. Se realizó un "annotation" de caras con PIL y mediapipe. Este proyecto se podría extender con deepface[[2]](#2) pero está en proceso.  
3. Otro proyecto interesante que está acá es el uso de RT-DETR[[1]](#1) que gracias a Carlos Alarcón por sus excelente pudo hacerse posible, les recomiendo revisarlos en estos dos links[[3]](#3)[[5]](#5) 
4. Adicionalmente, corté un video largo de Maradona y el gol con los ingleses para luego hacer la detección de personas. 
5. Con el video de Maradona, saqué el audio original y lo empaté con el video que había usado computer vision para que este tenga audio, el gol es más emocionante. Para esto, usé ffmpeg[[6]](#6)
6. Armé un video sencillo de gente entrando a un centro comercial. 


<a id="1">[1]</a>
Zhao Y, et al. (2024).
[DETRsBeat YOLOsonReal-time Object Detection](https://arxiv.org/pdf/2304.08069)
Arxiv

<a id="2">[2]</a>
Serengil
[Deepface](https://github.com/serengil/deepface)

<a id="3">[3]</a>
Alarcon, C. 
[RT-DETR](https://github.com/alarcon7a/rt-detr)


<a id="4">[4]</a>
OPENCV
[OPENCV](https://github.com/opencv/opencv)

<a id="5">[5]</a>
Alarcon, C.
[RT-DETR: Revolucionando la Detección de Objetos en Tiempo Real ¡GRATIS!](https://www.youtube.com/watch?v=fqgHlUH3OXQ)
Youtube

<a id="6">[6]</a>
[FFMPEG](https://ffmpeg.org/)

<!--
https://www.youtube.com/watch?v=aBVGKoNZQUw

https://www.youtube.com/watch?v=aBVGKoNZQUw

https://stackoverflow.com/questions/78841248/userwarning-symboldatabase-getprototype-is-deprecated-please-use-message-fac

-->
