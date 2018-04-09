# Código de guía para el proyecto

Aquí están los archivos del proyecto del chino y están los links donde explica paso a paso, el primero es como lo realizo
sólo usando una neurona y en el segundo link como lo hizo con una red neuronal y backpropagation, hasta el final de la página antes delos comentarios viene el link para el repositorio de donde tiene su proyecto.

 https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
 
 https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/

He visto que en la carpeta 'util' son las utilidades hay 6 archivos
1) mnist-utils.c y mnist-utils.h sirven para abrir la base de datos
2) mnist-stats.c y mnist-stats.h sirven para desplegar el progreso ya sea el 20% o 60% al entrenar la red, así como
   también dibujar los números en pantalla
3) screen.c y screen.h que sirven para limpiar la pantalla y poner colores más o menos entendí pero creo que no se usa
4) 3lnn.c y 3lnn.h es la red neuronal con backpropagation

Para correlo, en la terminal sólo escriben 'make' y después para ejecutarlo './bin/mnist-3lnn'
