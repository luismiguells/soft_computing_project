#include <stdlib.h>
#include "mnist.h"
#include "RNA.h"



//Se obtiene un vector el cual contiene los pixeles de la imagen
Vector *getVectorFromImage(MNIST_Image *img){
    
    //Se reserva memoria 
    Vector *v = (Vector*)malloc(sizeof(Vector) + (MNIST_IMG_WIDTH*MNIST_IMG_HEIGHT * sizeof(double)));
    
    v->size = MNIST_IMG_WIDTH*MNIST_IMG_HEIGHT;
    
    for (int i = 0; i < v->size; i++)
        v->vals[i] = img->pixel[i] ? 1 : 0;
    
    return v;
}



//Se entrena la red
void trainNetwork(Network *nn){
    
    //Se abren los archivos de prueba
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
    
    int errCount = 0;

    //Se recorre cada una de las imagenes de la red
    printf("La red se esta entrenando...\n");
    for (int imgCount = 0; imgCount < MNIST_MAX_TRAINING_IMAGES; imgCount++){
        
        //Se lee la imagen con su correspondiente etiqueta
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        
        //Se convierte la imagen a un vector para ingresar a la red
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(nn, inpVector);
        
        //Se hacen los calculos hacia delante
        feedForwardNetwork(nn);
        
        //Se realiza backpropagation
        backPropagateNetwork(nn, lbl);
        
        //Se clasifica la imagen escogiendo la salida mas alta
        int classification = getNetworkClassification(nn);

        //Se va contando cuantas veces se equivoca
        if (classification != lbl){
            errCount++;
        }
    }
    
    //Cierran archivos
    fclose(imageFile);
    fclose(labelFile);
    
}


//Se prueba la red
void testNetwork(Network *nn){
    
    //Se abren los archivos
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
    
    int errCount = 0;
    
    //Se recorren las imagenes
    for (int imgCount = 0; imgCount < MNIST_MAX_TESTING_IMAGES; imgCount++){
        
        //Se lee cada imagen con su etiqueta
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);

        //Se convierte la imagen a un vector para ingresar a la red        
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(nn, inpVector);
        
        //Se hacen los calculos hacia delante
        feedForwardNetwork(nn);
        
        //Se clasifica la imagen escogiendo la salida mas alta
        int classification = getNetworkClassification(nn);
        if (classification!=lbl){
             errCount++;
        }
        
        //Se muestra la imagen 
        displayImage(&img, lbl, classification, 7, 6);
        getchar();
        
    }
    
    //Se cierran los archivos
    fclose(imageFile);
    fclose(labelFile);
    
}


int main(int argc, const char * argv[]) {
    
    
    printf("\e[1;1H\e[2J"); //Limpiar pantalla
    printf("Red neuronal de 3 capas para numeros escritos a mano\n");
    
    //Se crea la red neuronal
    Network *nn = createNetwork(MNIST_IMG_HEIGHT*MNIST_IMG_WIDTH, 20, 10);

    //Se entrena la red    
    trainNetwork(nn);
    
    //Se prueba la red
    testNetwork(nn);
    
    //Se libera memoria
    free(nn);

    return 0;
}
