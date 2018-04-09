#include <stdint.h>
#include <stdio.h>

//Arhivo para entrenar la red con sus etiquetas
#define MNIST_TRAINING_SET_IMAGE_FILE_NAME "data/train-images-idx3-ubyte" 
#define MNIST_TRAINING_SET_LABEL_FILE_NAME "data/train-labels-idx1-ubyte" 

//Archivo para probar la red con sus etiquetas
#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte"  
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte"  



#define MNIST_MAX_TRAINING_IMAGES 60000 //Numero de imagenes de entranamiento                    
#define MNIST_MAX_TESTING_IMAGES 10000  //Numero de imagenes de prueba                   
#define MNIST_IMG_WIDTH 28 //Ancho de la imagen en pixeles                                  
#define MNIST_IMG_HEIGHT 28 //Alto de la imagen en pixeles

typedef struct MNIST_ImageFileHeader MNIST_ImageFileHeader;
typedef struct MNIST_LabelFileHeader MNIST_LabelFileHeader;                                

typedef struct MNIST_Image MNIST_Image;
typedef uint8_t MNIST_Label;

//Estructura para almacenar la imagen
struct MNIST_Image{
    uint8_t pixel[28*28];
};

//Estructura para leer el encabezado del archivo de imagenes de prueba
struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};


//Estructura para leeer el encabezado del archivo de etiqutas de imagenes de prueba
struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
};

FILE *openMNISTImageFile(char *fileName); //Se lee el encabezado imagenes


FILE *openMNISTLabelFile(char *fileName); //Se lee el encabezado etiqutas


MNIST_Image getImage(FILE *imageFile); //Se obtiene la imagen


MNIST_Label getLabel(FILE *labelFile); //Se obtiene la etiqueta
