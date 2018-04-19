#include <stdio.h>
#include <string.h>
#include <math.h>

//typedef struct Network Network;
//typedef struct Layer Layer;
//typedef struct Node Node;
//typedef struct Vector Vector;

// Tipo de capas y tipo de funcion de activacion
typedef enum LayerType{INPUT, HIDDEN, OUTPUT} LayerType;
typedef enum ActFctType{SIGMOIDE} ActFctType;

//Estructura que contiene el numero definido de valores de la imagen de entrada
typedef struct Vector{
    int size;
    double vals[];
}Vector;

//Estructura para modelar la neurona 
typedef struct Node{
    //double bias;
    double output;
    int Nweights;
    double weights[];
}Node;

//Estructura que contiene el numero de conexiones por capa
typedef struct Layer{
    int Nnodes;
    Node nodes[];
}Layer;

//Estructura que contiene toda la red neuronal
typedef struct Network{
    int inpNodeSize; // Tamaño de un nodo de la capa de entrada
    int inpLayerSize;// Tamaño de la capa de entrada
    int hidNodeSize; // Tamaño de un nodo de la capa oculta
    int hidLayerSize;// Tamaño de la capa oculta
    int outNodeSize; // Tamaño de un nodo de la capa de salida
    int outLayerSize;// Tamaño de la capa de salida
    double learningRate; // Taza de aprendizaje
    ActFctType hidLayerActType; // Funcion de activacion para la capa oculta
    ActFctType outLayerActType; // Funcion de activacion para la capa de entrada
    Layer layers[]; // Vector de capas
}Network;

/*Se crea la red que contiene tres capas: entrada, capa oculta y salida
 inpCount = numero de nodos en la entrada
 hiCount = numero de nodos en la capa oculta
 outCount = numero de nodos ne la salida */
Network *createNetwork(int inpCount, int hidCount, int outCount);

/* Se utiliza un vector donde esta hay informacion de las imagenes 
   para la entrada de la red neuronal*/
void feedInput(Network *nn, Vector *v);

// Se hace el recorrido de la entrada a la capa oculta y de ahí a la salida
void feedForwardNetwork(Network *nn);

//Se propaga el error de la salida a la capa oculta
//Target
void backPropagateNetwork(Network *nn, int targetClassification);

//Obtenemos el resultado de la red neuronal
int getNetworkClassification(Network *nn);