#include <stdio.h>

typedef struct Network Network;
typedef struct Layer Layer;
typedef struct Node Node;
typedef struct Vector Vector;

typedef enum LayerType{INPUT, HIDDEN, OUTPUT} LayerType;
typedef enum ActFctType{SIGMOID} ActFctType;

//Estructura que contiene el numero definido de valores 
struct Vector{
    int size;
    double vals[];
};

//Estructura para modelar la neurona 
struct Node{
    double bias;
    double output;
    int wcount;
    double weights[];
};

//Estructura que contiene el numero de conexiones por capa
struct Layer{
    int ncount;
    Node nodes[];
};

//Estructura que contiene toda la red neuronal
struct Network{
    int inpNodeSize;
    int inpuLayerSize;
    int hiNodeSize;
    int hiLayerSize;
    int outNodeSize;
    int outLayerSize;
    double learningRate;
    ActFctType hidLayerActType;
    ActFctType outLayerActType;
    Layer layers[];
};

/*Se crea la red que contiene tres capas: entrada, capa oculta y salida
 inpCount = numero de conexiones en la entrada
 hiCount = numero de conexiones en la capa oculta
 outCount = numero de conexiones ne la salida */
Network *createNetwork(int inpCount, int hidCount, int outCount);

/* Se utiliza un vector donde esta hay informacion de las imagenes 
   para la entrada de la red neuronal*/
void feedInput(Network *nn, Vector *v);

// Se hace el recorrido de la entrada a la capa oculta y de ahí a la saliad
void feedForwardNetwork(Network *nn);

//Se propaga el error de la salida a la capa oculta
void backPropagateNetwork(Network *nn, int targetClassification);

//Obtenemos el resultado de la red neuronal
int getNetworkClassification(Network *nn);
