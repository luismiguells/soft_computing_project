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