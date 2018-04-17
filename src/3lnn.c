#include <stdio.h>
#include <stdlib.h>
#include "3lnn.h"
#include "mnist.h"


//Se busca la conexión en cada uno de los nodos
Node *getNode(Layer *l, int nodeId){
    int nodeSize = sizeof(Node) + (l->nodes[0].wcount*sizeof(double));
    uint8_t *sbptr = (uint8_t*) l->nodes;

    sbptr += nodeId * nodeSize;

    return (Node*) sbptr;
}

//Se accedera a cada una de las capas
Layer *getLayer(Network *nn, LayerType ltype){
    Layer *l;

    switch(ltype){
        //Datos de entrada
        case INPUT:{
            l = nn->layers;
            break;
        }
        //Se obtiene la entrada para la capa ocula
        case HIDDEN:{
            uint8_t *sbptr = (uint8_t*) nn->layers;
            sbptr += nn->inpLayerSize;
            l = (Layer*)sbptr;
            break;
        }
        //Se obtiene la salida 
        default:{
            uint8_t *sbptr = (uint8_t*) nn->layers;
            sbptr += nn->inpLayerSize + nn->hidLayerSize;
            l = (Layer*)sbptr;
            break;
        }
    }

    return l;
}

//Funcion de la derivada de la funcion de activacin 
double getActFctDerivative(Network *nn, LayerType ltype, double outVal){
    double dVal = 0;
    ActFctType actFct;

    //Se aplica la funcion a la salida de la capa oculta y a la salida
    if(ltype == HIDDEN){
        actFct = nn->hidLayerActType;
    }else{
        actFct = nn->outLayerActType;
    }

    //Derivada
    if(actFct == SIGMOID){
        dVal = outVal * (1 - outVal);
    }

    return dVal;
}

//Se actualizan los pesos segun el error 
void updateNodeWeights(Network *nn, LayerType ltype, int id, double error){
    
    Layer *updateLayer = getLayer(nn, ltype);
    Node *updateNode = getNode(updateLayer, id);
    
    Layer *prevLayer;
    int prevLayerNodeSize = 0;
    //Se van actualizando de la salida a la entrada
    if (ltype == HIDDEN) {
        prevLayer = getLayer(nn, INPUT);
        prevLayerNodeSize = nn->inpNodeSize;
    }else{
        prevLayer = getLayer(nn, HIDDEN);
        prevLayerNodeSize = nn->hidNodeSize;
    }
    
    uint8_t *sbptr = (uint8_t*) prevLayer->nodes;
    
    for (int i = 0; i < updateNode->wcount; i++){
        Node *prevLayerNode = (Node*)sbptr;
        updateNode->weights[i] += (nn->learningRate * prevLayerNode->output * error);
        sbptr += prevLayerNodeSize;
    }
    
    //Se actualiza el bias del peso
    updateNode->bias += (nn->learningRate * 1 * error);
    
}

//Backpropagation en la capa oculta
void backPropagateHiddenLayer(Network *nn, int targetClassification){
    
    //Seleccionamos el tipo de capa
    Layer *ol = getLayer(nn, OUTPUT);
    Layer *hl = getLayer(nn, HIDDEN);
    
    for (int h = 0;h < hl->ncount;h++){
        //Creamos la variable donde se almacenara las conexiones
        Node *hn = getNode(hl,h);
        
        double outputcellerrorsum = 0;
        
        for (int o = 0; o < ol->ncount; o++){
            
            Node *on = getNode(ol,o);
            
            int targetOutput = (o == targetClassification) ? 1 : 0;
            
            //Calculo de las delta
            double errorDelta = targetOutput - on->output;
            //Delta por la derivada
            double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, on->output);
            //calculamos la suma de los errores
            outputcellerrorsum += errorSignal * on->weights[h];
        }
        
        double hiddenErrorSignal = outputcellerrorsum * getActFctDerivative(nn, HIDDEN, hn->output);
        
        //Actualizamos los pesos
        updateNodeWeights(nn, HIDDEN, h, hiddenErrorSignal);
    }
    
}




//Backpropagation de la salida 
void backPropagateOutputLayer(Network *nn, int targetClassification){
    
    //Seleccionamos la capa 
    Layer *ol = getLayer(nn, OUTPUT);
    
    for (int o = 0; o < ol->ncount; o++){
        
        Node *on = getNode(ol,o);
        
        //Se obtiene el error
        int targetOutput = (o == targetClassification) ? 1 : 0;
        //Se obtiene la delta
        double errorDelta = targetOutput - on->output;
        //Se multiplica la delta por la derivada
        double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, on->output);
        
        //Se actualizan los pesos
        updateNodeWeights(nn, OUTPUT, o, errorSignal);
        
    }
    
}

//Se propaga de la salida a la capa oculta de la capa oculta a la entrada
void backPropagateNetwork(Network *nn, int targetClassification){
    
    backPropagateOutputLayer(nn, targetClassification);
    
    backPropagateHiddenLayer(nn, targetClassification);
    
}

//Funcion de activiacion de cada una de las conexiones 
void activateNode(Network *nn, LayerType ltype, int id){
    
    //Obtenemos la capa
    Layer *l = getLayer(nn, ltype);
    //Obtenemos la conexcion
    Node *n = getNode(l, id);
    
    ActFctType actFct;
    
    //Se realiza la funcion de activacion
    if (ltype == HIDDEN){
        actFct = nn->hidLayerActType;
    }else{
        actFct = nn->outLayerActType;
    }
    
    //Formula de la funcion de activacion
    if (actFct == SIGMOID){
        n->output = 1 / (1 + (exp((double)-n->output)));
    }
    
}


//Se calcula la salida en de cada una de las capas multiplicando por el peso de cada una de las conexiones
void calcNodeOutput(Network *nn, LayerType ltype, int id){
    
    //Se obtienen la capa y las conexiones
    Layer *calcLayer = getLayer(nn, ltype);
    Node *calcNode = getNode(calcLayer, id);
    
    Layer *prevLayer;
    int prevLayerNodeSize = 0;
    
    //Revisamos si es una de las capas 
    if (ltype == HIDDEN) {
        prevLayer = getLayer(nn, INPUT);
        prevLayerNodeSize = nn->inpNodeSize;
    }else{
        prevLayer = getLayer(nn, HIDDEN);
        prevLayerNodeSize = nn->hidNodeSize;
    }
    
    uint8_t *sbptr = (uint8_t*) prevLayer->nodes;
    
    //Se agrega el bias
    calcNode->output = calcNode->bias;

    //Se realiza la multiplicacion por cada uno de los pesos
    for (int i = 0; i < prevLayer->ncount; i++){
        Node *prevLayerNode = (Node*)sbptr;
        calcNode->output += prevLayerNode->output * calcNode->weights[i];
        sbptr += prevLayerNodeSize;
    }

}

//Se calcula la salida
void calcLayer(Network *nn, LayerType ltype){
    Layer *l;
    l = getLayer(nn, ltype);
    
    for (int i = 0; i < l->ncount; i++){
        calcNodeOutput(nn, ltype, i);
        activateNode(nn,ltype,i);
    }
}

//Se realiza el calculo hacia delante 
void feedForwardNetwork(Network *nn){
    calcLayer(nn, HIDDEN);
    calcLayer(nn, OUTPUT);
}


//Se crea un vector el cual servira de entrada para la red
void feedInput(Network *nn, Vector *v) {
    
    Layer *il;
    il = nn->layers;
    
    Node *iln;
    iln = il->nodes;
    
    //Se copia la salida de la entrada
    for (int i = 0; i < v->size; i++){
        iln->output = v->vals[i];
        iln++;               
    }
    
}

//Se crea una capa de entrada y los pesos de forma aleatoria [0-1]
Layer *createInputLayer(int inpCount){
    
    int inpNodeSize     = sizeof(Node);         
    int inpLayerSize    = sizeof(Layer) + (inpCount * inpNodeSize);
    
    Layer *il = malloc(inpLayerSize);
    il->ncount = inpCount;
    
    //Se inicializan las variables
    Node iln;
    iln.bias = 0;
    iln.output = 0;
    iln.wcount = 0;
    
    //Se utiliza un puntero para ir obtinedo la informacion de la red
    uint8_t *sbptr = (uint8_t*) il->nodes;
    
    //Se copia la inicializacion n veces, donde n es el numero de entradas
    for (int i = 0; i < il->ncount; i++){
        memcpy(sbptr, &iln, inpNodeSize);
        sbptr += inpNodeSize;
    }
    
    return il;
}


//Se crea una capa con los pesos de manera aleatoria [0-1]
Layer *createLayer(int nodeCount, int weightCount){
    
    int nodeSize = sizeof(Node) + (weightCount * sizeof(double));
    Layer *l = (Layer*)malloc(sizeof(Layer) + (nodeCount*nodeSize));
    
    l->ncount = nodeCount;
    
    //Se inicializan las variables
    Node *dn = (Node*)malloc(sizeof(Node) + ((weightCount)*sizeof(double)));
    dn->bias = 0;
    dn->output = 0;
    dn->wcount = weightCount;
    for (int o = 0; o < weightCount; o++) dn->weights[o] = 0; 
    
    uint8_t *sbptr = (uint8_t*) l->nodes;     // single byte pointer
    
    //Se copian los valores
    for (int i = 0; i < nodeCount; i++){
        memcpy(sbptr+(i*nodeSize),dn,nodeSize);
    }
    
    free(dn);
    
    return l;
}



//Se inicia la red neuronal copiando las estructuras de datos de entrada, capa oculata y salida
void initNetwork(Network *nn, int inpCount, int hidCount, int outCount){
    
    //Se copia la informacion de la entrada a la estructura de la red neuronal
    Layer *il = createInputLayer(inpCount);
    memcpy(nn->layers, il, nn->inpLayerSize);
    free(il);
    
    //Se mueve el puntero a la salida de la entrada que sera la entrada de la capa ocualta
    uint8_t *sbptr = (uint8_t*) nn->layers;     
    sbptr += nn->inpLayerSize;
    
    //Se copia la informacion de la capa oculta
    Layer *hl = createLayer(hidCount, inpCount);
    memcpy(sbptr,hl,nn->hidLayerSize);
    free(hl);
    
    //Se mueve el puntero a la salida de la capa oculta que sera la entrada de la salida de la red
    sbptr += nn->hidLayerSize;
    
    //Se copia la informacion de la salida 
    Layer *ol = createLayer(outCount, hidCount);
    memcpy(sbptr,ol,nn->outLayerSize);
    free(ol);
    
}


//Inicializacion de los parametros de la red neuronal
void setNetworkDefaults(Network *nn){
    
    //Funciones de activacion de cada una de las capas
    nn->hidLayerActType = SIGMOID;
    nn->outLayerActType = SIGMOID;
    
    //Factor de aprendizaje
    nn->learningRate    = 0.2;      
    
}


//Se inicializan los pesos de manera random
void initWeights(Network *nn, LayerType ltype){
    
    int nodeSize = 0;
    if (ltype == HIDDEN) nodeSize=nn->hidNodeSize;
                  else nodeSize=nn->outNodeSize;
    
    //Selecciona la capa
    Layer *l = getLayer(nn, ltype);
    
    uint8_t *sbptr = (uint8_t*) l->nodes;
    
    //Se comienzan a asignar valores
    for (int o = 0; o < l->ncount; o++){
    
        Node *n = (Node *)sbptr;
        
        for (int i = 0; i < n->wcount; i++){
            n->weights[i] = 0.7*(rand()/(double)(RAND_MAX));//Funcion random
            if (i%2) n->weights[i] = -n->weights[i];  //La mitad de los pesos son negativos
        }
        
        //Se inicializa el bias
        n->bias =  rand()/(double)(RAND_MAX);
        if (o%2) n->bias = -n->bias;  //La mitad de los bias es negativo
        
        sbptr += nodeSize;
    }
    
}


//Se crea la red neuronal
Network *createNetwork(int inpCount, int hidCount, int outCount){
    
    //Se calcula el tamaño de la entrada
    int inpNodeSize     = sizeof(Node);        
    int inpLayerSize    = sizeof(Layer) + (inpCount * inpNodeSize);
    
    //Se calcula el tamaño de la capa oculta
    int hidWeightsCount = inpCount;
    int hidNodeSize     = sizeof(Node) + (hidWeightsCount * sizeof(double));
    int hidLayerSize    = sizeof(Layer) + (hidCount * hidNodeSize);
    
    //Se calcula el tamaño de la salida
    int outWeightsCount = hidCount;
    int outNodeSize     = sizeof(Node) + (outWeightsCount * sizeof(double));
    int outLayerSize    = sizeof(Layer) + (outCount * outNodeSize);
    
    // Allocate memory block for the network
    Network *nn = (Network*)malloc(sizeof(Network) + inpLayerSize + hidLayerSize + outLayerSize);
    
    //Se pone el tamaño de cada uno de los componentes de la red
    nn->inpNodeSize     = inpNodeSize;
    nn->inpLayerSize    = inpLayerSize;
    nn->hidNodeSize     = hidNodeSize;
    nn->hidLayerSize    = hidLayerSize;
    nn->outNodeSize     = outNodeSize;
    nn->outLayerSize    = outLayerSize;
    
    //Se crean cada una de las partes de la red
    initNetwork(nn, inpCount, hidCount, outCount);
    
    //Se inicializan las variables
    setNetworkDefaults(nn);
    
    //Se inicializan los pesos de cada una de las capas
    initWeights(nn, HIDDEN);
    initWeights(nn, OUTPUT);
    
    return nn;
}

//Se obtiene la clasificacion tomando el valor maximo de la salida
int getNetworkClassification(Network *nn){
    
    //Se selecciona la salida 
    Layer *l = getLayer(nn, OUTPUT);
    
    double maxOut = 0;
    int maxInd = 0;
    
    //Se recorre para encontrar el valor
    for (int i = 0; i < l->ncount; i++){
        
        Node *on = getNode(l,i);
        //Se verifiva el valor maximo
        if (on->output > maxOut){
            maxOut = on->output;
            maxInd = i;
        }
    }
    
    return maxInd;
}


