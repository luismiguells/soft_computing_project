#include <stdio.h>
#include <stdlib.h>
#include "RNA.h"
#include "mnist.h"

// Funcion que regresa un puntero a la direccion del nodo deseado
Node *getNode(Layer *l, int nodeId){
    // Tamaño predeterminado de un nodo
    int nodeSize = sizeof(Node) + (l->nodes[0].Nweights*sizeof(double));
    // Se apunta al primer nodo de la capa (puntero de un byte)
    uint8_t *Nodeptr = (uint8_t*) l->nodes;

    // Se recorre el puntero hasta el nodo deseado
    Nodeptr += nodeId * nodeSize;

    return (Node*) Nodeptr;
}

// Funcion que regresa un puntero a la direccion de la capa deseada
Layer *getLayer(Network *nn, LayerType ltype){
    Layer *l;

    switch(ltype){
        // Si solicitamos la direccion de la capa de entrada no se avanza, pues es la primera
        case INPUT:{
            l = nn->layers;
            break;
        }
        // En caso de la capa oculta, se avanza el tamaño de la capa de entrada
        // para asi apuntar al inicio de la capa 
        case HIDDEN:{
            uint8_t *layerptr = (uint8_t*) nn->layers;
            layerptr += nn->inpLayerSize;
            l = (Layer*)layerptr;
            break;
        }
        // Aqui es el caso de la capa de salida, donde se avanza el tamaño de la capa de entrada y la oculta
        // para asi apuntar al inicio de la capa de salida
        default:{
            uint8_t *layerptr = (uint8_t*) nn->layers;
            layerptr += nn->inpLayerSize + nn->hidLayerSize;
            l = (Layer*)layerptr;
            break;
        }
    }

    return l;
}

// Regresa la derivada de la funcion de activacion de cada nodo
double getActFctDerivative(Network *nn, LayerType ltype, double outVal){
    double dVal = 0;
    ActFctType actFct;

    // Si la capa actual es la oculta, se usa la funcion de activacion correspondiente
    if(ltype == HIDDEN){
        actFct = nn->hidLayerActType;
    }
    // Si no, entonces la capa actual es la de salida y se usa la funcion de activacion correspondiente
    else{
        actFct = nn->outLayerActType;
    }

    // Si la funcion de activacion a usar es la sigmoide, se calcula su derivada correspodiente
    if(actFct == SIGMOIDE){
        // Derivada de la funcion de activacion sigmoide
        dVal = outVal * (1 - outVal);
    }

    return dVal;
}


/*
Esta funcion actualiza los pesos en cada nodo
nn: puntero a la red neuronal
ltype: tipo de capa
id: indice que identifica el nodo al que se actualizaran los pesos
error: error correspondiente al nodo especifico
*/
void updateNodeWeights(Network *nn, LayerType ltype, int id, double error){
    
    // Se obtiene la capa y nodo deseado
    Layer *updateLayer = getLayer(nn, ltype);
    Node *updateNode = getNode(updateLayer, id);
    
    // Se busca el tipo de capa anterior a la actual
    Layer *prevLayer;
    int prevLayerNodeSize = 0;
    // Si la capa actual es la oculta entonces la anterior es la de entrada
    if (ltype == HIDDEN) {
        prevLayer = getLayer(nn, INPUT);
        prevLayerNodeSize = nn->inpNodeSize;
    }
    // Si no, la capa actual es la de salida y entonces la anterior es la oculta
    else{
        prevLayer = getLayer(nn, HIDDEN);
        prevLayerNodeSize = nn->hidNodeSize;
    }

     
    // Se usa un puntero para apuntar a los nodos de la capa anterior
    uint8_t *NodePrevptr = (uint8_t*) prevLayer->nodes;
    
    // Se calcula en nuevo peso del nodo actual con la sumatoria de las multiplicaciones de la taza de aprendizaje, la salidas
    // de la capa anterior, el error y se suma al peso actual. Se avanza nodo a nodo de la capa anterior para calcular el nuevo peso
    for (int i = 0; i < updateNode->Nweights; i++){
        Node *prevLayerNode = (Node*)NodePrevptr;
        updateNode->weights[i] += (nn->learningRate * prevLayerNode->output * error);
        NodePrevptr += prevLayerNodeSize;
    }
    
}



/*
Propaga el error en la capa oculta
nn: puntero a la red neuronal
targetClassification Es la etiqueta de la clasificacion correcta de la imagen de entrada
 */

void backPropagateHiddenLayer(Network *nn, int targetClassification){
    
    // Se obtiene las capas correspondientes
    Layer *ol = getLayer(nn, OUTPUT);
    Layer *hl = getLayer(nn, HIDDEN);
    
    // Se recorren los nodos de la capa oculta
    for (int h=0;h<hl->Nnodes;h++){
        Node *hn = getNode(hl,h);
        
        double outputNodeErrorSum = 0;
        
        // Se recorren los nodos de la capa de salida
        for (int o=0;o<ol->Nnodes;o++){
            
            Node *on = getNode(ol,o);
            
             // El targetOutput valdra 1 en el nodo donde el indice coincida con el valor real de la imagen 0-9
             // en los demas nodos sera igual a 0
            int targetOutput = (o==targetClassification)?1:0;
            
            // Se calcula el error que es la diferencia de la salida deseada con nuestra salida obtenida
            double errorDelta = targetOutput - on->output;
            // Se calcula cada error con la formula correspondiente respecto a la funcion de activacion usada
            double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, on->output);
            
            // Suma total del error en la capa de salida
            outputNodeErrorSum += errorSignal * on->weights[h];
        }
        
        // Error para la capa oculta respecto a los errores de la capa de salida y la funcion de activacion usada
        double hiddenErrorSignal = outputNodeErrorSum * getActFctDerivative(nn, HIDDEN, hn->output);
        
        // Se actualizan los pesos
        updateNodeWeights(nn, HIDDEN, h, hiddenErrorSignal);
    }
    
}




/*
Propaga el error en la capa de salida
nn: puntero a la red neuronal
targetClassification Es la etiqueta de la clasificacion correcta de la imagen de entrada
 */

void backPropagateOutputLayer(Network *nn, int targetClassification){
    
    Layer *ol = getLayer(nn, OUTPUT);
    
    // Se recorre cada nodo en la capa de salida
    for (int o=0;o<ol->Nnodes;o++){
        
        Node *on = getNode(ol,o);
        
        // El targetOutput valdra 1 en el nodo donde el indice coincida con el valor real de la imagen 0-9
        // en los demas nodos sera igual a 0
        int targetOutput = (o==targetClassification)?1:0;
        
        // Se calcula el error que es la diferencia de la salida deseada con nuestra salida obtenida
        double errorDelta = targetOutput - on->output;
        // Se calcula cada error con la formula correspondiente respecto a la funcion de activacion usada
        double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, on->output);
        
        // Se actualizan los pesos en cada nodo
        updateNodeWeights(nn, OUTPUT, o, errorSignal);
        
    }
    
}




/*
Esta funcion hace el retroceso para calcular la propagacion del error en capa capa
nn: puntero a la red neuronal
targetClassification Es la etiqueta de la clasificacion correcta de la imagen de entrada
*/

void backPropagateNetwork(Network *nn, int targetClassification){
    
    backPropagateOutputLayer(nn, targetClassification);
    
    backPropagateHiddenLayer(nn, targetClassification);
    
}




/*
Se le aplica la funcion de activacion correspondiente a un nodo especifico
nn: puntero a la red neuronal
ltype: tipo de capa
id: indice que identifica el nodo a calcular con la funcion de activacion
 */

void activateNode(Network *nn, LayerType ltype, int id){
    
    // Se obtiene la capa y nodo correspondiente
    Layer *l = getLayer(nn, ltype);
    Node *n = getNode(l, id);
    
    // Se crea una variable del tipo funcion de activacion
    ActFctType actFct;
    
    // Si la capa actual es la oculta se obtiene el tipo de funcion de activacion que se usara
    if (ltype == HIDDEN){
        actFct = nn->hidLayerActType;
    }

    // Si no, entonces la capa actual es la de salida y se obtiene el tipo de funcion de activacion que se usara
    else{
        actFct = nn->outLayerActType;
    }
    
    // Si el tipo de funcion a usar es la sigmoide, se calcula con su ecuacion correspondiente y se guarda el valor en la salida del nodo
    if (actFct == SIGMOIDE){
        n->output = 1 / (1 + (exp((double)-n->output)));
    }
    
}




/*
Calcula el valor de salida de un nodo especifico multiplicando sus pesos con las entradas correspondientes a las 
salidas de la capa anterior: yi += Wi*Xi
 nn: puntero a nuestra red neuronal
 ltype: Tipo de capa a calcular
 id: indice que identifica el nodo a calcular
 */

void calcNodeOutput(Network *nn, LayerType ltype, int id){
    
    // Se obtiene la capa y el nodo correspondiente a calcular
    Layer *calcLayer = getLayer(nn, ltype);
    Node *calcNode = getNode(calcLayer, id);
    
    // Se crea un puntero que apuntara a la capa anterior
    Layer *prevLayer;
    int prevLayerNodeSize = 0;
    
    // Si la capa actual es la oculta entonces la anterior es la de entrada
    if (ltype == HIDDEN) {
        prevLayer = getLayer(nn, INPUT);
        prevLayerNodeSize = nn->inpNodeSize;
    }

    // Si no, la capa actual es la de salida y entonces la anterior es la oculta
    else{
        prevLayer = getLayer(nn, HIDDEN);
        prevLayerNodeSize = nn->hidNodeSize;
    }
    
    // Se usa un puntero para apuntar a los nodos de la capa anterior
    uint8_t *NodePrevptr = (uint8_t*) prevLayer->nodes;


    
    // Se calcula la salida del nodo actual con la sumatoria de las multiplicaciones de los pesos por
    // las salidas de la capa anterior y se avanza nodo a nodo de la capa anterior para calcular la salida
    for (int i=0; i<prevLayer->Nnodes;i++){
        Node *prevLayerNode = (Node*)NodePrevptr;
        calcNode->output += prevLayerNode->output * calcNode->weights[i];
        NodePrevptr += prevLayerNodeSize;
    }

}




/*
 Calcula el valor de salida de cada nodo en la capa correspondiente
 nn: puntero a nuestra red neuronal
 ltype: Tipo de capa a calcular
 */

void calcLayer(Network *nn, LayerType ltype){
    // Se obtiene la capa correspondiente
    Layer *l;
    l = getLayer(nn, ltype);
    
    // Recorre cada nodo en la capa para obtener su salida y mandarla a la funcion de activacion
    for (int i=0;i<l->Nnodes;i++){
        calcNodeOutput(nn, ltype, i);
        activateNode(nn,ltype,i);
    }
}




/*
Se hace el recorrido de la capa de entrada a la oculta y de ahi a la capa de salida, calculando en cada nodo
los valores de salida multiplicando sus pesos con los valores de salida de los nodos de la capa anterior
yi += Wi*Xi
nn: puntero a nuestra red neuronal
 */
void feedForwardNetwork(Network *nn){
    calcLayer(nn, HIDDEN);
    calcLayer(nn, OUTPUT);
}




/*
Se alimenta la capa de entrada con los valores correspondientes de la imagen a predecir
nn: puntero a la red neuronal
v: puntero al vector que contiene los valores de la imagen
 */

void feedInput(Network *nn, Vector *v) {
    
    // Se crea un puntero tipo capa y este apunta la capa de entrada de la red
    Layer *il;
    il = nn->layers;
    
    // Se crea un puntero tipo nodo y este apunta a los nodos de la capa de entrada de la red
    Node *iln;
    iln = il->nodes;
    
    // Se copian los valores de la imagen de entrada en el nodo correspondiente
    for (int i=0; i<v->size;i++){
        iln->output = v->vals[i];
        iln++;               
    }
    
}



// Crea la capa de entrada con el numero de nodos deseados y regresa un puntero a esta capa
// inpCount: numero de nodos deseados en la capa
Layer *createInputLayer(int inpCount){
    
    // Tamaño de un nodo de la capa de entrada
    int inpNodeSize     = sizeof(Node);
    // Tamaño de la capa de entrada
    int inpLayerSize    = sizeof(Layer) + (inpCount * inpNodeSize);
    
    // Se reserva memoria suficiente para el tamaño de la capa
    Layer *il = malloc(inpLayerSize);
    // Se guarda el numero de nodos de la capa
    il->Nnodes = inpCount;
    
    // Se crea el nodo por default de la capa, con valores neutros
    Node iln;
    iln.output = 0;
    iln.Nweights = 0;
    
    
    // Se usa un puntero de un byte para moverse en la direcciones de memoria de la capa y llenar la capa con cada nodo
    uint8_t *Nodeptr = (uint8_t*) il->nodes;
    
    // Se copia el nodo por default N veces para tener en la capa el numero de nodos deseados
    for (int i=0;i<il->Nnodes;i++){
        memcpy(Nodeptr,&iln,inpNodeSize);
        // Se avanza nodo a nodo
        Nodeptr += inpNodeSize;
    }
    
    return il;
}



/*
Se crea la capa correspondiente y se fijan los pesos con valores aleatorios entre 0 y 1
nodeCount: Numero de nodos
weightCount Numero de pesos por nodo
 */

Layer *createLayer(int nodeCount, int weightCount){
    
    // Se calcula el tamaño de cada nodo
    int nodeSize = sizeof(Node) + (weightCount * sizeof(double));
    // Se reserva memoria para la capa, consideranco el numero de nodos
    Layer *l = (Layer*)malloc(sizeof(Layer) + (nodeCount*nodeSize));
    
    // Se guarda el numero de nodos en la capa
    l->Nnodes = nodeCount;
    
    // Se crea un nodo por default
    Node *dn = (Node*)malloc(sizeof(Node) + ((weightCount)*sizeof(double)));
    dn->output = 0;
    dn->Nweights = weightCount;
    // Inicializamos cada peso en 0
    for (int o=0;o<weightCount;o++) dn->weights[o] = 0;
    
    uint8_t *Nodeptr = (uint8_t*) l->nodes; // Creamos un puntero que apunte al primer nodo
    
    // Copiamos el nodo por default N veces para tener todos los nodos deseados en la capa
    for (int i=0;i<nodeCount;i++) memcpy(Nodeptr+(i*nodeSize),dn,nodeSize);
    
    free(dn);
    
    return l;
}


/* Inicializamos la red neuronal creando cada capa y copiandola en el bloque de memoria
correspondiente de nuestra red
nn es un puntero a la red neuronal
inpCount = numero de nodos en la entrada
hiCount = numero de nodos en la capa oculta
outCount = numero de nodos ne la salida
*/
void initNetwork(Network *nn, int inpCount, int hidCount, int outCount){
    
    // Se crea la capa de entrada con el numero de nodos deseados
    Layer *il = createInputLayer(inpCount);
    // Se copia la capa creada a la direccion de memoria correspondiente de nuestra red
    // memcpy(Bloque de memoria de destino, datos que se desean copiar, tamaño o cantidad de datos)
    memcpy(nn->layers,il,nn->inpLayerSize);
    // Se libera la memoria creada temporalmente
    free(il);
    
    // Se utiliza un puntero de un byte para movernos en el espacio de direccion de memoria reservado
    // Se fija el puntero al inicio de la capa de entrada y se le suma el tamaño de esta capa para 
    // apuntar al inicio de la capa oculta
    uint8_t *Layerptr = (uint8_t*) nn->layers;     // puntero de un byte
    Layerptr += nn->inpLayerSize;
    
    // Se crea la capa oculta con el numero de nodos deseados
    Layer *hl = createLayer(hidCount, inpCount);
    // Se copia la capa creada a la direccion de memoria correspondiente de nuestra red
    // memcpy(Bloque de memoria de destino, datos que se desean copiar, tamaño o cantidad de datos)
    memcpy(Layerptr,hl,nn->hidLayerSize);
    // Se libera la memoria creada temporalmente
    free(hl);
    
    // Se avanza en el puntero el tamaño de la capa oculta para apuntar al inicio de la capa de salida
    Layerptr += nn->hidLayerSize;
    
    // Se crea la capa de salida con el numero de nodos deseados
    Layer *ol = createLayer(outCount, hidCount);
    // Se copia la capa creada a la direccion de memoria correspondiente de nuestra red
    // memcpy(Bloque de memoria de destino, datos que se desean copiar, tamaño o cantidad de datos)
    memcpy(Layerptr,ol,nn->outLayerSize);
    // Se libera la memoria creada temporalmente
    free(ol);
    
}




/*
 En esta funcion se fijan los parametros que usara nuestra RNA como lo son las funciones de activacion
 y la taza de aprendizaje
 nn: puntero a nuestra red neuronal
 */

void setNetworkDefaults(Network *nn){
    
    // Fijamos la funcion de activacion que usara cada capa
    nn->hidLayerActType = SIGMOIDE;
    nn->outLayerActType = SIGMOIDE;
    
    // Fijamos el valor de la taza de aprendizaje que usaremos
    nn->learningRate    = 0.2;
    
}




/*
Inicializamos los pesos de la capa con valores aleatorios
nn: puntero a nuestra red neuronal
ltype: tipo de capa a inicializar
 */

void initWeights(Network *nn, LayerType ltype){
    
    int nodeSize = 0;
    // Si la capa es del tipo oculta: el tamaño de cada nodo es correspondiente al tamaño de un nodo de la capa oculta
    if (ltype == HIDDEN) 
        nodeSize=nn->hidNodeSize;
     
    // Si no, el tamaño de cada nodo es correspondiente al tamaño de un nodo de la capa de salida             
    else 
        nodeSize=nn->outNodeSize;
    
    // Obtenemos la capa correspondiente
    Layer *l = getLayer(nn, ltype);
    
    // Usamos un puntero para movernos en los nodos
    uint8_t *Nodeptr = (uint8_t*) l->nodes;

    // Recorremos cada nodo para fijar sus pesos con valores entre 0 y 0.7
    for (int o=0; o<l->Nnodes;o++){
    
        Node *n = (Node *)Nodeptr;
        
        for (int i=0; i<n->Nweights; i++){
            n->weights[i] = 0.7*(rand()/(double)(RAND_MAX));
            if (i%2) n->weights[i] = -n->weights[i];  // Hacemos que la mitad de los pesos sean negativos
        }
        
        // Avanzamos nodo a nodo
        Nodeptr += nodeSize;
    }
    
}



/*Se crea la red que contiene tres capas: entrada, capa oculta y salida
 inpCount = numero de nodos en la entrada
 hiCount = numero de nodos en la capa oculta
 outCount = numero de nodos ne la salida */

Network *createNetwork(int inpCount, int hidCount, int outCount){
    
    // Tamaño de cada nodo en la capa de entrada
    int inpNodeSize     = sizeof(Node);         
    // Se calcula el tamaño de la capa de entrada
    int inpLayerSize    = sizeof(Layer) + (inpCount * inpNodeSize);
    
    // El numero de pesos sera igual al numero de nodos de la capa de entrada
    int hidWeightsCount = inpCount;
    // Tamaño de cada nodo en la capa oculta
    int hidNodeSize     = sizeof(Node) + (hidWeightsCount * sizeof(double));
    // Se calcula el tamaño de la capa oculta
    int hidLayerSize    = sizeof(Layer) + (hidCount * hidNodeSize);
    
    // El numero de pesos sera igual al numero de nodos de la capa oculta
    int outWeightsCount = hidCount;
    // Tamaño de cada nodo en la capa de salida
    int outNodeSize     = sizeof(Node) + (outWeightsCount * sizeof(double));
    // Se calcula el tamaño de la capa de salida
    int outLayerSize    = sizeof(Layer) + (outCount * outNodeSize);
    
    // Se reserva mememoria para el tamaño de la red completa
    Network *nn = (Network*)malloc(sizeof(Network) + inpLayerSize + hidLayerSize + outLayerSize);
    
    // Guardamos los tamaños de cada capa y nodo de la red
    nn->inpNodeSize     = inpNodeSize;
    nn->inpLayerSize    = inpLayerSize;
    nn->hidNodeSize     = hidNodeSize;
    nn->hidLayerSize    = hidLayerSize;
    nn->outNodeSize     = outNodeSize;
    nn->outLayerSize    = outLayerSize;
    
    // Inicializamos la red, creando el numero de nodos correspondiente en cada capa
    initNetwork(nn, inpCount, hidCount, outCount);
    
    // Iniciamos con valores por default la red
    setNetworkDefaults(nn);
    
    // Inicializamos los pesos con valores aleatorios
    initWeights(nn, HIDDEN);
    initWeights(nn, OUTPUT);
    
    return nn;
}




/**
Regresa el indice del nodo con mayor valor de salida, de esta manera si el nodo en la posicion "3" es el del valor
de salida mas alto, entonces la prediccion de nuestra red neuronal es que el valor de entrada fue un "3"
nn: puntero a la red neuronal
 */

int getNetworkClassification(Network *nn){
    
    Layer *l = getLayer(nn, OUTPUT);
    
    // Variables que guardan el maximo valor de salida y el indice correspondiente
    double maxOut = 0;
    int maxInd = 0;
    
    // Se recorren los nodos de salida y se busca cual nodo tiene el valor mas alto de salida
    // Este nodo nos dira la prediccion de nuestra red, mediante su posicion
    for (int i=0; i<l->Nnodes; i++){
        
        Node *on = getNode(l,i);

        if (on->output > maxOut){
            maxOut = on->output;
            maxInd = i;
        }
    }
    
    return maxInd;
}