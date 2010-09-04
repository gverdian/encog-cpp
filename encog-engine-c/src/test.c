#include <stdio.h>
#include "encog.h"

int main(int argc, char *argv[])
{
	FlatLayer layers[3];
	
	EncogSetupFlatLayerSimple(&layers[0],2,ACTIVATION_SIGMOID);
	EncogSetupFlatLayerSimple(&layers[1],3,ACTIVATION_SIGMOID);
	EncogSetupFlatLayerSimple(&layers[2],1,ACTIVATION_SIGMOID);
	
	FlatNetwork *flat;
	EncogCreateFlatNetwork(3,layers,&flat);
	printf("%d",flat->size);
	
	return 0;
}