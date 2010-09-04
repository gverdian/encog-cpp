#include <stdio.h>
#include <stdlib.h>
#include "encog.h"

short defaultParamLength[] = { 
	1, // ACTIVATION_LINEAR
	1, // ACTIVATION_TANH
	1, // ACTIVATION_SIGMOID
	0, // ACTIVATION_SOFTMAX
	0, // ACTIVATION_BIPOLAR
	4, // ACTIVATION_STEP
	0, // ACTIVATION_RAMP
	0, // ACTIVATION_COMPETITIVE
	0, // ACTIVATION_SIN
	0, // ACTIVATION_LOG
	0 // ACTIVATION_GAUSSIAN
};

void EncogCreateFlatNetwork(short layerCount, FlatLayer *layers, FlatNetwork **result)
{
	int paramCount = 0;
	
	for(int i=0;i<layerCount;i++)
	{
		paramCount+=layers[i].paramsLength;
	}
	
	int size = sizeof(FlatNetwork);
	size+=paramCount*sizeof(double);
	size+=10*sizeof(unsigned short)*layerCount;
	
	FlatNetwork *flat = calloc(size,1);
	*result = flat;
	flat->size = size;
	flat->layerCount = layerCount;
	flat->inputCount = layers[0].count;
	flat->outputCount = layers[layerCount-1].count;
	
	void *ptr = flat->memory;
	flat->params = ptr; ptr+=paramCount;
	flat->layerCounts = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->layerContextCount = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->weightIndex = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->layerIndex = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->activationType = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->layerFeedCounts = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->contextTargetOffset = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->contextTargetSize = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->biasActivation = ptr; ptr+=layerCount*sizeof(unsigned short);
	flat->paramIndex = ptr; ptr+=layerCount*sizeof(unsigned short);
	
	
	double *paramPtr = flat->params;
	
	unsigned short index = 0;
	unsigned short neuronCount = 0;
	unsigned short weightCount = 0;

	unsigned short currentParamIndex = 0;

	for(int i=layerCount-1;i>=0;i--) {
			
		FlatLayer *layer = &layers[i];
		
		for( int j=0;j<layer->paramsLength;j++)
		{
			flat->params[currentParamIndex++] = layer->params[j];
		}
		
		FlatLayer *nextLayer = NULL;
			
		if( i>0 )
			nextLayer = &layers[i-1];
			
		flat->biasActivation[index] = layer->biasActivation;
		flat->layerCounts[index] = layer->totalCount;
		flat->layerFeedCounts[index] = layer->count;
		flat->layerContextCount[index] = layer->contectCount;
		flat->activationType[index] = layer->activation;
		flat->paramIndex[index]  = currentParamIndex;
/*		currentParamIndex = ActivationFunctions.copyParams(layer.getParams(),this.params,currentParamIndex);

		neuronCount += layer.getTotalCount();
			
		if( nextLayer!=null )
			weightCount+=layer.getCount()*nextLayer.getTotalCount();

		if (index == 0) {
			this.weightIndex[index] = 0;
			this.layerIndex[index] = 0;
		} else {
			this.weightIndex[index] = this.weightIndex[index - 1]
				+ (this.layerCounts[index] * this.layerFeedCounts[index - 1]);
			this.layerIndex[index] = this.layerIndex[index - 1]
				+ this.layerCounts[index - 1];
		}
			
		int neuronIndex = 0;
		for(int j=layers.length-1;j>=0;j--)
		{
			if( layers[j].getContextFedBy()==layer)
			{
				this.contextTargetSize[i] = layers[j].getContectCount();
				this.contextTargetOffset[i] = neuronIndex+layers[j].getTotalCount()-layers[j].getContectCount();
			}
			neuronIndex+=layers[j].getTotalCount();
		}

		index++;*/
	}

}

void EncogSetupFlatLayer(
	FlatLayer *layer,
	unsigned short count, 
	unsigned short activation, 
	double biasActivation, 
	struct FlatLayer *contextFedBy, 
	unsigned short paramLength)
{
	layer->count = count;
	layer->activation = activation;
	layer->biasActivation = biasActivation;
	layer->contextFedBy = contextFedBy;
	layer->params = calloc(sizeof(double),paramLength);
	layer->paramsLength = paramLength;
}

void EncogSetupFlatLayerSimple(
	FlatLayer *layer,
	unsigned short count, 
	unsigned short activation)
{
	EncogSetupFlatLayer(
		layer,
		count,
		activation,
		1.0,
		NULL,
		defaultParamLength[activation]);
}

