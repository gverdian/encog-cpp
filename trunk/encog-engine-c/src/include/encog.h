#ifndef __ENCOG_H

#define ACTIVATION_LINEAR 0
#define ACTIVATION_TANH 1
#define ACTIVATION_SIGMOID 2
#define ACTIVATION_SOFTMAX 3
#define ACTIVATION_BIPOLAR 4
#define ACTIVATION_STEP 5
#define ACTIVATION_RAMP 6
#define ACTIVATION_COMPETITIVE 7
#define ACTIVATION_SIN 8
#define ACTIVATION_LOG 9
#define ACTIVATION_GAUSSIAN 10

extern short defaultParamLength[];

typedef struct {
	unsigned short size;
	unsigned short layerCount;
	unsigned short inputCount;
	unsigned short outputCount;
	double *params;
	unsigned short *layerCounts;
	unsigned short *layerContextCount;
	unsigned short *weightIndex;
	unsigned short *layerIndex;
	unsigned short *activationType;
	unsigned short *layerFeedCounts;
	unsigned short *contextTargetOffset;
	unsigned short *contextTargetSize;
	unsigned short *biasActivation;
	unsigned short *paramIndex;
	unsigned char memory[];
} FlatNetwork;

typedef struct {
	unsigned short activation;
	unsigned short count;
	unsigned short paramsLength;
	double biasActivation;
	double *params;
	struct FlatLayer *contextFedBy;
} FlatLayer;

void EncogCreateFlatNetwork(short layerCount, FlatLayer *layers, FlatNetwork **result);
void EncogSetupFlatLayer(
	FlatLayer *layer,
	unsigned short count, 
	unsigned short activation, 
	double biasActivation, 
	struct FlatLayer *contextFedBy, 
	unsigned short paramLength);
	

void EncogSetupFlatLayerSimple(
	FlatLayer *layer,
	unsigned short count, 
	unsigned short activation);

#endif