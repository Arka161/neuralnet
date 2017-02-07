#include <iostream>
#include <cmath>
#include <ctime>
#include <cerrno>
#include <cstdlib>
using namespace std;

const int numInputs = 3;       
const int numPatterns = 4;      // Input patterns(Total possibility) 
const int numHidden = 4;
const int numEpochs = 200;     
const float LR_IH = 0.8;       // Learning rate(between 0 and 1) 
const float LR_HO = 0.07;      

int patNum = 0;
float normalerror = 0.0;
float outPred = 0.0;          
float errer = 0.0;                  // error

float hiddenVal[numHidden] = {0.0};            

float weightsIH[numInputs][numHidden]; // Input to Hidden weights.
float weightsHO[numHidden] = {0.0};            // Hidden to Output weights.

int trainInput[numPatterns][numInputs];
int trainOutput[numPatterns];         

// Functions used
void initWeight();
void calcNet();
void WeightChangesHO();
void WeightChangesIH();
void calcOverallError();
void initData();
void displayResults();
float getRand();


int main(){
    initWeight();
	initData();

    // Train
    for(int j = 0; j <= numEpochs; j++){

        for(int i = 0; i < numPatterns; i++){

    
            patNum = rand() % numPatterns;

            calcNet();

   
            WeightChangesHO();
            WeightChangesIH();
        }

        calcOverallError();
    }

    displayResults();

    return 0;
}

void initWeight(){
//Make weights to random values

    for(int j = 0; j < numHidden; j++){

        weightsHO[j] = (getRand() - 0.5) / 2;
        for(int i = 0; i < numInputs; i++){

            weightsIH[i][j] = (getRand() - 0.5) / 5;
            cout << "Weight = " << weightsIH[i][j] << endl;
        }
    }
}

void initData(){

    // the range -1 to 1.



    trainInput[0][0]   =  1;	trainInput[0][1]   = -1;	trainInput[0][2]   =  1; 	trainOutput[0]      =  1;

    trainInput[1][0]   = -1;	trainInput[1][1]   =  1;	trainInput[1][2]   =  1; 	trainOutput[1]      =  1;

    trainInput[2][0]   =  1;
    trainInput[2][1]   =  1;
    trainInput[2][2]   =  1; 
    trainOutput[2]      = -1;

    trainInput[3][0]   = -1;
    trainInput[3][1]   = -1;
    trainInput[3][2]   =  1; 
    trainOutput[3]      = -1;
}

void calcNet(){
// Calculates values for Hidden and Output nodes.
    for(int i = 0; i < numHidden; i++){
	  hiddenVal[i] = 0.0;

        for(int j = 0; j < numInputs; j++){
	        hiddenVal[i] = hiddenVal[i] + (trainInput[patNum][j] * weightsIH[j][i]);
        }

        hiddenVal[i] = tanh(hiddenVal[i]);        //hyberbolic func. 
    }

    outPred = 0.0;

    for(int i = 0; i < numHidden; i++){
        outPred = outPred + hiddenVal[i] * weightsHO[i];
    }
    normalerror = outPred - trainOutput[patNum];
}

void WeightChangesHO(){

    for(int k = 0; k < numHidden; k++){
        float weightChange = LR_HO * normalerror * hiddenVal[k];
        weightsHO[k] = weightsHO[k] - weightChange;

    }
}

void WeightChangesIH(){
// Adjust the Input to Hidden weights.

    for(int i = 0; i < numHidden; i++){

        for(int k = 0; k < numInputs; k++){

            float x = 1 - (hiddenVal[i] * hiddenVal[i]);
            x = x * weightsHO[i] * normalerror * LR_IH;
            x = x * trainInput[patNum][k];
            float weightChange = x;
            weightsIH[k][i] = weightsIH[k][i] - weightChange;
        }
    }
}

void calcOverallError(){
    errer = 0.0;

    for(int i = 0; i < numPatterns; i++){
         patNum = i;
         calcNet();
         errer = errer + (normalerror * normalerror);
    }

    errer = errer / numPatterns;
    errer = sqrt(errer);
}

void displayResults(){
    for(int i = 0; i < numPatterns; i++){
        patNum = i;
        calcNet();
        cout << "pat = " << patNum + 1 << 
                " actual = " << trainOutput[patNum] << 
                " neural model = " << outPred << endl;
    }
}

float getRand(){
    return float(rand() / float(RAND_MAX));
}
