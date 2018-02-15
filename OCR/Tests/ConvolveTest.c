#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int convolve(float *image, float *window, float *output, int iX, int iY, int wX, int wY)
{	
	//Intentionally flooring these divisions.
	int wXC = wX / 2;
	int wYC = wY / 2;
	
	/*unsigned int oXS = 0;
	unsigned int oXE = 0;
	unsigned int iXS = 0;
	unsigned int iXE = 0;
	unsigned int oYS = 0;
	unsigned int oYE = 0;
	unsigned int iYS = 0;
	unsigned int iYE = 0;*/
	int oLoc = 0;
	float currentOut = 0.0;
	int wLoc = 0;
	int iLoc = 0;
	int iXLoc = 0;
	int iYLoc = 0;
	int n = 0;
	int m = 0;
	int i = 0;
	int j = 0;
	int wXS = 0;
	int wYS = 0;
	int wXE = 0;
	int wYE = 0;
	
	for(n = 0; n < iX; n++)
	{
		for(m = 0; m < iY; m++)
		{
			oLoc = iY * n + m;
			if(n - wXC < 0)
			{
				wXS = wXC - n;
			}
			else
			{
				wXS = 0;
			}
			
			if(n + wXC >= iX)
			{
				wXE = (wX-1 - wXC) + (iX-1 -n) + 1;
			}
			else
			{
				wXE = wX;
			}
			
			if(m - wYC < 0)
			{
				wYS = wYC - m;
			}
			else
			{
				wYS = 0;
			}
			
			if(m + wYC >= iY)
			{
				wYE = (wY-1 - wYC) + (iY-1 -m) + 1;
			}
			else
			{
				wYE = wY;
			}
			
			currentOut = 0.0;
			for(i = wXS; i < wXE; i++)
			{
				for(j = wYS; j < wYE; j++)
				{
					wLoc = wY * i + j;
					iXLoc = n + (i - wXC); 
					iYLoc = m + (j - wYC);
					iLoc = iY * iXLoc + iYLoc;
					currentOut = currentOut + window[wLoc] * image[iLoc];
				}
			}
			output[oLoc] = currentOut;
			
		}
	}
	
	return 0;
}
		

int main()
{
	srand(time(NULL));
	
	const int wSize = 5;
	const int iSize = 28;
	const int wSizeSq = wSize*wSize;
	const int iSizeSq = iSize*iSize;
	float testW[wSizeSq];
	float testI[iSizeSq];
	float testO[iSizeSq];
	
	int num =0;
	printf("Input number of convolutions to run: ");
	scanf("%d", &num);
	
	clock_t start = clock(), diff;
	clock_t startSingle;
	clock_t diffSingle;
	clock_t randStart;
	clock_t diffRand;
	clock_t randTotal = 0;
	clock_t convTotal = 0;
	int n = 0;
	int k = 0;

	for(n= 0; n < num; n++)
	{
		randStart = clock();
		for(k = 0; k < iSizeSq; k++)
		{
			testI[k] = ((float)rand())/((float)RAND_MAX);
		}
		for(k = 0; k < wSizeSq; k++)
		{
			testW[k]= ((float)rand())/((float)RAND_MAX);
		}
		diffRand = clock() - randStart;
		randTotal = randTotal + diffRand;
		startSingle = clock();
		convolve(testI, testW, testO, iSize, iSize, wSize, wSize);
		diffSingle = clock()-startSingle;
		convTotal = convTotal + diffSingle;
	}
	diff = clock() - start;
	int i = 0;
	int j = 0;
	/*for(i = 0; i < 4; i++)
	{
		for(j = 0; j < 4; j++)
		{	
			printf(" ");
			printf("%f",testO[i*4+j]);
		}
		printf("\n");
	}*/
	float secT = diff  / CLOCKS_PER_SEC;
	printf("Total time: %f\n", secT);
	float secC = convTotal / CLOCKS_PER_SEC;
	printf("Convolve time: %f\n", secC);
	float secR = randTotal / CLOCKS_PER_SEC;
	printf("Generation time: %f\n", secR);
	return 0;
	
}

	/*if(i < wXC)
		{
			oXS = wXC - i;
			oXE = iX;
			iXS = 0;
			iXE = iX - (wXC - i);
		}
		else
		{
			oXS = 0;
			oXE = iX - (i - wXC);
			iXS = i - wXC;
			iXE = iX;
		}*/
		
			/*if(j < wYC)
			{
				oYS = wYC - j;
				oYE = iY;
				iYS = 0;
				iYE = iY - (wYC - j);
			}
			else
			{
				oYS = 0;
				oYE = iY - (j - wYC);
				iYS = j - wYC;
				iYE = iY;
			}*/
			
						//output[outputXStart:outputXEnd,outputYStart:outputYEnd] = output[outputXStart:outputXEnd,outputYStart:outputYEnd] + window[-(i+1),-(j+1)] * image[imageXStart:imageXEnd,imageYStart:imageYEnd]
