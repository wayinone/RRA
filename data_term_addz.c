#include <stdio.h>
#include <stdlib.h>
void data_term_addz(float distxz_nK[],float arrmin_xz_n[],int n,int K,float dataterm_add_K[]);

void data_term_addz(float distxz_nK[],float arrmin_xz_n[],int n,int K,float dataterm_add_K[]){
	int k,j;
	float a,aj;
	for (k=0;k<K;k++){
		a=0;
		for (j=0;j<n;j++){
			aj = distxz_nK[j*K+k]-arrmin_xz_n[j];
			if (aj<0){
				a = a+aj;
			}
		dataterm_add_K[k]=a;
		}
	}
}
