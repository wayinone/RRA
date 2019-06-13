#include <stdio.h>
#include <stdlib.h>
void exactsol_m2(float distxz_nN[],int n,int N,int Zid[]);

void exactsol_m2(float distxz_nN[],int n,int N,int Zid[]){
	float current_min=1000.0;
	int j,i2,i1;
	float f2_j,f1_j,fv_i2i1;
	for (i2=0;i2<N-1;i2++){
		for (i1=i2+1;i1<N;i1++){
			fv_i2i1=0;
			for (j=0;j<n;j++){
				f2_j=distxz_nN[j*N+i2];
				f1_j=distxz_nN[j*N+i1];
				if (f1_j>f2_j){
					fv_i2i1+=f2_j;}
				else{
					fv_i2i1+=f1_j;}}
			if (fv_i2i1<current_min){
				current_min=fv_i2i1;
				Zid[0] = i2;
				Zid[1] = i1;}
		}
	}
}