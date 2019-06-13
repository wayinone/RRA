#include <stdio.h>
#include <stdlib.h>
void exactsol_m3(float distxz_nN[],int n,int N,int Zid[]);

void exactsol_m3(float distxz_nN[],int n,int N,int Zid[]){
	float current_min=1000.0;
	int j,i3,i2,i1;
	float f3,f2,f1,fv_321,val_add;
	float *f32 = malloc(sizeof(float)*n);
	for (i3=0;i3<N-2;i3++){
		for (i2=i3+1;i2<N-1;i2++){
			for (j=0;j<n;j++){
				f3=distxz_nN[j*N+i3];
				f2=distxz_nN[j*N+i2];
				f32[j] = (f3>f2)? f2:f3;
			}
			for (i1=i2+1;i1<N;i1++){
				fv_321=0;
				for (j=0;j<n;j++){

					f1=distxz_nN[j*N+i1];
					val_add = (f1>f32[j]) ? f32[j]:f1;
					fv_321+=val_add;
				}
			if (fv_321<current_min){
				current_min=fv_321;
				Zid[0] = i3;
				Zid[1] = i2;
				Zid[2] = i1;}

			}
		}
	}
}

//void exactsol_m3(float distxz_nN[],int n,int N,int Zid[]){
//	float current_min=1000.0;
//	int j,i3,i2,i1;
//	float f3,f2,f1,fv_321;
//	for (i3=0;i3<N-2;i3++){
//		for (i2=i3+1;i2<N-1;i2++){
//			for (i1=i2+1;i1<N;i1++){
//				fv_321=0;
//				for (j=0;j<n;j++){
//					f3=distxz_nN[j*N+i3];
//					f2=distxz_nN[j*N+i2];
//					f1=distxz_nN[j*N+i1];
//					if (f3>f2){
//						if (f2>f1){
//							fv_321+=f1;}
//						else{
//							fv_321+=f2;}}
//					else{
//						if (f3<f1){
//							fv_321+=f3;
//						}
//						else{
//							fv_321+=f1;
//						}
//					}
//				}
//			if (fv_321<current_min){
//				current_min=fv_321;
//				Zid[0] = i3;
//				Zid[1] = i2;
//				Zid[2] = i1;}
//
//			}
//		}
//	}
//}