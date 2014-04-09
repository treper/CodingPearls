#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

/*int MininumEditDistance(string& a, string& b)
{
	//transformatino fomular : 
	//
	//E[i,j]=min(E[i-1,j-1]+diff(i,j),E[i,j-1]+1,E[i-1,j]+1)
	if (a[a.size()-1]==b[b.size()-1])
	{
		return MininumEditDistance(a.substr(0,a.size()-1),b.substr(b.size()-1));
	}
	else{
		return min(MininumEditDistance(a.substr(0,a.size()-1),b)+1 , MininumEditDistance(a,b.substr(0,b.size()-1))+1);
	}



}*/
int MininumEditDistance(const char* a,int m,const char* b,int n)
{
	//transformatino fomular : 
	//
	//E[i,j]=min(E[i-1,j-1]+diff(i,j),E[i,j-1]+1,E[i-1,j]+1)
	if (a[m-1]==b[n-1])
	{
		return MininumEditDistance(a,m-1,b,n-1);
	}
	else{
		return min(MininumEditDistance(a,m-1,b,n)+1 , MininumEditDistance(a,m,b,n-1)+1);
	}



}

void MininumEditDistanceDP(char a[],char b[])
{
	int m = strlen(a)+1;
	int n = strlen(b)+1;
	int* E = new int[m*n];
	//init base situation
	for (int i=0;i<m;i++)
	{
		//E[i][0]=i;
		*(E+i*n)=i;
	}
	for (int i=0;i<n;i++)
	{
		//E[0][n]=i;
		*(E+i)=i;
	}
	for (int i = 1;i<m;i++)
	{
		for (int j=1;j<n;j++)
		{
			/*int e1 = E[i][j-1]+1;
			int e2 = E[i-1][j]+1;
			int e3 = E[i-1][j-1]+a[i]!=b[j];
			*/
			
			int e1 = *(E+i*n+j-1)+1;
			int e2 = *(E+(i-1)*n+j)+1;
			int e3 = 0;
			if(a[i]!=b[j])
			{
				e3 = *(E+(i-1)*n+j-1)+1;
			}
			else
			{
				e3 = *(E+(i-1)*n+j-1);
			}
			*(E+i*n+j) = min(min(e1,e2),e3);
		}
	}

	cout<<*(E+m*n-1)<<endl;
	delete E;

}
