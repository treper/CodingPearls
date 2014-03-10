//leet code solutions
#include <iostream>
#include <vector>
using namespace std;

void removeDuplicateSortedArray(int A[], int n)
{
//类似合并两个已排序的数组,两个指针分别指向各个数组，只不过这里的数组是同一个
	if(n == 0)
		return 0;
	int index = 0;
	for(int i = 0; i<n; i++){
		if(A[index] != A[i])
		{
			A[++index] = A[i];
		}
	}
	return ++index;
}

void removeDuplicateSortedArray(int A[], int n)
{
	return distance(unique(A,A+n));
}
//错误思路
int FindInRotatedSortedArray(int A[],int n,int target)
{
	//默认为升序排列
	int first = 0, last = n;
	while(first != last)
	{
		int m = (first + last)/2;
		if(A[m]==target)
			return m;
		else if(A[m]<target)//6789012345 find 8
		{
			//按正常的二分查找应该往后找
			if(A[last]>target)
			{
				first = m + 1;
			}
			else//往前找
			{
				last = m - 1;
			}
		}
		else//A[m]>target  3456789012 find 1
		{
			if(A[first]<target)
			{
				last = m + 1;
			}
			else
			{
				first = m - 1;
			}
		}
	}
}
/**
* without duplicates
*　正确思路
*/
int FindInRotatedSortedArray(int A[],int n,int target)
{
	//默认为升序排列
	int first = 0, last = n;
	while(first != last)
	{
		int m = (first + last)/2;
		if(A[m]==target)
			return m;
		if(A[first] <= A[m])//3456789012 find 1
		{
			if(A[first]<=target && target<A[m])
			{
				last = m;
			}
			else
			{
				first = m + 1;
			}

		}
		else//6789012345 find 8
		{
			if(A[m]<=target && target<A[last-1])
			{
				first = m + 1;
			}
			else
			{
				last = m;
			}

		}
	}
}
/*
* with duplicates
*/
int FindInRotatedSortedArrayWithDuplicates(int A[],int n,int target)
{
	//默认为升序排列
	int first = 0, last = n;
	while(first != last)
	{
		int m = (first + last)/2;
		if(A[m]==target)
			return m;
		if(A[first] < A[m])//递增 3456789012 find 1
		{
			if(A[first]<=target && target<A[m])
			{
				last = m;
			}
			else
			{
				first = m + 1;
			}

		}
		else if (A[first]>A[m])//非递增 6789012345 find 8
		{
			if(A[m]<=target && target<A[last-1])
			{
				first = m + 1;
			}
			else
			{
				last = m;
			}

		}
		else//A[first]==A[m] 不确定是否递增 再看一步
		{
			first++;//
		}
	}
}

/**
* time complexity:O(n),space complexity:O(n)
*/
int LongestConsequetiveSequence(int A[], int n)
{

	//hash表
	unordered_map<int,bool> used;
	for(int i = 0; i<n; i++)
		used[A[i]]=true;

	int longest = 0;
	for(int i = 0; i<n; i++)
	{
		int l = 0;
		for(int j = A[i]; used[j]!=true; j++)
			l++;
		for(int j = A[i]; used[j]!=true; j--)
			l++;
		longest = max(longest,l);
		

	}
}
/*
* 要求是log(m+n)，否则mergesort类似O(m+n)
* time complexity O(log(m+n)),space complexity O(1)
* Common variations:find kth element of a array
*/
itn FindMedianOfTwoSortedArray(int A[],int m, int B[], int n)
{
	int k = m+n;
	if(k & 0x01)//odd
	{
		return KthOfTwoSortedArray(A,m,B,n,k/2+1);
	}
	else//even
	{
		return (KthOfTwoSortedArray(A,m,B,n,k/2) + KthOfTwoSortedArray(A,m,B,n,k/2+1))/2;
	}

}
int KthOfTwoSortedArray(int A[],int m, int B[], int n,int k)
{
	//example 1
	//1,2,3,4,5,6 len:6
	//4,5,8,9,10,11,12,14,15,15 len:10
	//median is 6
	//example 2
	//1,2,3,4,5,6,7,8,9,10,11,12,13 len:13
	//4,5,8,9,10,11,12,14,15,15 len:10
	//median is 8
	//example 3 no same element
	//1,2,3,4,5,6,7,8
	//9,10,11,12,14,15,23,24,26,36 
	if(m>n)//默认m<n
	{
		return KthOfTwoSortedArray(B,n,A,m,k);
	}
	if(m==0)//空了
	{
		return B[k-1];
	}
	if(k==1)
	{
		return min(A[0],B[0]);
	}

	int pa = min(k/2, m);
	int pb = k-pa

	if(A[pa-1]<B[pb-1])
	{
		return KthOfTwoSortedArray(A+pa,m-pa,B,n,k-pa);
	}
	else if(A[pa-1]>B[pb-1])
	{
		return KthOfTwoSortedArray(A,m,B+pb,n-pb,k-pb);
	}
	else
		return A[pa-1];
}
//非递归方法
int KthOfTwoSortedArrayNotRecursive(int A[],int m, int B[], int n,int k)
{}
int 3sum(int A[], int n, int target)
{
	
}

int main()
{

}