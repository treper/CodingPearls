//leet code solutions
//算法一定要自己写代码，光记忆解法是不行的
#include <iostream>
#include <vector>
using namespace std;

int removeDuplicateSortedArray(int A[], int n)
{
//类似合并两个已排序的数组,两个指针分别指向各个数组，只不过这里的数组是同一个
	if(n == 0)
		return 0;
	int index = 0;
	for(int i = 0; i<n; i++){
		//if(A[index] != A[i] && index!=i)//1,2,3,4,5,6                   1,2,2,3,3,4,5,6==>1,2,3,4,5,6
		if(A[index] != A[i])
		{
			A[++index] = A[i];
		}
	}
	return ++index;//the final length
}

void removeDuplicateSortedArray(int A[], int n)
{
	return distance(unique(A,A+n));
}

//implement strStr
//complexity O(m*n),space complexity O(1)
char* strStr(const char* haystack,const char* needle)
{
	if(haystack == NULL || needle == NULL) return NULL;
	int m = strlen(needle);
	int n = strlen(haystack);
	for(int i = 0;i<n-m;i++){
		int j =0;
		while(j<m && haystack[i+j] == needle[j])
			j++;
		if(j==m) return haystack + i;
	}
	return NULL;
}

//complexity O(m+n),space complexity O(n)
char* strStrKMP(const char* haystack,const char* needle)
{
	int m = strlen(needle);
	int n = strlen(haystack);

	//generate longest prefix sufix array
	vector<int> lps(m,0);

	int len=0,i=1;
//patern string:AAABAAA [0,1,2,0,1,2,3]
	while(i<n){
		if(haystack[i] == haystack[len])
		{
			len++;
			lps[i]=len;
			i++;
		}
		else
		{
			if(len>0)
			{
				len=lps[len-1]
			}
			else
			{
				lps[i]=0;
				i++;
			}
		}
	}

	//kmp match
	//patern string:AAABAAA [0,1,2,0,1,2,3]
	//haystack string:AAAAABAAA
	//i=j=4 -->需要转移4-lps[3]=2位
	//??将i移动的位数=当前已匹配到的位置(j)-lps[(已经匹配数目)j-1]=j-lps[j-1]
	// i往前等效于j后退 将j=j-(j-lps[j-1])=lps[j-1]
	while(true)
	{
		if(haystack[i] == needle[j])
		{
			i++;j++;
			if(j==m) return haystack + i - m;//i==j contains pattern length m
			//if need continue to find other occurences do the following line
			//j=lps[j-1];
		}
		else if(j>0)//first time compare or the j has been set to 0 by previous lps[j]
				j = lps[j-1]
		else
				i++;

	}
	return NULL;
}

//unique paths
//dynamic programming 
//step[i][j]=step[i-1][j]+step[i][j-1]
int UniquePaths(vector<vector<int> >& map)
{
	int rows = map.size(),cols = map[0].size();
	vector<vector<int> > step(rows,vector<int>(cols,0));
	//init status
	//first row
	for(int i=0;i<cols;i++)
		step[0][i]=1;
//first column
	for(int i=0;i<rows;i++)
		step[i][0]=1;

	for(int i =0;i<rows;i++)
		for(int j =0;j<cols;j++)
		{
			step[i][j]=step[i-1][j]+step[i][j-1]
		}

	return step[rows-1][cols-1];


}

int UniquePathsWithObstacle(vector<vector<int> >& map)
{
	//colums in first row after obstacl has 0 method
}

//Word Ladder I
//hit -> dog
//given a dict['big','doc','zip'] size is fixed find shortest path
//give step else 0
//bfs is more appropriate for shortest problem


void dfs(vector<string>& dict,
	string& start,
	string& end,
	unordered_set<string>& visited,
	vector<string>& path,
	vector<vector<string> >& results)
{
	if(start==end)//found target
	{
		path.push_back(end);
		results.push_back(path);
		return;
	}
	string newword=start;
	for(int i=0;i<dict[0].size();i++)
	{
		for(char c='a';c<='z';c++)
		{
			if(newword[i]==c)
				continue;
			newword[i]=c;
			if(dict.count(newword)!=0 && !visited[newword])
			{
				path.push_back(newword);
				dfs(dict,newword,end,path,visited);
				path.pop_back(newword);
			}
		}
		
	}

}
int WordLadderPath(vector<string>& dict,string& start,string& end)
{
	int step=0;
	vector<string> path;
	//path.push_back(start);
	dfs(dict,start,end,path);

}
//Breath First Search
void WordLadderII(unordered_set<string>& dict,string& start, string& end)
{
	unordered_set<string> current,next;
	unordered_set<string> visited;
	unordered_map<string,vector<string> > parent;
	current.insert(start);
	while(!current.empty() && !current.count(end))
	{
		for(auto & cur:current) visited.insert(cur);
		for(auto & cur:current) 
		{
			string newword=cur;
			for(int i=0;i<newword.size();i++)
			{
				for(char c='a';c<'z';c++)
				{
					if(newword[i]==c) continue;
					char t = newword[i];
					newword[i]=c;
					if(newword==end ||(dict.count(newword)!=0 && visited.count(newword)==0))
					{
						parent[newword].push_back(cur);
						next.insert(newword);
					}
					newword[i]=t;
				}
			}
		}
		current.clear();
		swap(current,next);
	}
	vector<string> path;
	vector<vector<string> > resutls;
	buildPath(start,end,parent,path,results);
}

//parent is a tree like structure ,DFS to build path
void buildPath(string& start,string& end, unordered_map<string,vector<string> >& parent, vector<string>& path,vector<vector<string> > results)
{
	if(current==end)
	{
		results.push_back(vector<string>(path.rbegin(),paht.rend()));
		return;
	}
	for(auto & next:parent[start])
	{
		path.push_back(next);
		buildPath(next,end,parent,path,results);
		path.pop_back();
	}


}

int WordLadderStepCount(unordered_set<stirng>& dict,string& start,string& end)
{
	unordered_set<string> visited;
	queue<string> current,next;
	current.push_back(start);visited.insert(start);
	int level=1;
	while(!current.empty())
	{
		while(!current.empty())
		{
			string newword = current.front();
			current.pop();
			for(int i=0;i<newword.size();i++)
			{
				for(char c='a';c<='z';c++)
				{
					if(newword[i]==c)continue;
					char t = newword[i];
					newword[i]=c;
					if(newword==end)
						return level+1;
					if(dict.count(newword)!=0 && visited.count(newword)==0)
					{
						next.insert(newword);
					}
				}
			}
		}
		swap(current,next);
		level++;
	}
	return 0;
}


void SurroundedRegions(vector<vector<string> >& board)
{
	if(board.empty()|| board[0].empty()) return;
	int m = board.size(),n = board[0].size();
	//把边缘点及其邻域点中为‘O’的排除，都标记为*
	for(int i=0;i<m;i++)
	{
		if(board[i][0]=='O') bfs(board,m,n,i,0);
		if(board[i][n-1]=='O') bfs(board,m,n,i,n-1);
	}

	for(int i=0;i<n;i++)
	{
		if(board[0][i]=='O') bfs(board,m,n,0,i);
		if(board[m-1][i]=='O') bfs(board,m,n,m-1,i);
	}


	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++) 
		{
                if (board[i][j] == '*') board[i][j] = 'O';
                else if (board[i][j] == 'O') board[i][j] = 'X';
        }
    }

}
void bfs(vector<vector<string> >& board,int m,int n,int i,int j)
{
	board[i][j]='*';
	queue<int> qs;
	qs.push(i*n+j);
	while(!qs.empty())
	{
		i=qs.front()/n,j=qs.front()%n;
		qs.pop();
		for(int k=0;k<4;k++)
		{
			int ni=i+di[k],nj=j+dj[k];
			if(ni==-1||ni==m||nj==-1||nj==n||board[ni][nj]!='O') continue;
			board[ni][nj]='*';
			qs.push(ni*n+nj);
		}
	}

}


//
//aab --> [[aa,b],[a,a,b]]
//complexity is O(2^n),space complexity is O(n)
//str[0,prev-1]已经处理，保证是回文串
//prev 表示str[prev-1]与str[prev]之间的空隙位置,start也同理
PalindromePartitionDFS(string& input,int prev,int start,vector<string>& intermidates,vector<string>& results)
{

	if(start==input.size())//
	{
		if(isPalindrome(str,prev,start-1))
		{
			intermidates.push_back(input.substr(prev,start-prev));
			results.push_back(intermidates);
			intermidates.pop_back();
		}	
	}
	//不断开
	PalindromePartitionDFS(input,prev,start+1,intermidates,results);
	//如果str[prev,start-1]是回文,可以断开，也可以不断开，不断开为上一行
	if(isPalindrome(input,prev,start-1))
	{
		intermidates.push_back(input.substr(prev,start-prev));
		PalindromePartitionDFS(input,start,start+1,intermidates,results);
		intermidates.pop_back();
	}

}
bool isPalindrome(string& str,int start,int end)
{
	while(start<end)
	{
		if(s[start++]!=s[end--]) return false;

	}
	return true;
}
//
PalindromePartitionDP(string& str)
{
	int n = str.size();
	vector<vector<bool> > isPalindrome(n,vector<bool>(n,false));
	vector<vector<vector<string> > > dp2(n,vector<vector<string > >())
	for(int i=0;i<n;i++)
	{
		for(int j=i;j>0;j--)
		{
			if(str[i]==s[j] && (i-j<2 ||str[j+1]==str[i-1]))
			{
				isPalindrome[i][j]=true;
			}
			if(j==0)
			{
				dp2[i].push_back(vector<string>(1,s.substr(0,i-j+1)));
			}
			else
			{
				for(auto p:dp2[j-1])
				{
					p.push_back(s.substr(i,l));
					dp2[i].push_back(p);
				}
			}
		}
		
	}
	return dp2[n-1];
}
RestoreIPAddress(string& str,int step,string& intermidates,vector<string>& results)
{

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

	int pa = min(k/2, m);//m<n
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
		return A[pa-1];//or return B[pb-1]
}
//非递归方法
int KthOfTwoSortedArrayNotRecursive(int A[],int m, int B[], int n,int k)
{

}
/*
*   time complexity n^2log2n
*/
int 3sum(int A[], int n, int target)
{
	qsort(A,n);//STL
	for(int i=0;i<n-2;i++)
	{
		int remain1 = target - A[i];
		for(int j=i+1;j<n-1;j++)
		{
			int remain2 = remain1 - A[j];
			//find the third element log2n
			int idx = BinarySearch(A+j+2,n-j,remain2);
			if(idx!=-1)
				printf("%d %d %d\n",A[i],A[j],A[idx]);
		}
	}
}
int BinarySearch(int A[],int n, int target)
{
	if(A[n/2]<target)
	{
		return BinarySearch(A+n/2,n/2,target);
	}
	else if(A[n/2]>target)
	{
		return BinarySearch(A,n/2-1,target);
	}
	else
		return n/2;
}
/*
* n^3log2n
*/
int 4sum()
{

}

/*
* Remove specific element
*/
int RemoveElement(int A[],int n, int target)
{
	int idx = 0;
	for(int i=0;i<n;i++)
	{
		if(A[i]!=target)
			A[idx++]=A[i];
	}
	return idx;

}

/*
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place, do not allocate extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1
*/

/*
step 1:from right to left find the first permutation number that vilate the increase trend
step 2:from right to left find the first change number larger than permutation number
step 3:swap permutation number and change number
step 4:reverse from patition to end
5 4 8 9 2 1 4 3
step 1: 1
step 2: 3
step 3:  5 4 8 9 2 3 4 1
step 3:  5 4 8 9 2 3 1 4
*/
void NextPermutation(vector<int>& A)
{
	int i,n=A.size();
	for (i = n; i >=0; i--)
	{
		if(A[i]>A[i-1])
			break;
	}
	if(i==0)
	{
		reverse(A.begin(),A.end());
		return;
	}
		
	int pn = A[i-1];//permutaion number
	for (i = n; i >=0; i--)
	{
		if(A[i]>pn)
			break;
	}
	int cn =A[i];
	reverse(A.begin()+i,A.end());
	return;
}
/*
* permutation sequence
* 123 132 213 231 312 321
* 如果第一个数确定，则后面的permutation有(n-1)!种,(n-1)!之后第一个数换，之后以此类推
* 通过k除以后面permutaion 的数目来获得当前数字的index
各个index值推导：
设变量K1 = K
a1 = K1 / (n-1)!

同理，a2的值可以推导为
a2 = K2 / (n-2)!
K2 = K1 % (n-1)!
 .......
a(n-1) = K(n-1) / 1!
K(n-1) = K(n-2) /2!
*/
void KthPermutation(int n,int k)
{
	vector<int> numbers(n,0),result(n,0);
	int permCount = 1;
	for(int i =0;i<n;i++)
	{
		numbers[i]=i+1;//1...n
		permCount *= (i+1);
	}
	k--;
	int index;
	for(int i =n;i>0;i--)
	{
		permCount /= i;
		index = k/permCount;
		result[i-1]=numbers[index];
		
		for (int j =index;j<n-1;j++)
		{
			numbers[j]=numbers[j+1];
		}
		k=k%permCount;
	}
	for(int i =n-1;i>=0;i--)
		cout<<result[i];

}
/*
* E[i,j]=min(E[i-1,j-1]+diff(i,j),E[i,j-1]+1,E[i-1,j]+1)
*/
int MininumEditDistanceRecursive(string& a, string& b)
{

	if (a[a.size()-1]==b[b.size()-1])
	{
		return MininumEditDistance(a.substr(0,a.size()-1),b.substr(b.size()-1));
	}
	else{
		return min(MininumEditDistance(a.substr(0,a.size()-1),b)+1 , MininumEditDistance(a,b.substr(0,b.size()-1))+1);
	}

}
/*
* E[i,j]=min(E[i-1,j-1]+diff(i,j),E[i,j-1]+1,E[i-1,j]+1)
* time complexity O(mn)
*/
int MinimumEditDistance(string& a,string& b)
{
	vector<vector<int> > E(a.size(),veector<int>(b.size(),0));
	for(int i = 0;i<a.size();i++)
	{
		E[i][0]=i;
	}
	for(int i = 0;i<b.size();i++)
	{
		E[0][i]=i;
	}
	//distance
	for(int i = 1;i<a.size();i++)
	{
		for(int j = 1;j<b.size();j++)
		{
			E[i][j] = min(E[i-1][j-1]+diff(a.at(i),b.at(j)), E[i-1][j]+1, E[i][j-1]+1);
		}
	}
	return E[a.size()-1][b.size()-1];
}
int diff(char a,char b)
{
	if(a==b)
		return 0;
	else
		return 1;
}

bool BinarySearchIn2DMatrix(vector<vector<int> > &matrix,int target)
{
	int row = matrix.size();
	if(row==0)
		return false;
	int col = matrix[0].size()
	if(col==0)
		return false;

	//find row first
	int first=0,last=row-1;
	int mid,target_row;
	while(first!=last)
	{
		mid = (first + last)/2;
		if(target < matrix[mid][0])
		{
			last=mid-1;
		}
		else if(target > matrix[mid][0])
		{
			first=mid+1;
		}
		else
		{
			return true;
		}
	} 
	target_row = mid;
	//search cols
	int first=0,last=col-1;
	while(first!=last)
	{
		mid = (first + last)/2;
		if(target < matrix[target_row][mid])
		{
			last=mid-1;
		}
		else if(target > matrix[target_row][mid])
		{
			first=mid+1;
		}
		else
		{
			return true;
		}		
	}
	return false;

}
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
/*
match[i][j] is bool when s1[0...i-1] and s2[0...j-1] are interleave string of s3[0...i+j-1]
transform fomular:
match[i][j] = 
(s1[i-1]==s3[i+j-1] && match[i-1][j])||(s2[j-1]==s3[i+j-1] && match[i][j-1])
*/
bool InterleaveString(string& s1,string& s2,string& s3)
{
	if(s1.size()+s2.size()!=s3.size())
		return false;

	vector<vector<bool> > match(s1.size()+1,vector<bool>(s2.size()+1,false));
	match[0][0]=true;

	for(int i = 1;i<s1.size();i++)
	{
		if(s1[i-1]==s3[i-1])
		{
			match[i][0]==true;//s2 is empty
		}
		else
			break;//
	}

	for(int i = 1;i<s2.size();i++)
	{
		if(s2[i-1]==s3[i-1])
		{
			match[0][i]==true;//s1 is empty
		}
		else
			break;//
	}

	for(int i = 1;i<s1.size();i++)
	{
		for(int j = 1;j<s2.size();j++)
		{
			match[i][j]=(s1[i-1]==s3[i+j-1] && match[i-1][j])||(s2[j-1]==s3[i+j-1] && match[i][j-1]);
		}
	}

	return match[s1.size()][s2.size()];
}
/*

empty set:
count[0]=1
set {1}:
count[1]=1
set {1,2}:
count[2]=
count[0]*count[1] + //1 as root,so left node is null(count[0]),right is must be 2,only one element so count[1]
count[1]*count[0] //2 as root,so left node is 1(count[1]),right is null so count[0]
set {1,2,3}:
count[3]=
count[0]*count[2] // 1 as root,left must be null(count[0]),right include {2,3} so count[2]
+ count[1]*count[1] // 2 as root,left must be 1(count[1]),right must be 3 so count[1]
count[2]*count[1] // 3 as root,left include {1,2}(count[2]),right must be null(count[2])

transfrom fomular
count[n]=sum(count[i]*count[j]),0<=i,j<n
*/
int UniqueBinarySearchTree(vector<int>& nodes)
{
	vector<int> count(nodes.size()+1,1);

	for(int i=2;i<=nodes.size();i++)
	{
		for(int j=0;j<i;j++)
		{
			count[i]+=count[j]*count[i-1-j];
		}
			
	}
	return count[nodes.size()];
}

/*
transform formular:
count[i][j] is the times string t[0...i] apear in string s[0...j]
if s[i-1]==t[j-1] count[i][j]=count[i][j-1]
else count[i][j]=count[i][j-1]+count[i-1][j-1]
*/
int DistinctSubsequence(string& s,string& t)
{
	vector<vector<int> > count(s.size()+1,vector<int>(t.size()+1,0));
	for(int i=0;i<s.size();i++)
	{
		count[0][i]=1;
	}

	for(int i=0;i<t.size();i++)
	{
		count[i][0]=0;
	}

	for(int i =0;i<t.size();i++)
	{
		for(int j =0;j<s.size();j++)
		{
			if(t[i-1]==s[j-1])
			{
				count[i][j]=count[i][j-1];
			}
			else
			{
				count[i][j]=count[i][j-1]+count[i-1][j-1]
			}
		}
	}

	return count(s.size(),t.sizee());

}
ListNode* BuildBST(ListNode*& root,int start,int end)
{
	int mid = start+(end-start)/2;//avoids overflow
	TreeNode* parent=new TreeNode(root->data);
	parent->left = BuildBST(root->left,start,mid-1)
	root = root->next;
	parent->right = BuildBST(root->right,mid+1,end);
	return parent;
}

TreeNode* ConvertSortedListToBST(ListNode* root)
{
	int len=0;
	ListNode* p = root;
	while(p)
	{
		len++;
		p=p->next;
	}

	return BuildBST(root,0,len-1);
}
/*
1.hash + binary search
2.sort + binary search

*/
int 2sumhash(vector<int>& vec,int target)
{
	map<int,int> m;
	for(int i =0;i<vec.size();i++)
	{
		m[vec[i]]=i;
	}

	vector<int> result;
	for(int i =0;i<vec.size();i++)
	{
		int remain = target-vec[i];
		if(vec.find(remain)!=vec.end())
		{
			result.push_back(m[i]+1);
			result.push_back(m[j]+1);
			break;
		}
	
	}
}
int 2sum(vector<int>& vec,int target)
{
	vector<int> res;
	sort(vec.begin(),vec.end());

	int start = 0,end=vec.size();
	while(start<end){
		int sum = vec[start]+vec[end];
		if(sum==target)
		{
			res.push_back(start.index);
			res.push_back(end);
			break;
		}
		else if(sum<target)
			start++;
		else if(sum>target)
			end--;
	}
}

/*
in order tree
*/

void MorrisTreeTraversal(Node* root)
{
	Node* cur = root;
	Node* pre = NULL;
	while(cur!=NULL)
	{
		if(cur->left==NULL)
		{
			visit(cur->val);
			cur=cur->right;
		}
		else
		{
			//find predecessor
			pre=cur->left;
			while(pre->right!=NULL && pre->right!=cur)
				pre=pre->right;

			if(pre->right==NULL)
			{
				pre->right = cur;
				cur=cur->left;
			}
			else
			{
				pre->right=Null;
				visit(cur->val);
				cur=cur->right;
			}
		}
	}
}

void ReverseSinlyLinkedList(Node*& head)
{
	Node* pre=NULL;
	Node* curr=head;
	while(curr){
		Node* next = curr->next;
		curr->next = pre;
		pre = curr;
		curr = next;
	}
	head = pre;
}

void LevelTree(Node* root,vector<vector<Node*> >& results,int level)
{
	vector<Node*> list;
	if(results.size==level)
	{
		results.push_back(list);
	}
	else
	{
		list = results[level];
	}

	list.push_back(root);

	LevelTree(root->left,results,level+1);
	LevelTree(root->right,results,level+1);
}

void LevelTraversalBinaryTree(Node* root)
{
	vector<vector<Node*> > results;
	LevelTree(root,resutls,0);
}

TreeNode* flatten(TreeNode* root)
{
	if(root->left==NULL && root->right==NULL)
		return root;
	Node* rHead = NULL, lHead = NULL;
	if(root->right!=NULL)
	{
		rHead = flatten(root->right);
	}
	if(root->left!=NULL)
	{
		lHead = flatten(root->left);
		Node* p = root->right;
		root->right=lHead;
		root->left = NULL;
		while(p->right) p =p->right;
	}
	else
	{
		
	}
}
void FlattenBinaryTree(Node* root)
{
	if(root==NULL)
		return;
	flatten(root);

}

void RecoverMissPlacedBinaryTree()
{

}
int main()
{

}