// {4, 6, 7}, {1, 2, 3}, {4, 5, 6}, {10, 12, 32}
//longest increasing subsequence
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

using namespace std;

class Box{
public:
	int w,d,h;
	Box(int w, int d, int h){
		this->w =w;
		this->d =d;
		this->h =h;
	}

};

struct sort_desc {
	bool operator()(Box* a, Box* b) {
		return (a->w*a->d > b->w*b->d);
	}
};

bool canUse(map<Box*,int>& RotParent,Box* a, Box* b)
{
	if (RotParent[a]==RotParent[b])
	{
		return false;
	}
	return true;
}

bool canUse(map<Box*,int>& RotParent,set<int>& Used, Box* a)
{
	if (Used.find(RotParent[a])!=Used.end())
	{
		return true;
	}
	return false;
}

//void testBoxStacking()
int main()
{
	string test_input_file="/home/mps/software/TagHierarchy/test_input.txt";
	//string test_input_file="E:\\pdfs\\∞Ÿ∂»‘∆\\projects\\Algoritm\\Algorithm\\Algorithm\\BoxStacking.txt";
	FILE* f =  fopen(test_input_file.c_str(),"rb");
	int w,d,h;
	vector<Box*> Boxes;
	while(fscanf(f,"%d %d %d",&w,&d,&h)!=EOF){
		Boxes.push_back(new Box(w,d,h));
	}
	cout<<"input boxes:"<<Boxes.size()<<endl;

	vector<Box*> Rot;
	//rot the original boxes
	map<Box*,int> RotParent;
	for (int i = 0;i<Boxes.size();i++)
	{
		Rot.push_back(Boxes[i]);
		RotParent[Boxes[i]]=i;

		int tw,td,th;
		Box* box;

		tw = Boxes[i]->h;
		td = Boxes[i]->w;
		th = Boxes[i]->d;
		box = new Box(tw,td,th);
		Rot.push_back(box);
		RotParent[box]=i;

		tw = Boxes[i]->d;
		td = Boxes[i]->h;
		th = Boxes[i]->w;
		box = new Box(tw,td,th);
		Rot.push_back(box);
		RotParent[box]=i;
	}

	sort(Rot.begin(),Rot.end(),sort_desc());

	vector<int> BoxLIS(Rot.size(),0);

	for (int i =0;i<Rot.size();i++)
	{
		cout<<Rot[i]->w<<" "<<Rot[i]->d<<" "<<Rot[i]->h<<endl;
		BoxLIS[i] = Rot[i]->h;
	}

	vector<int> BookKeeping(Rot.size(),-1);
	set<int> Used;

	//cout<<BookKeeping.size()<<" "<<BoxLIS.size()<<endl;

	//construct optimal size from bottom to top
	for (int i = 1;i<Rot.size();i++){
		for (int j = 0; j<i; j++)
		{
			//if (Rot[j]->w > Rot[i]->w && Rot[j]->d > Rot[i]->d && BoxLIS[i]<BoxLIS[j]+Rot[i]->h )  //not constrained,can use one box multiple times
			//if (RotParent[Rot[i]]!=RotParent[Rot[j]] && Rot[j]->w >= Rot[i]->w && Rot[j]->d >= Rot[i]->d && BoxLIS[i]<BoxLIS[j]+Rot[i]->h)// can use one box only once
			
			//if (canUse(RotParent,Rot[i],Rot[j])  && canUse(RotParent,Used,Rot[i]) && canUse(RotParent,Used,Rot[j]) && Rot[j]->w >= Rot[i]->w && Rot[j]->d >= Rot[i]->d && BoxLIS[i]<BoxLIS[j]+Rot[i]->h)// can use one box only once
			
			if (canUse(RotParent,Rot[i],Rot[j]) && Rot[j]->w >= Rot[i]->w && Rot[j]->d >= Rot[i]->d && BoxLIS[i]<BoxLIS[j]+Rot[i]->h)// can use one box only once
			{
				BoxLIS[i] = BoxLIS[j]+Rot[i]->h ;
				cout<<"stacking box "<<i<<"("<<Rot[i]->w<<","<<Rot[i]->d<<","<<Rot[i]->h<<")"<<" on box "<<j<<"("<<Rot[j]->w<<","<<Rot[j]->d<<","<<Rot[j]->h<<")"<<" BoxLIS["<<i<<"]="<<BoxLIS[i]<<endl;
				BookKeeping[i] = j;
			}
		}
	}

	int max = -1;
	for (int i =0 ;i<BoxLIS.size();i++)
	{
		if (max<BoxLIS[i])
		{
			max = BoxLIS[i];
		}
	}

	cout<<"max height:"<<max<<endl;
	cout<<"BookKeeping:"<<endl;
	for (int i =0;i<BookKeeping.size();i++)
	{
		cout<<BookKeeping[i]<<"\t";
	}
	cout<<endl;
	int i;
	for (i=BookKeeping.size()-1;i>0;i--)
	{
		if (BookKeeping[i]>0)
		{
			break;
		}
	}
	cout<<i<<endl;
	cout<<Rot[i]->w<<" "<<Rot[i]->d<<" "<<Rot[i]->h<<endl;

	while(true){
		int j =BookKeeping[i];
		if (j>=0)
		{
			cout<<Rot[j]->w<<" "<<Rot[j]->d<<" "<<Rot[j]->h<<endl;
			i=j;
		}
		else{
			break;
		}
	}

	for (int i =0;i<Rot.size();i++)
	{
		delete Rot[i];
	}
	return 0;

    
}