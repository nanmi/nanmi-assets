
#include <cstdio>
#include <string>
#include "vibe.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	Mat frame, gray, mask, FGModel;
	VideoCapture capture;
    string input = string(argv[1]);
	capture.open(input);
	if (!capture.isOpened())
	{
		cout<<"No camera or video input!\n"<<endl;
		return -1;
	}

    ViBe vibe;
    bool count = true;
    unsigned int ii = 0;
    while (true)
    {
       capture >> frame;
       if (frame.empty())
           continue;

       cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
       if (count)
       {
           vibe.init(gray);
           vibe.ProcessFirstFrame(gray);
           cout << "Training ViBe Success." << endl;
           count = false;
       }
       else
       {
           vibe.Run(gray);

           FGModel = vibe.getFGModel();
        
        cv::imwrite("../outputs/segmented_"+ std::to_string(ii) + ".jpg", FGModel);
        ii++;
    }

	return 0;
}
