#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace cv::dnn;
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;
/* Find best class for the blob (i. e. class with maximal probability) */
void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;
	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	//cout << probMat.at<char>(0,0) << "  " << probMat.at<char>(1,0) << "  " << probMat.at<char>(2, 0) << "  " << probMat.at<char>(3, 0) << "  " << probMat.at<char>(4, 0) << "  " << probMat.at<char>(5, 0) << "  " << endl;
	*classId = classNumber.x;
}
std::vector<String> readClassNames(const char *filename = "synset_words.txt")
{
	std::vector<String> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}

int main(int argc, char **argv)
{

	string imagePath = "D:\\PeripheralBloodSmear\\CATDOGTutorial\\Image\\test\\";

	string modelTxtPath = "D:\\PeripheralBloodSmear\\CATDOGTutorial\\caffe_models\\caffe_model_1\\caffenet_deploy_1.prototxt";
	string modelBinPath = "D:\\PeripheralBloodSmear\\CATDOGTutorial\\caffe_models\\caffe_model_1\\caffe_model_1_iter_8000.caffemodel";
	const char * classesPath = "D:\\PeripheralBloodSmear\\CATDOGTutorial\\caffe_models\\caffe_model_1\\dataset.txt";

	Ptr<dnn::Importer> importer;
	try                                     //Try to import Caffe GoogleNet model
	{
		importer = dnn::createCaffeImporter(modelTxtPath, modelBinPath);
	}
	catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
	{
		std::cerr << err.msg << std::endl;
	}

	if (!importer)
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxtPath << std::endl;
		std::cerr << "caffemodel: " << modelBinPath << std::endl;
		std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		exit(-1);
	}
	dnn::Net net;// =cv::dnn::readNetFromCaffe(modelTxtPath,modelBinPath);
	if (importer)
	{
		importer->populateNet(net);
		importer.release();                     //We don't need importer anymore
	}


	vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
	String folder = imagePath; // again we are using the Opencv's embedded "String" class

	glob(folder, filenames); // new function that does the job ;-)

	for (size_t i = 0; i < filenames.size(); ++i)
	{
		Mat src = imread(filenames[i]);

		if (!src.data)
			cerr << "Problem loading image!!!" << endl;

		Mat inputBlb;
		Size size(227, 227);//the dst image size,e.g.100x100
		cv::resize(src, inputBlb, size);       //GoogLeNet accepts only 224x224 RGB-images

											   //GoogLeNet accepts only 224x224 RGB-images
		dnn::Blob inputBlob = dnn::Blob(inputBlb);   //Convert Mat to batch of images
		net.setBlob(".data", inputBlob);        //set the network input
		net.forward();                          //compute output
		dnn::Blob prob = net.getBlob("prob");   //gather output of "prob" layer
		int classId;
		double classProb;
		getMaxClass(prob, &classId, &classProb);//find the best class
		std::vector<String> classNames = readClassNames(classesPath);
		std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
		std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
		std::cout << "File Name: " << filenames[i] << std::endl;
	}

	while (true) {}
} //main