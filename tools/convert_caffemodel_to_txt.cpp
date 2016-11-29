#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include <fstream>
#include <iostream>

using google::protobuf::Message;
using namespace std;
using namespace caffe;

void convert_caffemodel(string input_file);
void convert_caffemodel_bn(string input_file);
void convert_caffemodel_tn(string input_file);

int main(){
	string path = "/home/lifengfu/caffe-ternary-compress/examples/test/";
    string filename;
    cout << "Input file name: ";
    getline(cin, filename);
	convert_caffemodel(path + filename);

	// input_file = "/home/lifengfu/caffe-ternary-compress/examples/test/baseline_iter_100.caffemodel.bn";
	// convert_caffemodel_bn(input_file);

	// input_file = "/home/lifengfu/caffe-ternary-compress/examples/test/baseline_iter_100.caffemodel.tn";
	// convert_caffemodel_tn(input_file);

	return 0;
}

void convert_caffemodel(string input_file){
	NetParameter* proto = new NetParameter();
	ReadProtoFromBinaryFile(input_file, proto);
    string output_file = input_file + ".txt";
	// WriteProtoToTextFile(*proto, output_file);
	
	ofstream outfile(output_file.c_str(), ios::out);
	if(!outfile){
		cout << "error!" << endl;
	}else{
		outfile << proto->name() << endl;
		int n = proto->layer_size();
		n = (n<=5) ? n : 5; // save the first 5 layer
		//for(int i=0; i<proto->layer_size(); i++){
		for(int i=0; i<n; i++){
		  outfile << "\n------------------------\n";
		  outfile << " layer " << i << "\n";
		  outfile << " name: " << proto->layer(i).name() << "\n";	
		  outfile << " type: " << proto->layer(i).type() << "\n"; 
		  outfile << "--------------------------\n";
		  for(int j=0; j<proto->layer(i).blobs_size(); j++){
		    outfile << "\n------\n";
			outfile << "blob " << j << "\n";
		    for(int k=0; k<proto->layer(i).blobs(j).data_size(); k++){
			  outfile << proto->layer(i).blobs(j).data(k) << " ";
			  if((k+1)%9 == 0){
				outfile << "\n";
			  }
			}
		    outfile << "-------\n";
		  }
		}
		outfile.close();
	}
}

void convert_caffemodel_bn(string input_file){
	NetParameter* proto = new NetParameter();
	ReadProtoFromBinaryFile(input_file, proto);
    string output_file = input_file + ".txt";
	// WriteProtoToTextFile(*proto, output_file);
	
	ofstream outfile(output_file.c_str(), ios::out);
	if(!outfile){
		cout << "error!" << endl;
	}else{
		outfile << proto->name() << endl;
		for(int i=0; i<proto->layer_size(); i++){
		  outfile << "\n------------------------\n";
		  outfile << " layer " << i << "\n";
		  outfile << " name: " << proto->layer(i).name() << "\n";	
		  outfile << " type: " << proto->layer(i).type() << "\n"; 
		  outfile << "--------------------------\n";
		  for(int j=0; j<proto->layer(i).blobs_size(); j++){
		    outfile << "\n------\n";
			outfile << "blob " << j << "\n";
		    for(int k=0; k<proto->layer(i).blobs(j).binary_data_size(); k++){
			  outfile << proto->layer(i).blobs(j).binary_data(k) << " ";
			  if((k+1)%9 == 0){
				outfile << "\n";
			  }
			}
		    outfile << "-------\n";
		  }
		}
		outfile.close();
	}
}

void convert_caffemodel_tn(string input_file){
	NetParameter* proto = new NetParameter();
	ReadProtoFromBinaryFile(input_file, proto);
    string output_file = input_file + ".txt";
	// WriteProtoToTextFile(*proto, output_file);
	
	ofstream outfile(output_file.c_str(), ios::out);
	if(!outfile){
		cout << "error!" << endl;
	}else{
		outfile << proto->name() << endl;
		for(int i=0; i<proto->layer_size(); i++){
		  outfile << "\n------------------------\n";
		  outfile << " layer " << i << "\n";
		  outfile << " name: " << proto->layer(i).name() << "\n";	
		  outfile << " type: " << proto->layer(i).type() << "\n"; 
		  outfile << "--------------------------\n";
		  for(int j=0; j<proto->layer(i).blobs_size(); j++){
		    outfile << "\n------\n";
			outfile << "blob " << j << "\n";
		    for(int k=0; k<proto->layer(i).blobs(j).ternary_data_size(); k++){
			  outfile << proto->layer(i).blobs(j).ternary_data(k) << " ";
			  if((k+1)%9 == 0){
				outfile << "\n";
			  }
			}
/*
		    for(int k=0; k<proto->layer(i).blobs(j).data_size(); k++){
			  outfile << proto->layer(i).blobs(j).data(k) << " ";
			  if((k+1)%9 == 0){
				outfile << "\n";
			  }
			}
*/
		    outfile << "-------\n";
		  }
		}
		outfile.close();
	}
}
