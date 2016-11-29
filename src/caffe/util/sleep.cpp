#include<unistd.h>
#include<iostream>
#include "caffe/util/sleep.hpp"

unsigned caffe_sleep(unsigned seconds){
	std::cout << "Waiting for " << seconds << " seconds." << std::endl;
	return sleep(seconds);
}

void caffe_usleep(int micro_seconds){
	std::cout << "Waiting for " << micro_seconds << " micro-seconds." << std::endl;
	usleep(micro_seconds);
}
