#include "caffe/binary.hpp"
#include <iostream>
#include <string>

/*
bool choose(std::string option_name){
	std::cout << option_name << "? default is no ('n'/'0'/'return'): ";
	char str[10];
	str[0] = '0';
	std::cin.getline(str, 10);
	// std::cout << std::endl;
	char c = str[0];

	return !(c == '0' || c == 'n' || c == ' ' || c == '\0');
}
 extern bool BINARY = choose("Binary");
 extern bool TERNARY = choose("Ternary");
 extern bool DEBUG = choose("Debug");
 extern bool QUANTIZE = choose("Quantize");
 extern bool SCALE_WEIGHTS = choose("Scale Weights");
*/

 extern bool DEBUG = false;
 extern bool QUANTIZE = false;
 extern bool SCALE_WEIGHTS = false;
 extern bool BINARY = false;
 extern bool TERNARY = false;
 extern int TERNARY_DELTA = 7;
