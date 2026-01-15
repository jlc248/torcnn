// (c) Copyright, 2001-2009. Not to be provided or used in any format without the express written permission of the University of Oklahoma. All rights reserved.
#include <iostream>
#include <unistd.h>

// A simple exe that sends a pulse every second, useful for
// testing scripts.  Note that std::endl, not '\n' has to be
// called for logs to flush 'per line'
using std::cout;
using std::endl;

int main(){
   while(1){
   std::cout << "Pulse" << std::endl;
   sleep(1);
   }
}
