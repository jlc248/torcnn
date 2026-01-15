#!/usr/bin/perl -w
#--------------------------------------------------------------------------
#
# REST Jan 2003
#
# dirwatcher script used by DIRWATCH module
#--------------------------------------------------------------------------

#
# Watch a directory for changes
#
use strict;
use PDirwatch;

Main: {
 my $stdWatch = new PDirwatch; 
 $stdWatch->mainWatchLoop();
}
