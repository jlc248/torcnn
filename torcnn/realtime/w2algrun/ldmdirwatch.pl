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
use LdmWatcher;

Main: {
 my $stdWatch = LdmWatcher->new(); 
 $stdWatch->mainWatchLoop();
}
