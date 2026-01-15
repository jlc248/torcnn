#!/usr/bin/perl -w

# -------------------------------------------------------------------------
#
# REST Sept 2003
#
# --Monitor the proc of a process
#
# Output format: date, statm, stat
# "mam proc" will describe fields of stat and statm
#
# -------------------------------------------------------------------------

package w2run;
use strict;

use FindBin qw($Bin $Script $RealBin $RealScript);
use lib "$RealBin";
use PDateStamp;

Main: {
  my ($sleeper, $pid, $stat, $statm, $steps, $i);

  # Read args
  if ($#ARGV <1){
    print "$0 PID DELAYSECONDS\n";
    exit(0);
  }
  $pid = $ARGV[0];
  $sleeper = $ARGV[1];
  $sleeper = 2 if ($sleeper < 0);

  # Need a 'short' sleep or will wait too long after process
  # dies.  So sleep in steps of 2 seconds...
  $steps = $sleeper/2;
  
  while(1){

    # Read statm (shorter one)
    open(PROCFILE, "</proc/$pid/statm") || exit(0);
    $statm = <PROCFILE>; chomp $statm; close PROCFILE;
    $statm =~ s/,//g;

    # Read stat 
    open(PROCFILE, "</proc/$pid/stat") || exit(0);
    $stat = <PROCFILE>; chomp $stat; close PROCFILE;
    $stat =~ s/,//g;

    print PDateStamp::dateStamp();
    print " , $statm , $stat\n";

    # Sleep enough steps to add up to total time wanted
    for ($i=0; $i< $steps; $i++){
      select undef, undef, undef, 2;
    }
  }
}
