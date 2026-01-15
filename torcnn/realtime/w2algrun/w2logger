#!/usr/bin/perl -w
#--------------------------------------------------------------------------
# Circular Logger
#
# REST Dec 2001
#
# Back Rotates several log files:
# There must be at least 2 log files
#
#   logfile1.log --Newest
#...logfileN.log --Oldest of < MAX_LOG_SIZE
#
# Special cases: Environment variables
# 1. MAX_LOG_SIZE = 0 means infinite log..no clipping
#    MAX_NUM_LOGS meaningless here, will be 1
#
# w2logger logfile#name numLogs sizeLogs 
#
# PARAMS: "logfile#name" is passed in.  '#' is replaced by log number.
#         if logFile is passed in without '#', it is tacked on the end
#--------------------------------------------------------------------------

package w2logger;
use strict;
use FindBin qw($Bin $Script $RealBin $RealScript);
use lib "$RealBin";
use PLogUtil;
use Fcntl ':flock'; # Import flock constants

Main: {
  my ($curSize, $logFile, $firstLog, $logSize, $logNum, $logType );

  # Read args
  if ($#ARGV <0){
    print "ERROR: Need at least logfile name as input\n";
    exit(0);
  }
  $logFile = $ARGV[0];
  $logType = "";  $logType = $ARGV[1] if $ARGV[1];

  # Get log information
  PLogUtil::getLogInfo($ARGV[0], $logType, \$logFile, \$logNum, \$logSize);
  $firstLog = $logFile; $firstLog =~ s/#/1/g;

  open(OUTLOG, ">>".$firstLog);
  select ((select(OUTLOG), $|=1)[0]); # autoflush
  while(<STDIN>){ 

    # lock for writing against other w2loggers
    flock(OUTLOG, LOCK_EX);

    # If we want to limit log size...
    if ($logSize){

      # If the current log becomes too big...
      $curSize = (-s $firstLog);
      if (length($_)+$curSize > $logSize){
        PLogUtil::rotateFLockedLogs($logFile, $logNum);
        truncate(OUTLOG, 0);
      }
    }

    # Close and open for proper locking/flushing
    print OUTLOG $_;
    flock(OUTLOG, LOCK_UN);
    close(OUTLOG);

    # Reopen the file.  This allows other children to 
    # write to this file as well
    open(OUTLOG, ">>".$firstLog);
    select ((select(OUTLOG), $|=1)[0]); # autoflush

    # Code to flush on time.. saving for later in case autoflush
    # is too much for system... +++ Might try this later
    # if (time > $last_flush_time + 10){  # Every 10 seconds
    #   my $ofh = select OUTLOG;
    #   $| = 1;            # Make OUTFILE hot
    #   print OUTLOG "";   # Print nothing
    #   $| = 0;            # No longer hot
    #   select $ofh;
    #   $last_flush_time = time;
    # }
  }
  close(OUTLOG);
}
