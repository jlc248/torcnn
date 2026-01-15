#!/usr/bin/perl -w
#--------------------------------------------------------------------------
# Shared log functions between w2logger and PControl
#
# w2logger needs to rotate logs, PControl also rotates logs by command
# So we have shared routines for these actions
# and to prevent them from stepping on each other in action
#
# Remind me of how much I hate multithreading... :)
# And here I thought Operating Systems was a worthless course
#
# Commands:
# w2alg log rotate
# w2alg log clear
#--------------------------------------------------------------------------
package PLogUtil;
use strict;
use Fcntl ':flock'; # Import flock constants

# Defaults if all fails
use constant MAX_LOG_SIZE => 1000000;  # 1 meg or so (size in bytes)
use constant MAX_NUM_LOGS => 5;        # Max log number

#--------------------------------------------------------------------------
# rotateLogs(logFile, logNum)
# rotateFLockedLogs(logFile, logNum)
#      logFile -- Absolute path of log to rotate. Includes marker # 
#      logNum -- Total number of logs in set
#  Used by PControl to rotate a log that is probably being used
sub rotateLogs
{
  my ($logFile, $logNum) = @_;
  my ($firstLog);

  $firstLog = $logFile; $firstLog =~ s/#/1/g;

  open(OUTLOG, ">>".$firstLog);
  flock(OUTLOG, LOCK_EX);  # Wait for current w2logger 
                           # to stop writing to this
    rotateFLockedLogs($logFile, $logNum);
  truncate(OUTLOG, 0);
  flock(OUTLOG, LOCK_UN);
  close(OUTLOG);
}

#--------------------------------------------------------------------------
# clearLogs(logFile, logNum)
#
# -- wipe out logs.  There is no clearFLockedLogs since w2logger
# never clears a log while running
sub clearLogs
{
  my ($logFile, $logNum) = @_;
  my ($i, $curLog, $firstLog);

  # Humm.. maybe just rm *.log?  But could wipe someone elses stuff.
  # Lets play it safe...just the logs we currently know about.
  $firstLog = $logFile; $firstLog =~ s/#/1/g;

  # Truncate current log
  open(OUTLOG, ">>".$firstLog);
  print "Truncating $firstLog\n";
  flock(OUTLOG, LOCK_EX);  # Wait for current w2logger to stop writing to this
  truncate(OUTLOG, 0);
  flock(OUTLOG, LOCK_UN);
  close(OUTLOG);

  # Truncate all older logs to 0 size.  As long as log 1 is large
  # enough, only a small chance that rotation occured between
  # truncating current and here.
  for ($i=2;$i<=$logNum;$i++){
     $curLog = $logFile; $curLog =~ s/#/$i/g;
     if (-e $curLog){
       open(OUTLOG, ">>".$curLog);
       print "Truncating $curLog\n";
       truncate(OUTLOG, 0);
       close(OUTLOG);
     }else{
       print "Truncating $curLog (missing)\n";
     }
  }
}

#--------------------------------------------------------------------------
# rotateFLockedLogs(logFile, logNum)
#
#  Rotate logs we already have open and flocked
#  Used by w2logger to hold logs and prevent other loggers from 
#  rotating at same time.
sub rotateFLockedLogs
{
  my ($logFile, $logNum) = @_;
  my ($i, $j, $newerLog, $olderLog);

  # Shift over all logs to older ones if any
  for ($i=$logNum;$i>1;$i--){
     $j = $i-1;
     $newerLog = $logFile; $newerLog =~ s/#/$j/g;
     $olderLog = $logFile; $olderLog =~ s/#/$i/g;
     # We can't just move the first file because other w2loggers will 
     # follow the file (the handle would point to log 2 instead of 1)
     if ($j == 1){
       # use -p preserve to keep date stamp.  Useful
       if (-e $newerLog){ system "cp -p $newerLog $olderLog"; }
     }else{
       if (-e $newerLog){ system "mv $newerLog $olderLog"; }
     }
  }
}

#--------------------------------------------------------------------------
# getLogInfo(logfile, logtype, \%logPath, \%logSize, \$logNum)
# getLogDir(logType, \$logDir)
#   -- Get log file/dir referred to by current ENV 
# Env is set for current manager or process by the time this is called
#
#  If logtype is 'w2' uses the w2run log env's instead
#
# Ex: getLogInfo("w2log#.log", "w2", \$thePath, \$theSize, \$theNum);
#
sub getLogDir
{
  my ($logType, $refLogDir) = @_;
  my $logDir;

  # Two log types, global w2run and individual
  if ($logType eq "w2"){ $logDir  = $ENV{W2RUN_LOG_DIR};
  }else{ $logDir  = $ENV{LOG_DIR}; }

  # Shouldn't write to /, even if root
  if ($logDir eq ""){ $logDir = "/var/tmp"; }

  # Use var tmp if directory doesn't exist
  if (!(-d $logDir)){
     $logDir = "/var/tmp";
  }

  $$refLogDir = $logDir;
}

sub getLogInfo
{
  my ($logFile, $logType, $refLogPath, $refLogNum, $refLogSize) = @_;
  my ($logNum, $logSize, $logDir);

  # Ok..two modes..one uses the w2 envs, the other the regular log
  # environs.  This allows different log settings for the general w2run log
  if ($logType eq "w2"){
    # Use the w2 env variables...
    $logNum  = $ENV{W2RUN_NUM_LOGS};
    $logSize = $ENV{W2RUN_LOG_SIZE};
  }else{  # or regular
    $logNum  = $ENV{NUM_LOGS};
    $logSize = $ENV{LOG_SIZE};
  }

  # Force to defaults if not numbers
  $logNum = MAX_NUM_LOGS unless ($logNum =~ /^\d+$/);
  $logSize = MAX_LOG_SIZE unless ($logSize =~ /^\d+$/);

  # Ok, add the required '#' for log numbers unless we have only
  # one log and it's missing
  if ($logNum > 1){ $logFile .= "#" if (!($logFile =~ /#/)); }
  &getLogDir($logType, \$logDir);
  $logFile = "$logDir/$logFile";

  # return values
  $$refLogPath = $logFile;
  $$refLogNum = $logNum;
  $$refLogSize = $logSize;
}
1;
