#!/usr/bin/perl -w

# -------------------------------------------------------------------------
#
# REST 2001
#
# --Monitor a process and restart if needed.  Basically a sh to perl
#   translation of Lak's script with some extra paranoia process checking
#   and advanced logging features
#
#  $ENV{W2RUN_NUM_LOGS}  -- Number of log files for THIS script
#  $ENV{W2RUN_LOG_SIZE}  -- Size of log files for THIS script
#
#  $ENV{W2RUN_WHAT}; -- ARGV of what we are executing.
#  $ENV{W2RUN_LOGGER}; -- path to w2logger script exe
# -------------------------------------------------------------------------

package w2run;
use strict;

use FindBin qw($Bin $Script $RealBin $RealScript);
use lib "$RealBin";
use PDateStamp;
use PControl;
use Fcntl ':flock'; # Import flock constants

# Signals we try to catch
$SIG{INT} = \&reaper;   # control-c
$SIG{HUP} = \&reaper;   # Modem hang-up.  LOL :)
$SIG{QUIT} = \&reaper;  # User wants to core Control-\
$SIG{TERM} = \&reaper;  # Normal software termination
$SIG{TSTP} = \&reaper;  # User wants to stop Control-Z
$SIG{ABRT} = \&reaper;  # Abort was called

# -------------------------------------------------------------------------
# reaper
#
# Extra effort to try to prevent zombies and algs running
# without a monitor program
#
sub reaper {

  # +++ Need to catch broken pipe SIG
  # and cleanup logs accordingly...

  # Might need to be redefined, according to docs online
  $SIG{INT} = \&reaper;   # control-c
  $SIG{HUP} = \&reaper;   # Modem hang-up.  LOL :)
  $SIG{QUIT} = \&reaper;  # User wants to core Control-\
  $SIG{TERM} = \&reaper;  # Normal software termination
  $SIG{TSTP} = \&reaper;  # User wants to stop Control-Z
  $SIG{ABRT} = \&reaper;  # Abort was called
  
  if (defined($w2run::child_pid)){
   print PDateStamp::dateStamp();
   print "Stop($w2run::identifier) PID: $w2run::child_pid\n";
   exec "kill -KILL $w2run::child_pid";
  }
}

Main: {

  if (!defined($w2run::child_pid = open(CHILD, "-|"))) {
    die "cannot fork: $!";
  } elsif ($w2run::child_pid) {

# PARENT ------------------------------------------------------------------
# --Responsible for logging child's output.  Output stops when child dies
#
     my ($logfile, $logger, $numSysLog, $sleeper);

     # Logs for child
     $w2run::identifier=$ARGV[0]; shift;
     $logfile="$w2run::identifier#.log";
     if($ENV{W2RUN_LOGGER}){ $logger = $ENV{W2RUN_LOGGER};
     }else { $logger = "./w2logger"; }

     # We don't use ">>" in the exec in the child, because that would spawn 
     # yet another child under that process.  We let w2logger do the log 
     # work.
     print PDateStamp::dateStamp();
     print "Start($w2run::identifier) PID: $w2run::child_pid\n";

     # Write id to file the PMan will wait for
     my $lastid = $PControl::LASTRUN;
     open (LASTID, ">$lastid");
     flock(LASTID, LOCK_EX);
     print LASTID $w2run::child_pid;
     flock(LASTID, LOCK_UN);
     close LASTID;

     #-------------------------------------------------------------------------
     # Snag our child's output into its log file for it
     #
     open (FILE, "| $logger $logfile");
     select ((select(FILE), $|=1)[0]); # autoflush
       while (<CHILD>){ print FILE PDateStamp::dateStamp; print FILE $_; }
       close (CHILD);
     close (FILE);
     waitpid ($w2run::child_pid, 0);

     print PDateStamp::dateStamp();
     print "DIED($w2run::identifier)";

     # Wait so much time before trying to restart
     $sleeper = $ENV{RESTART_SLEEP} if ($ENV{RESTART_SLEEP});
     $sleeper = 2 if ($sleeper < 0);
     #sleep($sleeper);
     select undef, undef, undef, $sleeper;
    
     # Let another parent start before we go.
     exec "$0 $w2run::identifier";
  } else {

# CHILD -------------------------------------------------------------------
# --Responsible for starting up the process we want to run forever
# --Redirects stderr to stdout so parent can catch it
     $| = 1;  # autoflush stdout

     my $A = $ENV{W2RUN_WHAT};  # Keep ps as clean as we can

     # These prints are captured by parent in <CHILD> loop
     my $machine = `uname -a`;
     print "****$machine****EXELINE: $A\n";

     open (STDERR, ">&STDOUT");
     exec $A;
  } 
}

