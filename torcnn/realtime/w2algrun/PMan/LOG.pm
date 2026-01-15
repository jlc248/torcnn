# -------------------------------------------------------------------------
#
# REST May 2002
#
# General module for a log watcher
#
# LOG_DIR      -- General log path
# LOG_FILE     -- Name of log file to watch
# LOG_NAME     -- Name for xterm window, or defaults to LOG_FILE
# -------------------------------------------------------------------------

package PMan::LOG;

use PMan;
@ISA = qw(PMan);

# General method for starting up an algorithm
sub start
{
  my $self = shift;

  # Read our variables
  my $logdir = $self->envOr("LOG_DIR", "");
  my $logfile = $self->envOr("LOG_FILE", "");
  my $logname = $self->envOr("LOG_NAME", $logfile);

  # Some checking of params
  if ($logfile eq ""){
     print "ERROR: Must define LOG_FILE \n";
     return;
  }

  # Same path checking/fixing as w2logger.pl
  if ($logdir ne ""){
    if (-d $logdir){ $logfile = "$logdir/$logfile";
    }else{ $logfile = "/var/tmp/$logfile"; }
  }
  `touch $logfile` unless (-e $logfile);

  # Actual exe...make a xterm window and create tail within
  $exe = 
"xterm -T \"Watching $logname ($logfile)\" -e tail --follow=name $logfile";

  print "Starting log $logname\n";
  $self->runForever("A", "$exe");
}

# PManager is requesting our id list
# This is used for the default stop method, as well as lognames
# for rotate/clear commands
# We have one process id we have named by exe
sub setIDS
{
  my $self = shift;
  my $logfile = $self->envOr("LOG_FILE", "");
  my $logname = $self->envOr("LOG_NAME", $logfile);
  $self->setID("A");
}
