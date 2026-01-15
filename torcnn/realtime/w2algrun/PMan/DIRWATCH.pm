# -------------------------------------------------------------------------
#
# REST Jan 2003
#
# General module for wathing a directory for changes
#
# DIRWATCH_WATCH_DIR -- Path of directory to watch 
# ideas:
# DIRWATCH_RECURSIVE -- Recursively watch?
# --notify email
# -------------------------------------------------------------------------

package PMan::DIRWATCH;

use FindBin qw($Bin $Script $RealBin $RealScript);
use PMan;
@ISA = qw(PMan);

# General method for starting up an algorithm
sub start
{
  my $self = shift;
  my $path = $self->envOr("DIRWATCH_WATCH_DIR", "");
  my $watchscript = "$RealBin/dirwatch.pl";
  #my $watchscript = "$RealBin/watch";

  if ($path eq ""){
    print "Error: Define DIRWATCH_WATCH_DIR in configuration file\n";
    return;
  }
  my $params = "$path";

  print "Starting directory watcher on '$path'\n";
  $self->runForever("A", "$watchscript");
}

# PManager is requesting our id list
# This is used for the default stop method, as well as lognames
# for rotate/clear commands
# We have one process id we have named by exe
sub setIDS
{
  my $self = shift;
  $self->setID("A");
}
