# -------------------------------------------------------------------------
#
# REST April 2002
#
# General module for creating and running one process
# This is designed to run algorithms that only require
# a single command line option.
#
# PROCESS_NAME -- Name of process for status, etc.
#                 Defaults to PROCESS_EXE if blank.
# PROCESS_EXE  -- Exe name
#
# PROCESS_PARAMS -- Params for process
#
# DIR -- Directory to run from
# -------------------------------------------------------------------------

package PMan::PROCESS;

use PMan;
@ISA = qw(PMan);

# General method for starting up an algorithm
sub start
{
  my $self = shift;
  my $exe = $self->envOr("PROCESS_EXE", "");
  my $name = $self->envOr("PROCESS_NAME", $exe);
  my $params = $self->envOr("PROCESS_PARAMS", "");
  my $rundir = $self->envOr("DIR", ""); # Humm..should higher object handle?

  # Some checking.
  if ($exe eq ""){
    print "Error: Define PROCESS_EXE in configuration file\n";
  }

  print "Starting general process '$exe'\n";
  if ($rundir eq ""){
    #$self->runForever("\U$name", "$exe $params");
    $self->runForever("A", "$exe $params");
  }else{
    #$self->runForever("\U$name", "$exe $params", $rundir);
    $self->runForever("A", "$exe $params", $rundir);
  }
}

# PManager is requesting our id list
# This is used for the default stop method, as well as lognames
# for rotate/clear commands
# We have one process id we have named by exe
sub setIDS
{
  my $self = shift;
  my $exe = $self->envOr("PROCESS_EXE", "");
  my $name = $self->envOr("PROCESS_NAME", $exe);
  $self->setID("A");
}
