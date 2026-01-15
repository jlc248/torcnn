# -------------------------------------------------------------------------
#
# RSSD module.  Runs rssd (in foreground mode)
#
# REST July 2002
#
# Put this in your GLOBAL section as "manager RSSD"
#
# PROCESS_PARAMS -- Extra params for rssd

package PMan::RSSD;

use PMan;
@ISA = qw(PMan);

# General method for starting up an algorithm
sub start
{
  my $self = shift;
  my $params = $self->envOr("PROCESS_PARAMS", "");

  # Some checking.  First hunt for binary
  my $findit = `which rssd`;
  if (!($findit =~ /rssd/)){
    print "ERROR: Couldn't find rssd in your path\n";
    exit(1);
  }

  print "Starting rssd (in foreground mode)\n";
  print "--Your RMTPORT is set to $ENV{RMTPORT}\n";
  $self->runForever("A", "rssd -t $params");
}

#--------------------------------------------------------------------------
# stop
#  -- override method ..force kill ANY rssd process we find..
#  ++This is a hack, RSSD in -t mode seems to leave child processing
# running on quit.  
sub stop { 
  my $self = shift;
  my $name = $self->{cname};

  # Normal stop code...
  $self->setIDS;
  foreach(keys %{$self->{ids}}){ 
   print "$name --> RSSD Stopping Parent Process $_ \n";
   $self->stopForever("\U$_");
  }

  #
  # RSSD Orphan hunter
  #

  # Match up found processes in system that are running
  $col = $ENV{COLUMNS}; $ENV{COLUMNS} = 500;
  if (open (PS_PIPE, "ps -A -o comm -o pid |")){
    print "Hunting for RSSD orphans...\n";
    while ($line = <PS_PIPE>){

       # Broke down the match for my sanity
       if ($line =~ /rssd (.*)/){   #rssd 1231 (for example)
         $rightSide = $1;
         # Now if right of it contains ONLY numbers, it's the process
         if ($rightSide =~ /([0-9]+)/){ 
          $pid = $1; 
          print " --->found possible orphan rssd PID:$pid.  Sending KILL\n";
           `kill -KILL $pid`;  # Only kills if owned by user
         }
      }
    }
    close (PS_PIPE);
  }
  $ENV{COLUMNS} = $col;
}


# We have one process id we have named "RSSD" above
sub setIDS
{
  my $self = shift;
  $self->setID("A");
}
