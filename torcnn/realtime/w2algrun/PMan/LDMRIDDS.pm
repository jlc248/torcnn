#
#
# Alpha ---
#
#
package PMan::LDMRIDDS;
use PMan;
@ISA = qw(PMan);

#--------------------------------------------------------------------------
# LDMRIDDS
#
# Config variables used:
#   TOP_DIR <--- Global or local top directory
#   LDM_EXEC <--- Optional override of /wdss/algs/ldm_ridds/ldm_ridds
#

sub start
{
  my $self = shift;
  
  print "Starting LDMRIDDS\n";

  my $topDir = $self->envOr("TOP_DIR", ""); # +++ use gettop to guess it
  my $ldmExec = $self->envOr("LDM_RIDDS", "/wdss/algs/ldm_ridds/ldm_ridds");
  my $ldmData = $self->envOr("LDM_DATA_DIR", "/mnt/ldm/data");
  $ldmData .= "/";
  $ldmData .= $self->{name};

  # Run forever the following.  Give each a unique identifier 
  $self->runForever("M", "$topDir$ldmExec -current -data=. ", $ldmData);
}

sub stop
{
  my $self = shift;
  $self->stopForever("M");
}
