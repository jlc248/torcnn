package PMan::TOP;

#
# TOP
#
#  -- Runs gtop if it exists
#  (assumes we have the which command)
# +++ create regular top in xterm if requested...
#
use PMan;
@ISA = qw(PMan);

# General method for starting up an algorithm
sub start
{
  my $self = shift;
  my $findit = `which gtop`;
  if ($findit =~ /gtop/){
     $self->runForever("TOP", "gtop");
  }else{
     print
"ERROR: Manager TOP: Couldn't find gtop or top in your path.\n";
  }
}

#
# Let manager know what ids we are running
#
sub setIDS
{
  my $self = shift;
  $self->setID("TOP");
}
