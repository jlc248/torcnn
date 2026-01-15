#--------------------------------------------------------------------------
#
# PGroup.pm  - Class which owns multiple PMans
#
# REST 2002
#
# -- Parses 'set' and 'manager' commands from config file.
# -- Maintains an 'ENV' group
#--------------------------------------------------------------------------
package PGroup;

use Cwd;
use PMan;
use PControl;

$PGroup::LASTMANAGER    = "";      # Last seen manager  

#--------------------------------------------------------------------------
# new(pGroupNameString such as "KTLX", pGroupOwnerObject)
#
sub new
{
   my ($type, $name, $owner) = @_;
   my ($self) = { };
   $self->{pMans} = ();      # List of PMan this pGroup owns
   $self->{env} = {};        # Extra ENV setup for this pGroup
   $self->{renv} = {};       # used to restore old ENV
   $self->{name} = $name;    # PGroup name, such as 'KTLX'
   $self->{owner} = $owner;  # PControl object this belongs to

   # Configuration parsing flags
   $self->{useGroupEnv} = 1; # Put 'set' in group or manager env?
   $self->{curPMan} = undef; # Current manager for 'set'
   bless $self, $type;
}

# Return our name
sub getName { my $self = shift; return $self->{name}; }

#--------------------------------------------------------------------------
# handleConfigLine
#
# create a manager or process 'set' commands.  Uses the useGroupEnv
# flag to tell if current scope is group or manager.
#
# New 'group' line found by PControl.  Set our useGroupEnv flag, so that
# sets go into our group env and not the current manager
sub initGroupConfig{ my $self = shift; $self->{useGroupEnv} = 1; }

sub handleConfigLine {
   my $self = shift;
   my ($line, $want, $loadM) = @_;
   my $name = $self->{name};

   # Humm.  should macro work in the manager tag, or
   # only non-manager tags.. +++
   # $group in line replaced with our name.  This is for group sharing
   $line =~ s/\$GROUP/$name/gi;

   #
   # Handle "MANAGER" command
   #
   if ($line =~ /^[ \t]*manager[ \t]*([^ \n\t#]*)/i) {
     my ($manName, $modName, $aPMan);

     # Allow name override for any manager of form "Module-Name"
     $manName = $1;
     if ($manName =~/^(.*)-(.*)/)
     {
       $modName = $1;        # Module is left side
       $manName = $2;        # Name in list is right side
     }else{
       $modName = $manName;  # Name same as manager module
     }

     # Keep right side of last seen manager for macro
     $PGroup::LASTMANAGER    = $manName; 
          
     # Create a new PMan object, set curPMan to the name, 
     # Current manager is no longer valid, make a new one
     # Future 'set' commands belong to manager, even if we
     # don't create the object
     $self->{curPMan} = undef;
     $self->{useGroupEnv} = 0;
     if (($loadM) && ($want eq "ALL") || ($want eq $manName)){

         # Load dynamic module
         my $className = $self->loadPManager($modName);
         if ($className) {
          $aPMan = PMan::new ($className, $self, $manName);
          push  @{$self->{pMans}}, $aPMan;
          $self->{curPMan} = $aPMan;
          #print "***Created $manName $aPMan ($self->{curPMan})\n";
          return 1;
         }else{
           print
           "ERROR::Failure loading PMan $className, skipping...\n";
         }
     } # end want if 
   } # end manager

   #
   # Handle "SET" commands by setting our env, or current manager object
   #
   elsif ($line =~ /^[ \t]*set[ \t]*([^ \t]*)[ \t]*\"(.*)/i) { 
     my ($setName, $setLine);
     $setName = $1; $setLine = $2;

     # Macro replace manager name here
     $setLine =~ s/\$MANAGER/$PGroup::LASTMANAGER/gi;

     # Grab from first quote to last quote, There can be other quotes inside
     if ($setLine =~ /(.*)\"/){ $setLine = $1; }

     if ($self->{useGroupEnv} == 1){
       ${$self->{env}}{$setName} = $setLine;
     }else{
       if (defined($self->{curPMan})){ 
         $self->{curPMan}->initEnv($setName, $setLine); 
       } # else we ignore the line
     } 
   #
   # Unrecognized line
   #
   }else{
     print "Group ".$self->{name}.
           " did not understand configuration line, skipping\n";
     print "---->$line\n";
   }
   return 0;
}

# -------------------------------------------------------------------------
# command($what)
# Send command $what to all of our loaded algorithms
# -------------------------------------------------------------------------
sub command {
  my $self = shift;
  my ($what) = @_;

  if ($self->{pMans}){
    $self->setEnv();
      foreach(@{$self->{pMans}}){ $_->command($what); } 
    $self->restoreEnv();
  }
}

#--------------------------------------------------------------------------
# ==ENV routines---
#
# dumpEnv -- Print out our 'ENV' settings
# setEnv  -- Set the ENVIRONMENT variable for THIS pGroup only
# restoreEnv -- Set the ENV back to previous values
#
sub dumpEnv {
  my $self = shift;
  my %envHash;

  while(($key, $value) = each(%{$self->{env}})){
   print "$key set to $value\n";
  }
}
sub setEnv {
  my $self = shift;
  my ($env, $key, $value);

  # We need GLOBAL pGroup env first, unless of course WE are the global
  if ($self->{name} ne "GLOBAL"){ ($self->{owner})->setEnv(); }

  # Override with local, but use GLOBAL otherwise
  while(($key, $value) = each(%{$self->{env}})){ 
    ${$self->{renv}}{$key} = $ENV{$key};   # Save to restore
    $ENV{$key} = $value; 
  }
}
sub restoreEnv {
  my $self = shift;

  $renv = $self->{renv};
  while(($key, $value) = each(%$renv)){ 
    $ENV{$key} = $value;
  }
}

#--------------------------------------------------------------------------
# loadPManager(pMan)
#
# --Given a name such as "SCIT", 'loads' the PMan::SCIT.pm module
# after checking for existence.
# Returns class name, i.e. "PMan::SCIT" or "" if nothing found
#
sub loadPManager {
  my $self = shift;
  my ($pMan) = @_;
 
  ##### Make sure algorithm exists in the library folder...
  # by checking for RBM/algorithm.pm
  if (-e "${PControl::REALBIN}/PMan/$pMan.pm"){
    eval "require PMan::$pMan";
    return "PMan::$pMan";
  }else{
     return "";
  }
}
1;
