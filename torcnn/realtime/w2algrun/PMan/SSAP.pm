#-------------------------------------------------------------------------
#
#  ALPHA....
#

package PMan::SSAP;
use PMan;
@ISA = qw(PMan);

#-------------------------------------------------------------------------
# General method for starting up an algorithm
sub start
{
  my $self = shift;
  my ($radar) = $self->getRadarName();

  # SSAP runs once for each radar, not globally like rssd for instance
  if ($radar eq "GLOBAL"){
    print("ERROR:SSAP must be run for a particular radar\n";
    return 0;
  }

  # Get variables we need
  my ($ssapmode) = "R";
  $ssapmode = "E" if ($ENV{DEALIAS_DATA});
  my ($nse) = "F"; $nse = "T" if ($ENV{NSE_PRESENT});
  my ($ltg) = "F"; $ltg = "T" if ($ENV{LTG_PRESENT});
  my ($rapid_update) = "F";
  $rapid_update = "T" if ($ENV{RUN_RAPID_UPDATE});
  my ($topDir) = $ENV{TOPDIR};
  my ($dataDir) = $ENV{DATADIR};

  # Handle this radar only...
  $ENV{RADARNAME} = $radar;
  # Run the ssapedit script to edit the ssaparm file
  my($ssapedit) = "cd $topdir/wdss/tools/ssapedit; ./ssapedit $radar";
  my(@commands) = (
   "cp $topDir/wdss/algs/ssap/*.dat $dataDir/$radar/",
   "cp $topDir/wdss/algs/ssap/WARN/warn_$radar.dat $dataDir/$radar/",
   "cp $topDir/wdss/algs/ssap/ssaparm.dat_rt $dataDir/$radar/ssaparm.dat",
   "cp $topDir/wdss/algs/ssap/NXglobal.* $dataDir/$radar/.",
   "cp $topDir/wdss/tools/ssapedit/ssapedit_info $dataDir/$radar/",
     "$ssapedit RADAR_NAME $radar",
     "$ssapedit DIR_NAME $dataDir/$radar/", #need slash here
     "$ssapedit CIRC_BUFFER_TYPE $ssapmode",
     "$ssapedit RUN_NSE $nse",
     "$ssapedit RUN_LTG $ltg",
     "$ssapedit RAPID_UPDATE $rapid_update",
     "$ssapedit WRITE_RAPID_TILTDATA $rapid_update"
  );
    
  foreach (@commands){ ($self->{owner})->sysExec($_); }
    
  # Start ssap;  ssap in real-time expects the first input to be 4
  my($execline) = "$topdir/w2/newscripts/w2passargs 4 - $topdir/wdss/algs/ssap/ssa88d";
  print"$execline\n";
  $self->runForever("1", $execline, "$DATADIR/$radar");
  print"\n";
  return 1; # assume we worked
}
