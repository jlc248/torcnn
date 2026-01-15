#!/usr/bin/perl -w
#
# draw graphs of the output produced by w2procwatch.pl
#
#

Main: {
  # Read args
  if ($#ARGV <1){
    print "$0 inlogfile output_file \n";
    print " Draws a graph of the log file created by w2procwatch.pl and saves to the output file in png format. \n";
    print "    If the output_file is SHOW, the file is not saved to disk; it is merely shown using the ImageMagick display program.\n";
    exit(0);
  }

  my($infile) = $ARGV[0];
  my($outfile) = $ARGV[1];
  my($tmpplot) = "/tmp/procplot_plot_$$";

  # you can get this information by doing ' man 5 proc  and looking at statm'
  my(@titles) = split / /, "size resident";# share text stack library dirty";
  my($startcol) = 8; # no of fields in date plus the comma
  
  # you can get this information by doing ' man 5 proc  and looking at stat'
  my($firstline) = `head -1 $infile` or die "Unable to read from $infile.\n";
  chop $firstline;
  $firstline =~ s/ +/ /g;  # Make sure no double spaces before split
  my($progname) = ($firstline =~ /.*\((.*)\).*/);
  my(@pieces) = split / +/, $firstline;
  my($startime) = $pieces[3];
  my($lastline) = `tail -1 $infile` or die "Unable to read from $infile.\n";
  $lastline =~ s/ +/ /g;  # Make sure no double spaces before split
  @pieces = split / +/, $lastline;
  my($endtime) = $pieces[3];

  open TMP, ">$tmpplot" or die "Unable to open $tmpplot for writing.\n";
  print TMP "# gnuplot plotting file written by w2procplot.pl\n";
  print TMP "set terminal png color medium\n";
  print TMP "set output \"$outfile\" \n";
  print TMP "set grid\n";
  print TMP "set data style lines \n";
  print TMP "set xlabel \"Time ($startime to $endtime)\" \n";
  print TMP "set ylabel \"Memory\" \n";
  print TMP "set title \"$progname\" \n";
  
  my($plotcommand) = "plot ";
  for ($i=0; $i <= $#titles; $i++){
    my($col) = $startcol + $i;
    $plotcommand = "$plotcommand '$infile' using $col title \"$titles[$i]\",";
  }
  chop $plotcommand;
  print TMP $plotcommand;
  close TMP;

  system("gnuplot $tmpplot");

  unlink($tmpplot);
  if ( $outfile eq "SHOW" ){
     system("display $outfile");
     unlink($outfile);
  }
}
