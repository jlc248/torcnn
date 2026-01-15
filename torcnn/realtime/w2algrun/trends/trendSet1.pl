#!/usr/bin/perl

$data_dir = "/data/realtime/radar/multi";
$table = "LightningTable";
$subtype = "scale_0";
$history = 30;		# how many files back should we look?
$outputDir = "/tmp/NALtrendSet";


#@fields = ("MaxRef","Reflectivity_0C","Reflectivity_-20C", "MaxVIL");
@fields = ("MaxRef","MESH","MaxVIL","CGDensity","Reflectivity_-20C", "Size","VILMA", "LowLvlShear", "MidLvlShear","Reflectivity_0C","Reflectivity_-10C");

@plotme = ( 
	['Reflectivity_0C', 'Reflectivity_-10C', 'Reflectivity_-20C'],
	['LowLvlShear', 'MidLvlShear'],
	['VILMA'],
	['CGDensity'],
	['MESH','MaxVIL']
);

# no configuration changes needed below this line

$outputDir .= "/" . $table . "/" . $subtype;
if (! -e $outputDir) { `mkdir -p $outputDir`; }

$lat_field = -1;
$lon_field = -1;

$dir = $data_dir . "/" . $table . "/" . $subtype;
chdir $dir;
$n = 0;
$rowname = -1;

#print "should be in $dir, now in ",`pwd`;
$lastfile = "none";

while (1 > 0) {		# run forever
  chdir $dir;
  $newest_file = `ls -1r *xml* | head -1`;  chomp $newest_file;
  if ($lastfile eq $newest_file) {
    #print "still at file $newest_file...skipping...\n";
    sleep 30;
    next;
  }
  else { 
    $lastfile = $newest_file; 
    print "found new file $lastfile ... processing...\n";
    $n = 0;
  }
#print $newest_file,"\n";
  foreach $i (`ls -1r *xml* | head -$history`) {

    # read in each file
    chomp $i;
    $fullTime[$n]= substr($i,0,15);
    #20090318-174131.xml.gz
    $HMS[$n] =  substr($i,9,2) . ":" . substr($i,11,2) . ":" . substr($i,13,2);
    $datestr[$n] = substr($i,4,2) . "/" . substr($i,6,2) . "/" . substr($i,0,4);
    #$trendtime[$n] = substr($i,6,2) . "." . substr($i,9,6);
    @data = `zcat $i`;
    #print "read $i time = $fullTime[$n] $HMS[$n]\n";

    # parse out the different fields
    $datasection = 0;
    $col = -1;
    foreach $line (@data) {
#     <time units="secondsSinceEpoch" value="1236974466"/>

      if ($line =~ /time units=\"secondsSinceEpoch\" value=\"(.*?)\"/) {
        $trendtime[$n] = $1 * 1000;
      }
      if ($line =~ /<data>/) { $datasection = 1; }
      if ($line =~ /<\/data>/) { $datasection = 0;}
      if ($datasection>0 && $line =~ /<datacolumn name="(.*?)" units=\"(.*?)\"/) {
        $item = 0;
        $col++;
        $columnName[$col] = $1;
        $columnUnit[$col] = $2;
        if ($columnName[$col] eq "RowName") { 
          $rowname = $col;
        }
        for ($k=0;$k<=$#fields;$k++) {
          if ($columnName[$col] eq $fields[$k]) {
            $fieldnum[$k] = $col;
          }
          if ($columnName[$col] eq "Latitude") {
            $lat_field = $col;
          }
          if ($columnName[$col] eq "Longitude") {
            $lon_field = $col;
          }
        } 
        
      }
      if ($datasection && $col >= 0 && $line =~ /item value="(.*?)"/) {
        $items[$n][$col][$item] = $1;
        $item++;
      }
    }
    $num_items[$n] = $item; # store for later
    $n++;
    #print "times = $n  columns = " . ($col+1) . "   items = $item\n";
  }

  if ($rowname == -1) { 
    print "Aborting because we couldn't find RowName\n";
    next;
  }
  if ($lat_field == -1 || $lon_field == -1) {
    print "Aborting because we couldn't find the Lat/Lon\n";
    next;
  }

  for ($i=0;$i<$num_items[0];$i++) { # the first table is our starting point -- only include RowNames that are in the first table
    for ($k=0;$k<=$#fields;$k++) {
      #print "found $columnName[$fieldnum[$k]] in column $fieldnum[$k]\n";
      $dataset[$i][$k] = 
        "    var data$k = [";
    }
  }

  for ($i=0;$i<$num_items[0];$i++) { 
    #print "Working on cell # " . $items[0][$rowname][$i] . " at " . $HMS[0] . "\n";

    for ($j=0;$j<$n;$j++) { # time loop
      for ($k=0;$k<$num_items[$j];$k++) {  # items at past time loop
        if ( $items[0][$rowname][$i] ==  $items[$j][$rowname][$k]) {
          #print "found cell $items[$j][$rowname][$k] at time $HMS[$j]\n";
          for ($m=0;$m<=$#fields;$m++) {  #loop through each field that we want to extract
            $dataset[$i][$m] .= "[$trendtime[$j],$items[$j][$fieldnum[$m]][$k]],";
            #print "  $columnName[$fieldnum[$m]] = $items[$j][$fieldnum[$m]][$k]\n";
          }
        }
      }
    }

# current position
    $latLonString[$i] = "$items[0][$lon_field][$i],$items[0][$lat_field][$i],0";
  } 



  for ($i=0;$i<$num_items[0];$i++) { 
  #print "---------------------- $items[0][$rowname][$i]-----------------\n";
  #print $items[0][$rowname][$i] . "\n";
    #at the same time, generate the data field javascript for each variable:
    for ($k=0;$k<=$#fields;$k++) {
      #data:
      chop  $dataset[$i][$k];  #remove comma
      $dataset[$i][$k] .= "];\n";
      #print  $dataset[$i][$k];
    }
  }

  # write out the file

  $outputFile = $outputDir . "/" .  $fullTime[0] . ".kml";
  $outputFileKMZ =  $fullTime[0] . ".kmz";
  $outputFileTar =  $outputDir . "/" .  $fullTime[0] . ".tar";
  #print "writing out to $outputFile\n";
  if (! -e $outputDir) { `mkdir -p $outputDir`; }
  open FILE,">$outputFile";

  print FILE "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" .
           "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n" .
           "<Document>\n" .
           "  <name>$table ($subtype) at $datestr[0] $HMS[0]</name>\n" .
# style for polygons needs to be user-adjustable:
           "  <StyleMap id=\"msn_polygon\">\n" .
           "     <Pair>\n" .
           "             <key>normal</key>\n" .
           "             <styleUrl>#sn_polygon</styleUrl>\n" .
           "     </Pair>\n" .
           "     <Pair>\n" .
           "             <key>highlight</key>\n" .
           "             <styleUrl>#sh_polygon</styleUrl>\n" .
           "     </Pair>\n" .
           "  </StyleMap>\n" .
           "  <Style id=\"sh_polygon\">\n" .
           "         <IconStyle>\n" .
           "                 <color>ffffff55</color>\n" .
           "                 <scale>1.2</scale>\n" .
           "                 <Icon>\n" .
           "                         <href>http://maps.google.com/mapfiles/kml/shapes/polygon.png</href>\n" .
           "                 </Icon>\n" .
           "         </IconStyle>\n" .
           "         <ListStyle>\n" .
           "         </ListStyle>\n" .
           "  </Style>\n" .
           "  <Style id=\"sn_polygon\">\n" .
           "         <IconStyle>\n" .
           "                 <color>ffffff55</color>\n" .
           "                 <Icon>\n" .
           "                         <href>http://maps.google.com/mapfiles/kml/shapes/polygon.png</href>\n" .
           "                 </Icon>\n" .
           "         </IconStyle>\n" .
           "         <ListStyle>\n" .
           "         </ListStyle>\n" .
           "  </Style>\n" .
# end of header string
           "\n";


  for ($i=0;$i<$num_items[0];$i++) { 
    print FILE << "EOF1";
  <Placemark>
    <name>#$items[0][$rowname][$i] </name>
    <snippet maxLines=\"2\">$fullTime[0] $table ($subtype)</snippet>
    <styleUrl>#msn_polygon</styleUrl>
    <Point>
      <coordinates>$latLonString[$i]</coordinates>
    </Point>
    <description>
<![CDATA[
 <head>
    <title>WDSSII Trend</title>
    <link href="layout.css" rel="stylesheet" type="text/css"></link>
    <!--[if IE]><script language="javascript" type="text/javascript" src="../excanvas.pack.js"></script><![endif]-->
    <script language="javascript" type="text/javascript" src="http://wdssii.nssl.noaa.gov/trends/jquery.js"></script>
    <script language="javascript" type="text/javascript" src="http://wdssii.nssl.noaa.gov/trends/jquery.flot.js"></script>

 </head>
    <body>
    <h1>Trends for feature # $items[0][$rowname][$i] at  $datestr[0] $HMS[0]</h1>
EOF1
;
    for my $m ( 0 .. $#plotme) {
      print FILE
       "<div id=\"placeholder$m\" style=\"width:500px;height:200px;\"></div>\n";
      print FILE "<hr>\n";
    }

    # loop through each of the plots:
    for my $m ( 0 .. $#plotme) {
      print FILE "<script id=\"source$m\" language=\"javascript\" type=\"text/javascript\">\n";
      print FILE "\$(function () {\n";

      my $ref = $plotme[$m];
      my $nref = @$ref - 1;
      # each member of each plot:
      my $datablock = "        [ ";
      for my $n (0 .. $nref) {
        for ($k=0;$k<=$#fields;$k++) {
          if ($fields[$k] eq $plotme[$m][$n] ) {
#print "field $m $n = $plotme[$m][$n] = $k\n";
            print FILE $dataset[$i][$k];
            $datablock .= "{ data: data$k, label: \"$fields[$k]\"  },";
          }
        }
        #print $plotme[$m][$n],"\n";
      }
      chop $datablock;
      $datablock .= "],";

    print FILE << "EOF2";
            \$.plot(\$("#placeholder$m"),
		$datablock
            {    yaxis: { min: 0 },
                xaxis: { mode: "time" },
                points: { show: true },
                lines: {show: true },
		legend: { position: "sw" }
    });

});
</script>
EOF2
;
    }
    print FILE << "EOF3";
 </body>
]]>
    </description>
  </Placemark>
EOF3
;
  }
  print FILE "</Document>\n</kml>\n";
  close FILE;
exit;
  chdir $outputDir;
  `zip $outputFileKMZ $outputFile`;
  `/bin/cp  $outputFileKMZ index.kmz`;
  `tar cvf $outputFileTar $outputFileKMZ index.kmz`;
  `/home/ldm/bin/pqinsert -q /home/ldm/data/ldm.pq  $outputFileTar`;
  #`/bin/rm $outputFile  $outputFileTar $outputFileKMZ2`;

}		# end of "run forever" loop
