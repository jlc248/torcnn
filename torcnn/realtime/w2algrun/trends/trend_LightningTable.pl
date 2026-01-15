#!/usr/bin/perl

$data_dir = "/data/realtime/radar/multi";
$table = "LightningTable";
$subtype = "scale_0";
$history = 17;		# how many files back should we look?
$outputDir = "/tmp/NALtrend";


#@fields = ("MaxRef","Reflectivity_0C","Reflectivity_-20C", "MaxVIL");
@fields = ("MaxRef","MESH","MaxVIL","CGDensity","Reflectivity_-20C", "Size","VILMA-2min", "FlashRate-2min","LowLvlShear", "MidLvlShear");
@yaxis2 = ("Size","CGDensity","VILMA-2min","FlashRate-2min","MESH", "LowLvlShear", "MidLvlShear");

# no configuration changes needed below this line

$outputDir .= "/" . $table . "/" . $subtype;
if (! -e $outputDir) { `mkdir -p $outputDir`; }

$lat_field = -1;
$lon_field = -1;

$dir = $data_dir . "/" . $table . "/" . $subtype;
chdir $dir;
$n = 0;
$rowname = -1;

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
    #print "found new file $lastfile ... processing...\n";
    $n = 0;
  }
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
        "\"" . $columnName[$fieldnum[$k]] . "\": {\n" .
        "    label: \"" . $columnName[$fieldnum[$k]] . " (" . 
        $columnUnit[$fieldnum[$k]] . ")" . "\",\n" .
        "    data: [";
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
    #generate HTML columns for above the graph.  These will show which 
    #variables are located in which column:
    $tableCol1[$i] = "<td>";
    $tableCol2[$i] = "<td>";
    #at the same time, generate the data field javascript for each variable:
    for ($k=0;$k<=$#fields;$k++) {
      #columns above graph:
      $tablecolNum = 1;
      for ($m=0;$m<=$#yaxis2;++$m) {
        if ($columnName[$fieldnum[$k]] eq $yaxis2[$m]) { $tablecolNum = 2; }
      }
      if ($items[0][$fieldnum[$k]][$i] <= -99000) { $tableVal = "Missing"; }
      else { $tableVal = $items[0][$fieldnum[$k]][$i]; }
      if ($tablecolNum == 2) {
          $tableCol2[$i] .= $columnName[$fieldnum[$k]] . ": <b>" .
           $tableVal . "</b> \t" . 
           $columnUnit[$fieldnum[$k]] . "<br>";
      } else {
          $tableCol1[$i] .= $columnName[$fieldnum[$k]] . ": <b>" .
           $tableVal . "</b> \t" . 
	   $columnUnit[$fieldnum[$k]] . "<br>";
      }
      #data:
      chop  $dataset[$i][$k];
      $dataset[$i][$k] .= "]";
      for ($m=0;$m<=$#yaxis2;++$m) {
        if ($columnName[$fieldnum[$k]] eq $yaxis2[$m]) { $dataset[$i][$k] .= ", yaxis: 2"; }
      }
      $dataset[$i][$k] .= "\n}\n";
      #print  $dataset[$i][$k];
    }
    $tableCol1[$i] .= "</td>\n";
    $tableCol2[$i] .= "</td>\n";
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
    <table border="0" width="100%" cellpadding="10">
      <tr>
        $tableCol1[$i]
        $tableCol2[$i]
      </tr>
    </table>
    <div id="placeholder" style="width:600px;height:300px;"></div>

    <p id="choices">Show:</p>

<script id="source" language="javascript" type="text/javascript">
\$(function () {
    var datasets = {
EOF1
;

    for ($k=0;$k<=$#fields;$k++) {
      print FILE $dataset[$i][$k];
      if ($k <$#fields) { print FILE ","; }
    }
    print FILE << "EOF2";
    };

    // hard-code color indices to prevent them from shifting as
    // countries are turned on/off
    var i = 0;
    \$.each(datasets, function(key, val) {
        val.color = i;
        ++i;
    });
    
    // insert checkboxes 
    var choiceContainer = \$("#choices");
    i = 0;
    \$.each(datasets, function(key, val) {
        if (i == 0) {
          choiceContainer.append('<br/><input type="checkbox" name="' + key +
                               '" checked="checked" >' + val.label + '</input>');
        } else {
          choiceContainer.append('<br/><input type="checkbox" name="' + key +
                               '" >' + val.label + '</input>');

        }
        ++i;
    });
    choiceContainer.find("input").click(plotAccordingToChoices);

    
    function plotAccordingToChoices() {
        var data = [];

        choiceContainer.find("input:checked").each(function () {
            var key = \$(this).attr("name");
            if (key && datasets[key])
                data.push(datasets[key]);
        });

        if (data.length > 0)
            \$.plot(\$("#placeholder"), data, {
                yaxis: { min: 0 },
                y2axis: { min: 0 },
                xaxis: { mode: "time" },
                points: { show: true },
                lines: {show: true },
		legend: { position: "sw" }
            });
    }

    plotAccordingToChoices();
});
</script>

 </body>
]]>
    </description>
  </Placemark>

EOF2
;
  }
  print FILE "</Document>\n</kml>\n";
  close FILE;
#exit;
  chdir $outputDir;
  `zip $outputFileKMZ $outputFile`;
  `/bin/cp  $outputFileKMZ index.kmz`;
  `tar cvf $outputFileTar $outputFileKMZ index.kmz`;
  `/home/ldm/bin/pqinsert -q /home/ldm/data/ldm.pq  $outputFileTar`;
  #`/bin/rm $outputFile  $outputFileTar $outputFileKMZ2`;

}		# end of "run forever" loop
