#.============================================================================
#. Copyright 2000 ANSYS, Inc. Proprietary Data.
#. Unauthorized use, distribution, or duplication is prohibited.
#. All Rights Reserved.
#.============================================================================
#. $Revision$
#. $Date$
#.============================================================================
#.File Name:     report.tcl
#.Author:        Eric P. Clevenger
#.Description:   Creates those procs which allow creation of javascript
#.               for inclusion in an ANSYS report.
#.============================================================================

# ===========================================================================
# = NAMESPACE
# ============================================================================

package require APDL
package require msgcat
# ----------------------------------------------------------------------------
# GLOBAL NAMESPACE:  ansys::report
#
# Purpose:        This namespace creates the report context
#
# ----------------------------------------------------------------------------
namespace eval ansys::report {

   variable ReportArray
   variable AnimDone
   variable dArray
   variable tags

   variable CustomTableData
   set CustomTableData(0,0) ""

   set ReportArray(reportDir) {}

   set ReportArray(curImageNumber) 0
   set ReportArray(curAnimNumber) 0
   set ReportArray(curTableNumber) 0
   set ReportArray(curListingNumber) 0

   # Key that the GUI is up
   set ReportArray(menu) 0
   # Key that we are doing a image overlay sequence
   set ReportArray(imageSeq) 0;
   set ReportArray(reverseVideo) 1

   set ReportArray(sizeDownImg) 0
   set ReportArray(sizeDownAnim) 0
   set ReportArray(animActive) 0
   set ReportArray(grphBGimg) -999
   set ReportArray(batchCapture) 0

   set tags(start) "<ansys>"
   set tags(stop) "</ansys>"
   # Netscape 4 composer tag
   set tags(start4) "&lt;ansys>"
   set tags(stop4) "&lt;/ansys>"

   # These are the only procs that should be used externally by a customer
   namespace export setdirectory
   namespace export imagecapture
   namespace export tablecapture
   namespace export animcapture
   namespace export outputcapture
   namespace export imageoverlay
   namespace export interpdynamicdata
   namespace export finished

   # ==========================================================================
   # = PROCEDURES
   # ==========================================================================
   
   # --------------------------------------------------------------------------
   # GLOBAL PROCEDURE:  setdirectory directory ?replace? ?imageSize? ?repFormat?
   #
   # Purpose:        This procedure is called to begin the process of 
   #                 report generation.
   #
   # Arguments:      
   #                 directory
   #                    The directory to capture report information to.
   #                 ?replace?
   #                    Option to replace $directory. This will remove
   #                    $directory.
   #                 ?imageSize?
   #                    The percentage of the program specified window size
   #                    that the image should be.  The default is 100 percent.
   #                    This must be a value between 25 and 100.
   #                 ?repFormat?
   #                    Option to specify the format of the information in 
   #                    the report.  This may be one of javascript, html or
   #                    ds (I.e. DesignSpace report) the default is javascript.
   #
   # Return Value:   N/A
   #
   # Comments:       The other report generation procs will fail if this
   #                 one is not called first.
   #
   # See Also:       N/A
   #
   # --------------------------------------------------------------------------
   proc setdirectory { directory {replace 0} {imageSize 100} \
         {repFormat javascript} {sizeGraphics 1} } {
      variable ReportArray

      global tcl_platform
      global ansysRevDot

      # Give the option of making the image size smaller
      set ReportArray(imageSize) [expr $imageSize > 100 ? 100 : $imageSize]
      set ReportArray(imageSize) [expr $imageSize < 25 ? 25 : $imageSize]
      set ReportArray(imageSize) [expr $imageSize / 100.0]

      set ReportArray(batchSize) [expr int($ReportArray(imageSize) * 428)]
      set ReportArray(img,width) [expr int($ReportArray(imageSize) * 584)]
      set ReportArray(img,height) [expr int($ReportArray(imageSize) * 438)]

      set directory [::apdl::substitute $directory]
      set ReportArray(reportDir) $directory

      switch -- [string tolower $repFormat] {
         ds -
         html -
         javascript {
            set ReportArray(format) [string tolower $repFormat]
         }
         default {
            set ReportArray(format) [string tolower $repFormat]
            return -code error \
               -errorinfo \
                     "The report format must be set to \"javascript or html\"" \ 
                     "The report format must be set to \"javascript or html\""
         }
      }

      if {![file exists [file join $ReportArray(reportDir)]]} {
         set ReportArray(curImageNumber) 0
         set ReportArray(curAnimNumber) 0
         set ReportArray(curTableNumber) 0
         set ReportArray(curListingNumber) 0
         file mkdir $ReportArray(reportDir)
      } else {
         if {$replace} {
            set time [clock seconds]
            catch {file delete -force \
                  [file join $ReportArray(reportDir) ansysTables.js]}
            catch {file delete -force \
                  [file join $ReportArray(reportDir) ansysImages.js]}
            catch {file delete -force \
                  [file join $ReportArray(reportDir) ansysListings.js]}
            catch {file delete -force \
                  [file join $ReportArray(reportDir) ansysAnimations.js]}
            catch {file delete -force \
                  [file join $ReportArray(reportDir) images]}
            catch {file delete -force \
                  [file join $ReportArray(reportDir) images]}
            catch {file delete -force \
                  [file join $ReportArray(reportDir) anim_images]}
            foreach animDir [glob -nocomplain \
                  [file join $ReportArray(reportDir) animseq_*]] {
               catch {file delete -force $animDir}
            }
            set ReportArray(curImageNumber) 0
            set ReportArray(curAnimNumber) 0
            set ReportArray(curTableNumber) 0
            set ReportArray(curListingNumber) 0
 
         } else {
            append2Report
         }
      }

      if { [string match . ReportArray(reportDir)] } {
         set directory [pwd]
      }

      set ReportArray(reportName) \
            "[lindex [file split $directory] end].html"

      # Determine information for a batch run
      if {[ans_getvalue active,,int] == 0} {
         set ReportArray(batch) 1
         if {[ans_getvalue common,,d2com,,int,7] <= 0} {
            # No /SHOW command had been issued
            catch {ans_sendcommand )/SHOW,report,grph,,8} err
#            catch {ans_sendcommand )/SHOW,OFF} err
         }
      }

      # Get the options from the "registry" and set the options

      if {[catch {ansys::registry get \
         "HKEY_CURRENT_USER\\Software\\ANSYS, Inc.\\ANSYS\\ANSYS $ansysRevDot\\GUI\\Reports" \
         reverseVideo} reverseVideo]} {
         # If there is not a registry entry the default will be used.
         set reverseVideo 1
      }

      # Size Graphics Window
      ::apdl::noprint 1
      # The ans_getvalue sets levlui to 0 to prevent cursor flashing
      # so the active,,menu returns 0 if ans_getvalue is used.
      catch {ans_sendcommand )*get,_z1,active,,menu} err
      if {[ans_getvalue parm,_z1,value] \
            || [ans_getvalue common,,d2com,,int,61] } {
         set ReportArray(menu) 1
      }
      if {$sizeGraphics} {
         sizeGraphics down
         setOptions reverseVideo $reverseVideo
      }
      catch {ans_sendcommand )*DEL,_z1} err
      ::apdl::noprint 0

      ::apdl::noprint 1
      # Turn of min max symbols
      catch {ans_sendcommand /PLOPTS,MINM,OFF} err
      ::apdl::noprint 0

      return
   }

   proc setOptions { option value } {
      variable ReportArray
      switch -- $option {
         reverseVideo {
            # Reverse black and white and greys.
            set ReportArray(reverseVideo) $value
            if {$value} {
               # Reverse the video if desired
               ::apdl::noprint 1
               if {![file exists [file join \
                     $ReportArray(reportDir) report_cmap.cmap] ]} {
                  # Get the current color mapping
                  catch { ans_sendcommand \
                        )/cmap,report_cmap,cmap,'$ReportArray(reportDir)',save }
               }
               catch {ans_sendcommand )/RGB,INDEX,100,100,100, 0} err
               catch {ans_sendcommand )/RGB,INDEX,  0,  0,  0,15} err
               # Turn off the background image
               if {$ReportArray(grphBGimg) == -999} {
                  set ReportArray(grphBGimg) [ans_evalexpr zpbakiqr(1)]
               }
               catch {ans_sendcommand )/COLOR,PBAK,OFF} err
               catch {ans_sendcommand )/REPLOT} err
               ::apdl::noprint 0
            } else {
               # Set the color mapping back to what it was coming in.
               set grphChng 0
               if {[file exists [file join \
                     $ReportArray(reportDir) report_cmap.cmap] ]} {
                  ::apdl::noprint 1
                  catch { ans_sendcommand \
                        )/cmap,report_cmap,cmap,'$ReportArray(reportDir)' }
                  ::apdl::noprint 0
                  set grphChng 1
               }
               if {[ansys::report::getReportInfo grphBGimg] == 1} {
                  # Turn the background image back on
                  ::apdl::noprint 1
                  catch { ans_sendcommand )/COLOR,PBAK,ON }
                  ::apdl::noprint 0
                  set ReportArray(grphBGimg) -999
                  set grphChng 1
               }
               if {$grphChng} {
                  catch {ans_sendcommand )/REPLOT} err
               }
            }
            update

         }
         batchCapture {
            set ReportArray(batchCapture) $value
         }
      }
   }
   
   # --------------------------------------------------------------------------
   # GLOBAL PROCEDURE:  finished
   #
   # Purpose:        This procedure is called to finish the process of 
   #                 report generation.
   #
   # Arguments:      N/A
   #
   # Return Value:   N/A
   #
   # Comments:       The graphics will not set correctly back if reverse
   #                 video was performed.
   #
   # See Also:       N/A
   #
   # --------------------------------------------------------------------------
   proc finished { } {

      variable ReportArray

      if {[getReportInfo reverseVideo]} {
         setOptions reverseVideo 0
      }

      catch {file delete -force \
               [file join $ReportArray(reportDir) report_cmap.cmap]}

      sizeGraphics up

   }

   # START OF LISTING PROCS {
   # --------------------------------------------------------------------------
   # GLOBAL PROCEDURE:  outputcapture caption ansysCommand
   #
   # Purpose:        This procedure is called to capture the output listing
   #                 from the ansysCommand
   #
   # Arguments:      
   #                 caption
   #                    The caption to show before this listing.
   #                 ansysCommand
   #                    An ansys command that the ouput from is desired.
   #
   # Return Value:   N/A
   #
   # Comments:       This is used to capture things like area listings,
   #                 status, etc.
   #
   # See Also:       N/A
   #
   # --------------------------------------------------------------------------
   proc outputcapture { caption ansCmd } {
      variable ReportArray

      if {[string match {} $ReportArray(reportDir)]} {
         return -code error \
               -errorinfo "The report directory must be set with setdirectory" \
               "The report directory must be set with setdirectory"
      }

      incr ReportArray(curListingNumber)

      set caption [::apdl::substitute $caption]

      # Get the current jobname from ANSYS it may have changed
      set jobname [getReportInfo jobname]

      set relDir [::ansys::getRelativePath "$ReportArray(reportDir)" 10]
      ::apdl::nooutput 1
      catch {ans_sendcommand )/output,$jobname,lst,\'$relDir\',,FILE}
      catch {ans_sendcommand )$ansCmd} err
      ::apdl::nooutput 0

      switch -- $ReportArray(format) {
            javascript {
      set fileID [open [file join $ReportArray(reportDir) ansysListings.js] a]
      if {$ReportArray(curListingNumber) == 1} {
         puts $fileID "// JavaScript functions to access ANSYS listings\
               for report"
         if {[ans_getfilter 0 VERIFY]} {
            puts $fileID "// \"$ReportArray(reportName)\" created on DATE "
         } else {
            puts $fileID "// \"$ReportArray(reportName)\" created on [clock format [clock seconds]] "
         }
      }
      puts $fileID "function listing_$ReportArray(curListingNumber) (name) {"
      puts $fileID "   var undefined;"
      puts $fileID "   // jobname = '$ReportArray(jobname)'"
      puts $fileID "   if (name == undefined) {"
      puts $fileID "   document.writeln('<B>$caption</B>');"
      puts $fileID "   } else {"
      puts $fileID "   document.writeln('<B>' + name + '</B>');"
      puts $fileID "   }"
      puts $fileID "   document.writeln('<HR WIDTH=\"100%\">');"
      puts $fileID "   document.writeln('<PRE WIDTH=132>');"
      set fileOut [open [file join $ReportArray(reportDir) $jobname.lst] r]
      while {[gets $fileOut line] >= 0} {
         regsub -all {\\} $line {\\\\} line
         puts $fileID "document.writeln('$line');"
      }
      close $fileOut
      puts $fileID "document.writeln('</PRE>');"
      puts $fileID "document.writeln('<HR WIDTH=\"100%\">');"
      set fileOut [open [file join $ReportArray(reportDir) $jobname.lst] r]
      while {[gets $fileOut line] >= 0} {
         puts $fileID "//$line"
      }
      close $fileOut
      puts $fileID "}"
      close $fileID
         }
         html {
         }
         ds {
            if {[catch {getObjects} err]} {
               # DS is not present
            } else {
            set dsHTML ""
            append dsHTML "<PRE WIDTH=132>"
            set fileOut [open [file join \
                  $ReportArray(reportDir) $jobname.lst] r]
            while {[gets $fileOut line] >= 0} {
               regsub -all {\\} $line {\\\\} line
      #            append dsHTML "<BR>"
                  append dsHTML "$line \n"
            }
            close $fileOut
            append dsHTML "</PRE>"

            # Insert a post output object for the image
            ::AnsTComClient::eval \
            {set htmlObject [$dsPost_obj AddPostOutput]}
            #
            ::AnsTComClient::eval {for {set intZero 0} {$intZero <= -1} {incr i} {}}
            ::AnsTComClient::eval {$htmlObject SourceType $intZero}
            #
            ::AnsTComClient::eval [list set caption $caption]
            ::AnsTComClient::eval {$htmlObject Name $caption}
            #
   
            # Remove any braces
            regsub -all {\{} $dsHTML {} dsHTML
            regsub -all {\}} $dsHTML {} dsHTML
            # Add backslashes for "
            regsub -all {\"} $dsHTML {\\"} dsHTML
            # Add backslashes for *
            regsub -all {\*} $dsHTML {\\*} dsHTML
            # Add backslashes for \n
            regsub -all {\n} $dsHTML {\\n} dsHTML
            # Add backslashes for $
            regsub -all {\$} $dsHTML {\\$} dsHTML
            ::AnsTComClient::eval [subst -nobackslashes {set htmlText "${dsHTML}"}]
                  ::AnsTComClient::eval {$htmlObject HTML $htmlText}
            }
         }
      }
      # remove the temporary file
      catch {file delete -force \
               [file join $ReportArray(reportDir) $jobname.lst]}
      return
   }
   # END OF LISTING PROCS }

   # START OF TABLE PROCS {
   # --------------------------------------------------------------------------
   # GLOBAL PROCEDURE:  tablecapture tableID caption ?args?
   #
   # Purpose:        This procedure is called to capture a table of ANSYS
   #                 data.
   #
   # Arguments:      
   #                 tableID
   #                    The id of the table that is to be captured. This
   #                    is one of the following:
   #
   #                       1  -  The element types and number of and the
   #                             number of nodes used.
   #                       2  -  Material properties, this requires an
   #                             additional argument.  The material number
   #                             must be passed in.
   #                       13 -  The first three natural frequencies
   #
   #                 caption
   #                    The caption to show before this listing.
   #                 ansysCommand
   #                    An ansys command that the ouput from is desired.
   #
   # Return Value:   N/A
   #
   # Comments:       This is used to capture things like area listings,
   #                 status, etc.
   #
   # See Also:       N/A
   #
   # --------------------------------------------------------------------------
   proc tablecapture { tableID caption args} {
      variable ReportArray

      if {[string match {} $ReportArray(reportDir)]} {
         return -code error \
               -errorinfo "The report directory must be set with setdirectory" \
               "The report directory must be set with setdirectory"
      }

      incr ReportArray(curTableNumber)

      set caption [::apdl::substitute $caption]

      # Get the current jobname from ANSYS it may have changed
      set jobname [getReportInfo jobname]

      switch -- $tableID {
         0 {
            tableId0 $caption [lindex $args 0] [lindex $args 1]
         }
         1 {
            tableId1 $caption
         }
         2 {
            tableId2 $caption [lindex $args 0]
         }
         3 {
            tableId3 $caption
         }
         4 {
            tableId4 $caption
         }
         5 {
            tableId5 $caption
         }
         6 {
            tableId6 $caption
         }
         7 {
            tableId7 $caption
         }
         8 {
            tableId8 $caption
         }
         9 {
            tableId9 $caption
         }
         10 {
            tableId10 $caption
         }
         11 {
            tableId11 $caption
         }
         12 {
            tableId12 $caption
         }
         13 {
            tableId13 $caption
         }
         14 {
            tableId14 $caption
         }
         15 {
            tableId15 $caption
         }
         16 {
            tableId16 $caption
         }
         17 {
            tableId17 $caption
         }
         18 {
            tableId18 $caption
         }
         19 {
            tableId19 $caption
         }
         20 {
            tableId20 $caption
         }
         21 {
            tableId21 $caption
         }
         22 {
            tableId22 $caption
         }
         23 {
            tableId23 $caption
         }
         24 {
            tableId24 $caption
         }
         25 {
            tableId25 $caption
         }
         26 {
            tableId26 $caption
         }
         27 {
            tableId27 $caption
         }
         28 {
            tableId28 $caption
         }
         29 {
            tableId29 $caption
         }
         30 {
            tableId30 $caption
         }
         31 {
            tableId31 $caption
         }
         32 {
            tableId32 $caption
         }
         33 {
            tableId33 $caption
         }
         34 {
            tableId34 $caption
         }
         35 {
            tableId35 $caption
         }
         36 {
            tableId36 $caption
         }
         37 {
            tableId37 $caption
         }
         38 {
            tableId38 $caption
         }
         39 {
            tableId39 $caption
         }
         40 {
            tableId40 $caption
         }
         41 {
            tableId41 $caption
         }
         42 {
            tableId42 $caption
         }
         43 {
            tableId43 $caption
         }
         44 {
            tableId44 $caption
         }
         45 {
            tableId45 $caption
         }
         46 {
            tableId46 $caption
         }
         47 {
            tableId47 $caption
         }
         48 {
            tableId48 $caption
         }
      }
   }

   proc materialTable { matID } {

      set matProp [list \
         {1 {EX} {Modulus of elasticity} {X-Direction}} \
         {2 {EY} {Modulus of elasticity} {Y-Direction}} \
         {3 {EZ} {Modulus of elasticity} {Z-Direction}} \
         {10 {ALPX} {Thermal expansion coefficient} {X-Direction}} \
         {11 {ALPY} {Thermal expansion coefficient} {Y-Direction}} \
         {12 {ALPZ} {Thermal expansion coefficient} {Z-Direction}} \
         {60 {REFT} {Reference Temperature} {} } \
         {28 {PRXY} {Major Poisson\'s ratio} {Z-Plane} } \
         {29 {PRYZ} {Major Poisson\'s ratio} {X-Plane} } \
         {30 {PRXZ} {Major Poisson\'s ratio} {Y-Plane} } \
         {4 {NUXY} {Minor Poisson\'s ratio} {Z-Plane} } \
         {5 {NUYZ} {Minor Poisson\'s ratio} {X-Plane} } \
         {6 {NUXZ} {Minor Poisson\'s ratio} {Y-Plane} } \
         {7 {GXY} {Shear moduli} {Z-Plane} } \
         {8 {GYZ} {Shear moduli} {X-Plane} } \
         {9 {GXZ} {Shear moduli} {Y-Plane} } \
         {15 {DAMP} {Damping} {} } \
         {14 {MU} {Coefficient of friction} {} } \
         {13 {DENS} {Density} {} } \
         {22 {C} {Specific Heat} {} } \
         {26 {ENTH} Enthalpy {} } \
         {16 {KXX} {Thermal conductivity} {X-Direction} } \
         {17 {KYY} {Thermal conductivity} {Y-Direction} } \
         {18 {KZZ} {Thermal conductivity} {Z-Direction} } \
         {23 {HF} {Convection} {} } \
         {25 {EMIS} {Emissivity} {} } \
         {59 {QRATE} {Heat generation rate} {} } \
         {24 {VISC} {Viscosity} {} } \
         {46 {SONC} {Sonic velocity} {} } \
         {19 {RSVX} {Electrical resistivity} {X-Direction} } \
         {20 {RSVY} {Electrical resistivity} {Y-Direction} } \
         {21 {RSVZ} {Electrical resistivity} {Z-Direction} } \
         {34 {PERX} {Electrical permittivity} {X-Direction} } \
         {35 {PERY} {Electrical permittivity} {Y-Direction} } \
         {36 {PERZ} {Electrical permittivity} {Z-Direction} } \
         {31 {MURX} {Magnetic relative permeability} {X-Direction} } \
         {32 {MURY} {Magnetic relative permeability} {Y-Direction} } \
         {33 {MURZ} {Magnetic relative permeability} {Z-Direction} } \
         {37 {MGXX} {Magnetic coercive forces} {X-Direction} } \
         {38 {MGYY} {Magnetic coercive forces} {Y-Direction} } \
         {39 {MGZZ} {Magnetic coercive forces} {Z-Direction} } \
         {27 {LSST} {Dielectric loss tangent} {} } \
         ]

#         {{EX} {Modulus of elasticity} {X-Direction}} \
#         {{EY} {Modulus of elasticity} {Y-Direction}} \
#         {{EZ} {Modulus of elasticity} {Y-Direction}} \
#         {{ALPX} {Thermal expansion coefficient} {X-Direction}} \
#         {{ALPY} {Thermal expansion coefficient} {X-Direction}} \
#         {{ALPZ} {Thermal expansion coefficient} {X-Direction}} \
#         {{REFT} {Reference Temperature} {} } \
#         {{PRXY} {Major Poisson\'s ratio} {X-Direction} } \
#         {{PRYZ} {Major Poisson\'s ratio} {Y-Direction} } \
#         {{PRXZ} {Major Poisson\'s ratio} {Z-Direction} } \
#         {{NUXY} {Minor Poisson\'s ratio} {X-Direction} } \
#         {{NUYZ} {Minor Poisson\'s ratio} {Y-Direction} } \
#         {{NUXZ} {Minor Poisson\'s ratio} {Z-Direction} } \
#         {{GXY} {Shear moduli} {Z-Plane} } \
#         {{GYZ} {Shear moduli} {X-Plane} } \
#         {{GXZ} {Shear moduli} {Y-Plane} } \
#         {{DAMP} {Damping} {} } \
#         {{MU} {Coefficient of friction} {} } \
#         {{DENS} {Density} {} } \
#         {{C} {Specific Heat} {} } \
#         {{ENTH} {Enthalpy} {} } \
#         {{KXX} {Thermal conductivity} {X-Direction} } \
#         {{KYY} {Thermal conductivity} {Y-Direction} } \
#         {{KZZ} {Thermal conductivity} {Z-Direction} } \
#         {{HF} {Convection} {} } \
#         {{EMIS} {Emissivity} {} } \
#         {{QRATE} {Heat generation rate} {} } \
#         {{VISC} {Viscosity} {} } \
#         {{SONC} {Sonic velocity} {} } \
#         {{RSVX} {Electrical resistivity} {X-Direction} } \
#         {{RSVY} {Electrical resistivity} {Y-Direction} } \
#         {{RSVZ} {Electrical resistivity} {Z-Direction} } \
#         {{PERX} {Electrical permittivity} {X-Direction} } \
#         {{PERY} {Electrical permittivity} {Y-Direction} } \
#         {{PERZ} {Electrical permittivity} {Z-Direction} } \
#         {{MURX} {Magnetic relative permeability} {X-Direction} } \
#         {{MURY} {Magnetic relative permeability} {Y-Direction} } \
#         {{MURZ} {Magnetic relative permeability} {Z-Direction} } \
#         {{MGXX} {Magnetic coercive forces} {X-Direction} } \
#         {{MGYY} {Magnetic coercive forces} {Y-Direction} } \
#         {{MGZZ} {Magnetic coercive forces} {Z-Direction} } \
#         {{LSST} {Dielectric loss tangent} {} } \
#
      catch {unset matReturn}
      set tempDep 0
      foreach material $matProp {
         catch {unset matlInfo}
         set matlInqr [lindex $material 0]
         set matl [lindex $material 1]
         set str1 [lindex $material 2]
         set str2 [lindex $material 3]
         # Has this material been defined ?
         set matDefined [ans_evalexpr mpinqr($matID,$matlInqr,1)]
         if {$matDefined} {
         # Is this a temperature dependent material
            set ntemp [ans_evalexpr mpinqr($matID,$matlInqr,3)]
            if {$ntemp > 1} {
               set tempDep $ntemp
               lappend matlInfo "$str1 $str2"
               for {set i 1} {$i <= $ntemp} {incr i} {
                  set temp [ans_getvalue $matl,$matID,TVAL,$i]
                  set matlValue [ans_getvalue $matl,$matID,TEMP,$temp]
                  lappend matlInfo $temp
                  lappend matlInfo $matlValue
               }
            } else {
               set matlValue [ans_getvalue $matl,$matID]
                  lappend matlInfo "$str1 $str2"
                  lappend matlInfo {}
                  lappend matlInfo $matlValue
            }
         lappend matReturn $matlInfo
         }
      }
      if {[info exists tempDep] && [info exists matReturn]} {
         return [list $tempDep $matReturn]
      } else {
         return -999
      }
   }



   # ************************************************************
   #  tableid1 creates a table of the finite element entities used
   #  in the analysis.
   # ************************************************************
   proc tableId1 { caption } {

      # Build up the rows of information
      # The elementNames are obtained from the ELCXXX routines
      set elementNames [list NULL \
         LINK1 PLANE2 BEAM3 BEAM4 SOLID5 SURF6 COMBIN7 \
         LINK8 INFIN9 LINK10 LINK11 CONTAC12 PLANE13 COMBIN14 FLUID15 PIPE16 \
         PIPE17 PIPE18 SURF19 PIPE20 MASS21 SURF22 BEAM23 BEAM24 PLANE25 \
         CONTAC26 MATRIX27 SHELL28 FLUID29 FLUID30 LINK31 LINK32 LINK33 \
         LINK34 PLANE35 SOURC36 COMBIN37 FLUID38 COMBIN39 COMBIN40 SHELL41 \
         PLANE42 SHELL43 BEAM44 SOLID45 SOLID46 INFIN47 CONTAC48 CONTAC49 \
         MATRIX50 SHELL51 CONTAC52 PLANE53 BEAM54 PLANE55 HYPER56 SHELL57 \
         HYPER58 PIPE59 PIPE60 SHELL61 SOLID62 SHELL63 SOLID64 SOLID65 \
         FLUID66 PLANE67 LINK68 SOLID69 SOLID70 MASS71 SOLID72 SOLID73 \
         HYPER74 PLANE75 SURF76 PLANE77 PLANE78 FLUID79 FLUID80 FLUID81 \
         PLANE82 PLANE83 HYPER84 SOLID85 HYPER86 SOLID87 VISCO88 VISCO89 \
         SOLID90 SHELL91 SOLID92 SHELL93 STIF94 SOLID95 SOLID96 SOLID97 \
         SOLID98 SHELL99 UNKNOWN UNKNOWN UNKNOWN UNKNOWN UNKNOWN \
         UNKNOWN VISCO106 VISCO107 VISCO108 VISCO109 INFIN110 INFIN111 \
         UNKNOWN HF113 HF114 INTER115 FLUID116 SOLID117 UNKNOWN HF119 \
         HF120 PLANE121 SOLID122 SOLID123 CIRCU124 CIRCU125 UNKNOWN \
         UNKNOWN UNKNOWN FLUID129 FLUID130 FLUID131 FLUID132 FLUID133 \
         FLUID134 FLUID135 FLUID136 FLUID137 FLUID138 FLUID139 FLUID140 \
         FLUID141 FLUID142 SHELL143 UNKNOWN PLANE145 PLANE146 SOLID147 \
         SOLID148 UNKNOWN SHELL150 SURF151 SURF152 SURF153 SURF154 \
         SOIL155 SOIL156 SHELL157 HYPER158 UNKNOWN LINK160 BEAM161 \
         SHELL162 SHELL163 SOLID164 COMBI165 MASS166 LINK167 UNKNOWN \
         TARGE169 TARGE170 CONTA171 CONTA172 CONTA173 CONTA174 UNKNOWN \
         UNKNOWN UNKNOWN UNKNOWN UNKNOWN UNKNOWN SHELL181 PLANE182 \
         UNKNOWN UNKNOWN SOLID185 UNKNOWN UNKNOWN BEAM188 BEAM189 \
         UNKNOWN UNKNOWN UNKNOWN RIGID193 RIGID194 ]

      set maxElemTypeNum [ans_evalexpr etyiqr(0,14)]
      set maxElemNum [ans_evalexpr elmiqr(0,14)]
      set maxNodeNum [ans_evalexpr ndinqr(0,12)]
      ::apdl::noprint 1
      ::apdl::noerr 1
      catch {ans_sendcommand )*DEL,_z1} err
      catch {ans_sendcommand )*DIM,_z1,,$maxElemNum} err
      catch {ans_sendcommand )*VGET,_z1(1),ELEM,1,ATTR,TYPE} err
      eval set elemUsed [lindex [ans_getvector _z1] 0]
      catch {ans_sendcommand )*DEL,_z1} err
      ::apdl::noprint 0
      ::apdl::noerr 0
      set elemUsed [lsort $elemUsed]
      set i 1
      while {$i} {
         regexp {([^\.]+)} [lindex $elemUsed 0] match type
         set reg "\[^$type\]\\."
         set next [lsearch -regexp $elemUsed $reg]
         if {$next == -1} {
            lappend elemType(types) $type
            set elemType($type,number) [llength $elemUsed]
            set elemType($type,enam) [ans_getvalue etype,$type,attr,enam]
            set elemType($type,enam) \
                  [lindex $elementNames $elemType($type,enam)]
            set i 0
         } else {
            if {$type != 0} {
               lappend elemType(types) $type
               set elemType($type,number) $next
               set elemType($type,enam) [ans_getvalue etype,$type,attr,enam]
               set elemType($type,enam) \
                     [lindex $elementNames $elemType($type,enam)]
            }
            set elemUsed [lreplace $elemUsed 0 [expr $next - 1]]
         }
      }

      # Build up a list of lists to feed to the genHTMLTable
      lappend elements [list {<b> "Entity" </b>} {<b> "Number Defined" </b>}]
      foreach type $elemType(types) {
            lappend elements [subst "{<b> {$elemType($type,enam)} </b>} \
               {{} {$elemType($type,number)} {}}"]
      }
      lappend elements [subst [list {<b> "Nodes" </b>} {{} $maxNodeNum {}}]]

      genHTMLTable $caption $elements

      return
   }

   # ************************************************************
   #  tableid2 creates a table of the requested material ID's
   #  properties used in the analysis.
   # ************************************************************
   proc tableId2 { caption materialID } {
      variable ReportArray

      set defMaterials [materialTable $materialID]
      if {$defMaterials == -999} {
         ans_senderror 3 "Material $materialID contains only nonlinear material properties."
         return
      }

      set fileID [open [file join $ReportArray(reportDir) ansysTables.js] a]
      if {$ReportArray(curTableNumber) == 1} {
         puts $fileID "// JavaScript functions to access ANSYS tables\
               for report"
         if {[ans_getfilter 0 VERIFY]} {
            puts $fileID "// \"$ReportArray(reportName)\" created on DATE "
         } else {
            puts $fileID "// \"$ReportArray(reportName)\" created on [clock format [clock seconds]] "
         }
      }
      puts $fileID "function table_$ReportArray(curTableNumber) (name) {"
	   puts $fileID "   var undefined;"
      puts $fileID "   // jobname = '$ReportArray(jobname)'"
      puts $fileID "   document.writeln('<table>');"
      puts $fileID "   if (name == undefined) {"
      puts $fileID "      document.writeln('<tr><td><B>$caption</B></td></tr>');"
      puts $fileID "   } else {"
      puts $fileID "      document.writeln('<tr><B>' + name + '</B></tr>');"
      puts $fileID "   }"

      puts $fileID "   document.writeln('<tr>');"
      puts $fileID "   document.writeln('<td>');"
      puts $fileID "   document.writeln('<table border=1 cellpadding=7 cellspacing=0>');"

      set tempDepend [lindex $defMaterials 0]
      set defMaterials [lindex $defMaterials 1]
      foreach matProp $defMaterials {
         set title [lindex $matProp 0]
         set matProp [lrange $matProp 1 end]
         if {$tempDepend} {
            set numRowSpan \
                  [expr [expr $tempDepend%6]?[expr ($tempDepend/6+1)*2]\
                     :[expr ($tempDepend/6)*2]]
            set numColSpan \
                  [expr [expr $tempDepend >= 6]?6:[expr $tempDepend + 1]]

            puts $fileID "   document.writeln('<tr>');"
            puts $fileID \
               "   document.writeln('<td colspan=\"$numColSpan\"><b>$title</b></td>');"
            puts $fileID "   document.writeln('</tr>');"

            # Put out the temperature and values 6 at a time.
            for {set i 0} {$i < $tempDepend} {incr i 6} {
               puts $fileID "   document.writeln('<tr>');"
               puts $fileID \
                  "   document.writeln('<td>[::msgcat::mc "Temp"]</td>');"
               set matSubSet \
                     [lrange $matProp [expr $i * 2] [expr ($i * 2) + 11]]
               foreach {temp val} $matSubSet {
                  puts $fileID \
                     "   document.writeln('<td align=center width=110>[format "%12.4g" $temp]</td>');"
               }
               puts $fileID "   document.writeln('</tr>');"
               puts $fileID "   document.writeln('<tr>');"
               puts $fileID \
                  "   document.writeln('<td>[::msgcat::mc "Value"]</td>');"
               foreach {temp val} $matSubSet {
                  puts $fileID \
                     "   document.writeln('<td align=center>[format "%12.4g" $val]</td>');"
               }
               puts $fileID "   document.writeln('</tr>');"
               puts $fileID "   document.writeln('<tr>');"
            }
         } else {
            set val [lindex $matProp 1]
            puts $fileID "   document.writeln('<tr>');"
            puts $fileID "   document.writeln('<td><b>$title</b></td>');"
            puts $fileID "   document.writeln('<td align=center>[format "%12.4g" $val]</td>');"
            puts $fileID "   document.writeln('</tr>');"
            puts $fileID "   document.writeln('<tr>');"
         }
      }
      puts $fileID "   document.writeln('</tr>');"
      puts $fileID "   document.writeln('</table>');"

      puts $fileID "   document.writeln('</td>');"
      puts $fileID "   document.writeln('</tr>');"
      puts $fileID "   document.writeln('</table>');"

      puts $fileID "// The following lines are tab delimited and can be pasted in tables that"
      puts $fileID "// recognize tab delimited text as a table. The table should be one more"
      puts $fileID "// than needed to allow deletion of the first column to remove the //."
      puts $fileID ""
      puts $fileID [format "//\t%s" $caption]
      puts $fileID ""
      foreach matProp $defMaterials {
         set title [lindex $matProp 0]
         set matProp [lrange $matProp 1 end]
         if {$tempDepend} {
            puts $fileID [format "//\t%s" $title]
            set numRowSpan \
                  [expr [expr $tempDepend%6]?[expr ($tempDepend/6+1)*2]\
                     :[expr ($tempDepend/6)*2]]

            # Put out the temperature and values 6 at a time.
            for {set i 0} {$i < $tempDepend} {incr i 6} {
               set matSubSet \
                     [lrange $matProp [expr $i * 2] [expr ($i * 2) + 11]]
               catch {unset matTemp}
               catch {unset matVal}
               puts -nonewline $fileID [format "//\t%s" [msgcat::mc "Temp"]]
               foreach {temp val} $matSubSet {
                  puts -nonewline $fileID [format "\t%12.4g" $temp]
               }
               puts $fileID ""
               puts -nonewline $fileID [format "//\t%s" [msgcat::mc "Value"]]
               foreach {temp val} $matSubSet {
                  puts -nonewline $fileID [format "\t%12.4g" $val]
               }
               puts $fileID ""
            }
         } else {
            set val [lindex $matProp 1]
            puts $fileID [format "//\t%s\t%12.4g" $title $val]
         }
      }
      puts $fileID "}"
      close $fileID
      return
   }

   proc tableId3 { caption } {
      variable ReportArray

      set numTbl 0

      # Constraints
      set nC [ans_evalexpr disiqr(0,14)]
      set kC [ans_evalexpr kdsiqr(0,14)]
      set lC [expr [ans_evalexpr ldsiqr(0,14)] \
            + [ans_evalexpr lcniqr(0,14)]]
      set aC [expr [ans_evalexpr adsiqr(0,14)] \
            + [ans_evalexpr acniqr(0,14)]]
      # Constraint Equations
      set cpC [ans_evalexpr cpinqr(0,14)]
      set ceC [ans_evalexpr ceinqr(0,14)]


      if {[expr $nC + $kC + $lC + $aC + $cpC + $ceC]} {
         lappend bcs [list " " {<b> "Number of Constraints" </b>}]
         incr numTbl
      }
      if {$nC} {
         lappend bcs [list {<b> Node </b>}      $nC]
      }
      if {$kC} {
         lappend bcs [list {<b> Keypoints </b>} $kC]
      }
      if {$lC} {
         lappend bcs [list {<b> Lines </b>}     $lC]
      }
      if {$aC} {
         lappend bcs [list {<b> Areas </b>}     $aC]
      }
      if {$cpC} {
         lappend bcs [list {<b> Couplings </b>} $cpC]
      }
      if {$ceC} {
         lappend bcs [list {<b> Equations </b>} $ceC]
      }

      # Forces
      set nF [ans_evalexpr foriqr(0,14)]
      set kF [ans_evalexpr kfoiqr(0,14)]

      if {[expr $nF + $kF]} {
         if {$numTbl} {
            lappend bcs [list "" ]
         }
         lappend bcs [list " " {<b> "Number of Forces" </b>}]
         incr numTbl
      }
      if {$nF} {
         lappend bcs [list {<b> Node </b>}      $nF]
      }
      if {$kF} {
         lappend bcs [list {<b> Keypoints </b>} $kF]
      }

      # Surface Loads
      set eSL 0
	  # SURFLOAD_PRES,SURFLOAD_CONV,
	  # SURFLOAD_CHRG,SURFLOAD_IMPD,
	  # BODYLOAD_SELV
      set inqr [list \
	        ansElemSurfLoadIqr(1,0,1,12) ansElemSurfLoadIqr(2,0,1,12) \
			ansElemSurfLoadIqr(6,0,1,12) ansElemSurfLoadIqr(3,0,1,12) \
            ansElemBodyLoadIqr(0,24,12)]
      foreach iqr $inqr {
         set eSL [expr $eSL + [ans_evalexpr $iqr]]
      }
      set lSL 0
      set inqr [list lpriqr(0,14) lctiqr(0,14) lmgiqr(0,14) \
            limiqr(0,14) lfsiqr(0,14)]
      foreach iqr $inqr {
         set lSL [expr $lSL + [ans_evalexpr $iqr]]
      }
      set aSL 0
      set inqr [list apriqr(0,14) actiqr(0,14) amgiqr(0,14) \
            aimiqr(0,14) afsiqr(0,14)]
      foreach iqr $inqr {
         set aSL [expr $aSL + [ans_evalexpr $iqr]]
      }

      if {[expr $eSL + $lSL + $aSL]} {
         if {$numTbl} {
            lappend bcs [list "" ]
         }
         lappend bcs [list " " {<b> "Number of Surface Loads" </b>}]
         incr numTbl
      }
      if {$eSL} {
         lappend bcs [list {<b> Elements </b>}  $eSL]
      }
      if {$lSL} {
         lappend bcs [list {<b> Lines </b>}     $lSL]
      }
      if {$aSL} {
         lappend bcs [list {<b> Areas </b>}     $aSL]
      }

      # Body Loads
      set nBL 0
	  # BODYLOAD_TEMP,BODYLOAD_HGEN,
	  # BODYLOAD_CHRG,BODYLOAD_FLUE
	  # BODYLOAD_DGEN,BODYLOAD_MVDI,
	  # BODYLOAD_JS,BODYLOAD_VLTG
	  # BODYLOAD_EF,BODYLOAD_H,
	  # BODYLOAD_PORT
      set inqr [list \
	        ansNodeBodyLoadIqr(1,0,2)  ansNodeBodyLoadIqr(3,0,2) \
		    ansNodeBodyLoadIqr(6,0,2)  ansNodeBodyLoadIqr(2,0,2) \
            ansNodeBodyLoadIqr(4,0,2)  ansNodeBodyLoadIqr(5,0,2) \
			ansNodeBodyLoadIqr(7,0,2)  ansNodeBodyLoadIqr(8,0,2) \
            ansNodeBodyLoadIqr(10,0,2) ansNodeBodyLoadIqr(10,0,2) \
			ansNodeBodyLoadIqr(11,0,2) ]
      foreach iqr $inqr {
         set nBL [expr $nBL + [ans_evalexpr $iqr]]
      }
      set eBL 0
      set eBL 0
	  # BODYLOAD_TEMP,BODYLOAD_HGEN,
	  # BODYLOAD_FLUE,BODYLOAD_DGEN
	  # BODYLOAD_MVDI,BODYLOAD_JS,
	  # BODYLOAD_VLTG,BODYLOAD_CHRG
	  # BODYLOAD_EF,BODYLOAD_H,
	  # BODYLOAD_PORT
      set inqr [list \
	        ansElemBodyLoadIqr(1,0,14) ansElemBodyLoadIqr(3,0,14) \
            ansElemBodyLoadIqr(2,0,14) ansElemBodyLoadIqr(4,0,14) \
			ansElemBodyLoadIqr(5,0,14) ansElemBodyLoadIqr(7,0,14) \
            ansElemBodyLoadIqr(8,0,14) ansElemBodyLoadIqr(6,0,14) \
			ansElemBodyLoadIqr(10,0,14) ansElemBodyLoadIqr(10,0,14) \
            ansElemBodyLoadIqr(11,0,14) ]
      foreach iqr $inqr {
         set eBL [expr $eBL + [ans_evalexpr $iqr]]
      }
      set kBL 0
      set inqr [list ktpiqr(0,14) khgiqr(0,14) kcdiqr(0,14) \
            kfuiqr(0,14) kmciqr(0,14) kvdiqr(0,14) kdciqr(0,14) \
            kvltiqr(0,14) kefiqr(0,14) khiqr(0,14) kprtiqr(0,14) ]
      foreach iqr $inqr {
         set kBL [expr $kBL + [ans_evalexpr $iqr]]
      }
                                                              
      if {[expr $nBL + $eBL + $kBL]} {
         if {$numTbl} {
            lappend bcs [list "" ]
         }
         lappend bcs [list {" "} [list <b> [msgcat::mc "Number of Body Loads"] </b>]]
         incr numTbl
      }
      if {$nBL} {
         lappend bcs [list {<b> [msgcat::mc "Node"] </b>}      $nBL]
      }
      if {$eBL} {
         lappend bcs [list {<b> [msgcat::mc "Elements"] </b>}  $eBL]
      }
      if {$kBL} {
         lappend bcs [list {<b> [msgcat::mc "Keypoints"] </b>} $kBL]
      }

      # Temperature
      if {$numTbl} {
         lappend bcs [list "" ]
      }
      lappend bcs [list [list <b> [msgcat::mc "Temperature"] </b>] \
            {<b> [msgcat::mc "Value"] </b>}]
      set numUniBC [ans_evalexpr bfinqr(1)]
      lappend bcs [list [list <b> [msgcat::mc "Uniform"] </b>] $numUniBC]
      set numRefBC [ans_getvalue common,,bfcom,,real,8]
      lappend bcs [list [list <b> [msgcat::mc "Reference"] </b>] $numRefBC]

      if {[ans_getvalue commmon,,acelcm,,int,1]} {
      }
         # Linear Accel
         set lacelX [format "%12.4g" [ans_getvalue common,,acelcm,,real,19]]
         set lacelY [format "%12.4g" [ans_getvalue common,,acelcm,,real,20]]
         set lacelZ [format "%12.4g" [ans_getvalue common,,acelcm,,real,21]]
         set lacelK [expr $lacelX + $lacelY + $lacelZ]
         # Angular Vel
         set AvelX [format "%12.4g" [ans_getvalue common,,acelcm,,real,25]]
         set AvelY [format "%12.4g" [ans_getvalue common,,acelcm,,real,26]]
         set AvelZ [format "%12.4g" [ans_getvalue common,,acelcm,,real,27]]
         set AvelK [expr $AvelX + $AvelY + $AvelZ]
         # Angular Accel
         set AacelX [format "%12.4g" [ans_getvalue common,,acelcm,,real,28]]
         set AacelY [format "%12.4g" [ans_getvalue common,,acelcm,,real,29]]
         set AacelZ [format "%12.4g" [ans_getvalue common,,acelcm,,real,30]]
         set AacelK [expr $AacelX + $AacelY + $AacelZ]

         if {[expr $lacelK + $AvelK + $AacelK ]} {
            lappend bcs [list "" ]
            lappend bcs [list [list <b> [msgcat::mc "Global Cartesian"] </b>] {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}]
         }
         if {$lacelK} {
            lappend bcs [list [list <b> [msgcat::mc "Linear acceleration"] </b>] \
               $lacelX $lacelY $lacelZ]
         }
         if {$AvelK} {
            lappend bcs [list [list <b> [msgcat::mc "Angular velocity"] </b>] \
               $AvelX $AvelY $AvelZ]
         }
         if {$AacelK} {
            lappend bcs [list [list <b> [msgcat::mc "Angular acceleration"] </b>] \
               $AacelX $AacelY $AacelZ]
         }
      #
      # Inertia Loading
      #
      # Linear Accel
      if {[ans_getvalue commmon,,acelcm,,int,2]} {
      }
         set IL(lacelX) [format "%12.4g" [ans_getvalue common,,acelcm,,real,22]]
         set IL(lacelY) [format "%12.4g" [ans_getvalue common,,acelcm,,real,23]]
         set IL(lacelZ) [format "%12.4g" [ans_getvalue common,,acelcm,,real,24]]
         set IL(lacelK) [expr $IL(lacelX) + $IL(lacelY) + $IL(lacelZ)]
         # Angular Vel
         set IL(AvelX) [format "%12.4g" [ans_getvalue common,,acelcm,,real,31]]
         set IL(AvelY) [format "%12.4g" [ans_getvalue common,,acelcm,,real,32]]
         set IL(AvelZ) [format "%12.4g" [ans_getvalue common,,acelcm,,real,33]]
         set IL(AvelK) [expr $IL(AvelX) + $IL(AvelY) + $IL(AvelZ)]
         # Angular Accel
         set IL(AacelX) [format "%12.4g" [ans_getvalue common,,acelcm,,real,34]]
         set IL(AacelY) [format "%12.4g" [ans_getvalue common,,acelcm,,real,35]]
         set IL(AacelZ) [format "%12.4g" [ans_getvalue common,,acelcm,,real,36]]
         set IL(AacelK) [expr $IL(AacelX) + $IL(AacelY) + $IL(AacelZ)]
         # Number of Components
         set IL(numCM) [ans_getvalue common,,omegacm,,int,4]

         if {[expr $IL(lacelK) + $IL(AvelK) + $IL(AacelK)]} {
            lappend bcs [list "" ]
            lappend bcs [list [list <b> [msgcat::mc "Global Cartesian"] </b>] {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}]
         }
         if {$IL(lacelK)} {
            lappend bcs [list [list <b> [msgcat::mc "Inertia Linear acceleration"] </b>] \
               $IL(lacelX) $IL(lacelY) $IL(lacelZ)]
         }
         if {$IL(AvelK)} {
            lappend bcs [list [list <b> [msgcat::mc "Inertia Angular velocity"] </b>] \
               $IL(AvelX) $IL(AvelY) $IL(AvelZ)]
         }
         if {$IL(AacelK)} {
            lappend bcs [list [list <b> [msgcat::mc "Inertia Angular acceleration"] </b>] \
               $IL(AacelX) $IL(AacelY) $IL(AacelZ)]
         }
   
         if {$IL(numCM)} {
            lappend bcs [list "" ]
            lappend bcs [list [list <b> [msgcat::mc "Number of Inertia Component Loads"] </b>] $IL(numCM)]
         }

      genHTMLTable $caption [subst \
         $bcs
         ]

      return
   }

   # Sum of Reaction Forces
   proc tableId4 { caption } {

      ::apdl::nooutput 1 scratch.gui 0
      ::apdl::noerr 1
      if {[catch {ans_sendcommand )PRRSOL,F} err]} {
         # We have nothing defined at this point
         ::apdl::nooutput 0
         set reactX invalid
         set reactY invalid
         set reactZ invalid
      } else {
   
         ::apdl::nooutput 0
         if {[catch {open scratch.gui r} fileID]} {
            set reactX invalid
            set reactY invalid
            set reactZ invalid
         } else {
            set checkNow 0
            while {[gets $fileID line] >= 0} {
                if {$checkNow} {
                   set reactX [lindex $line 1]
                   if {[string match [list] $reactX]} {
                     set reactX 0.0
                   }
                   set reactX [format "%#12.4g" $reactX]
                   #
                   set reactY [lindex $line 2]
                   if {[string match [list] $reactY]} {
                     set reactY 0.0
                   }
                   set reactY [format "%#12.4g" $reactY]
                   #
                   set reactZ [lindex $line 3]
                   if {[string match [list] $reactZ]} {
                     set reactZ 0.0
                   }
                   set reactZ [format "%#12.4g" $reactZ]
                   break
                }
                if { [regexp -nocase {total values} $line ]} {
                   set checkNow 1        
                }
            } 
            close $fileID
         }
      }
      ::apdl::noerr 0
              
      genHTMLTable $caption [subst [list \
         [list {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}] \
         [list {{} $reactX {}} {{} $reactY {}} {{} $reactZ {}}] \
         ]] 0
      return
   }

   # Sum of Reaction Moments
   proc tableId5 { caption } {

      ::apdl::nooutput 1
      # save users selected node and element set
      catch {ans_sendcommand )cm,_temp,nodes} err
      catch {ans_sendcommand )cm,_tempe,elem } err
      # need all elements that contribute
      catch {ans_sendcommand )esel,all } err

      # get the X reactions
      # select nodes with imposed disps
      catch {ans_sendcommand )nsel,s,d,u,-.00001,.00001} err
      catch {ans_sendcommand )nsel,u,f,mx,.00001,1e20 } err
      catch {ans_sendcommand )nsel,u,f,mx,-.00001,-1e20 } err
      catch {ans_sendcommand )\$fsum,,1 } err
      set reactMX [format "%12.4g" [ans_getvalue fsum,,item,mx]]

      # get the Y reactions
      # select nodes with imposed disps
      catch {ans_sendcommand )nsel,s,d,u,-.00001,.00001} err
      catch {ans_sendcommand )nsel,u,f,my,.00001,1e20 } err
      catch {ans_sendcommand )nsel,u,f,fy,-.00001,-1e20 } err
      catch {ans_sendcommand )\$fsum,,1 } err
      set reactMY [format "%12.4g" [ans_getvalue fsum,,item,my]]

      # get the Z reactions
      # select nodes with imposed disps
      catch {ans_sendcommand )nsel,s,d,u,-.00001,.00001} err
      catch {ans_sendcommand )nsel,u,f,mz,.00001,1e20 } err
      catch {ans_sendcommand )nsel,u,f,mz,-.00001,-1e20 } err
      catch {ans_sendcommand )\$fsum,,1 } err
      set reactMZ [format "%12.4g" [ans_getvalue fsum,,item,mz]]

      # get the users selected node and element set back
      catch {ans_sendcommand )cmsel,s,_temp   } err
      catch {ans_sendcommand )cmsel,s,_tempe } err
      # remove temporary node and element components
      catch {ans_sendcommand )cmdel,_temp   } err
      catch {ans_sendcommand )cmdel,_tempe } err
      ::apdl::nooutput 0

      ::apdl::nooutput 1 scratch.gui 0
      ::apdl::noerr 1
      if {[catch {ans_sendcommand )PRRSOL,M} err]} {
         # We have nothing defined at this point
         ::apdl::nooutput 0
         set reactMX invalid
         set reactMY invalid
         set reactMZ invalid
      } else {
   
         ::apdl::nooutput 0
         if {[catch {open scratch.gui r} fileID]} {
            set reactMX invalid
            set reactMY invalid
            set reactMZ invalid
         } else {
            set checkNow 0
            while {[gets $fileID line] >= 0} {
                if {$checkNow} {
                   set reactMX [lindex $line 1]
                   if {[string match [list] $reactMX]} {
                     set reactMX 0.0
                   }
                   set reactMX [format "%#12.4g" $reactMX]
                   #
                   set reactMY [lindex $line 2]
                   if {[string match [list] $reactMY]} {
                     set reactMY 0.0
                   }
                   set reactMY [format "%#12.4g" $reactMY]
                   #
                   set reactMZ [lindex $line 3]
                   if {[string match [list] $reactMZ]} {
                     set reactMZ 0.0
                   }
                   set reactMZ [format "%#12.4g" $reactMZ]
                   break
                }
                if { [regexp -nocase {total values} $line ]} {
                   set checkNow 1        
                }
            } 
            close $fileID
         }
      }
      ::apdl::noerr 0
              
      genHTMLTable $caption [subst [list \
         [list {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}] \
         [list {{} $reactMX {}} {{} $reactMY {}} {{} $reactMZ {}}] \
         ]] 0
      return
   }

   # Max Displacements
   proc tableId6 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,u,x,,1,1 } err
      set minUX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxUX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,u,y,,1,1 } err
      set minUY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxUY [format "%12.4g" [ans_getvalue sort,,max]]
      if {[ans_getvalue active,,dof,max] > 2} {
         catch {ans_sendcommand )nsort,u,z,,1,1 } err
         set minUZ [format "%12.4g" [ans_getvalue sort,,min]]
         set maxUZ [format "%12.4g" [ans_getvalue sort,,max]]
      } else {
         set minUZ [format "%12.4g" 0.0]
         set maxUZ [format "%12.4g" 0.0]
      }
      catch {ans_sendcommand )nsort,u,sum,,,1 } err
      set minUSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxUSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Maximum </b>} {{} $maxUX {}} {{} $maxUY {}} {{} $maxUZ {}} {{} $maxUSUM {}}] \
         ]]

      return
   }

   # Directional Stress
   proc tableId7 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,s,x } err
      # send the command twice to get around a bug (reloads the PG object)
      catch {ans_sendcommand )nsort,s,x } err
      set minSX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxSX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,s,y } err
      set minSY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxSY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,s,z } err
      set minSZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxSZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}] \
         [list {<b> Minimum </b>} {{} $minSX {}} {{} $minSY {}} {{} $minSZ {}}] \
         [list {<b> Maximum </b>} {{} $maxSX {}} {{} $maxSY {}} {{} $maxSZ {}}] \
         ]]
      return
   }

   # Shear Stress
   proc tableId8 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,s,xy } err
      set minXY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxXY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,s,yz } err
      set minYZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxYZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,s,xz } err
      set minXZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxXZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "XY" </b>} {<b> "YZ" </b>} {<b> "XZ" </b>}] \
         [list {<b> Minimum </b>} {{} $minXY {}} {{} $minYZ {}} {{} $minXZ {}}] \
         [list {<b> Maximum </b>} {{} $maxXY {}} {{} $maxYZ {}} {{} $maxXZ {}}] \
         ]]
      return
   }

   # Principal Stress
   proc tableId9 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,s,1 } err
      set minS1 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxS1 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,s,2 } err
      set minS2 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxS2 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,s,3 } err
      set minS3 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxS3 [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "1st" </b>} {<b> "2nd" </b>} {<b> "3rd" </b>}] \
         [list {<b> Minimum </b>} {{} $minS1 {}} {{} $minS2 {}} {{} $minS3 {}}] \
         [list {<b> Maximum </b>} {{} $maxS1 {}} {{} $maxS2 {}} {{} $maxS3 {}}] \
         ]]
      return
   }

   # Equivalent Stress & Stress Intensity
   proc tableId10 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,s,eqv } err
      set minSE [format "%12.4g" [ans_getvalue sort,,min]]
      set maxSE [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,s,int } err
      set minSI [format "%12.4g" [ans_getvalue sort,,min]]
      set maxSI [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Stress Intensity" </b>} {<b> "Equivalent Stress" </b>}] \
         [list {<b> Minimum </b>} {{} $minSI {}} {{} $minSE {}}] \
         [list {<b> Maximum </b>} {{} $maxSI {}} {{} $maxSE {}}] \
         ]]
      return
   }

   # Thermal Gradients
   proc tableId11 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,tg,x } err
      set minTGX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTGX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,tg,y } err
      set minTGY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTGY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,tg,z } err
      set minTGZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTGZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,tg,sum } err
      set minTGSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTGSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Minimum </b>} {{} $minTGX {}} {{} $minTGY {}} {{} $minTGZ {}} {{} $minTGSUM {}}] \
         [list {<b> Maximum </b>} {{} $maxTGX {}} {{} $maxTGY {}} {{} $maxTGZ {}} {{} $maxTGSUM {}}] \
         ]]
      return
   }

   # Thermal Flux
   proc tableId12 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,tf,x } err
      set minTFX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTFX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,TF,y } err
      set minTFY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTFY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,TF,z } err
      set minTFZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTFZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,TF,sum } err
      set minTFSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTFSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Minimum </b>} {{} $minTFX {}} {{} $minTFY {}} {{} $minTFZ {}} {{} $minTFSUM {}}] \
         [list {<b> Maximum </b>} {{} $maxTFX {}} {{} $maxTFY {}} {{} $maxTFZ {}} {{} $maxTFSUM {}}] \
         ]]
      return
   }

   # Thermal Temperatures
   proc tableId13 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,temp } err
      set minTemp [format "%12.4g" [ans_getvalue sort,,min]]
      set maxTemp [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {{} Temperature {}}] \
         [list {<b> Minimum </b>} {{} $minTemp {}}] \
         [list {<b> Maximum </b>} {{} $maxTemp {}}] \
         ]]
      return
   }

   # Natural Frequencies
   proc tableId14 { caption } {

      ::apdl::noprint 1
      set numModes [ans_getvalue common,,engcom,,int,10003]
      set numModes [expr $numModes > 3?3:$numModes]
      for {set i 0} {$i < $numModes} {incr i} {
         set mode($i) [ans_getvalue mode,$i,freq]
         lappend freqs [format "%12.4g" [ans_getvalue mode,$i,freq]]
      }
      switch -- $numModes {
         1 {
            set title [list {<b> 1st </b>}]
         }
         2 {
            set title [list {<b> 1st </b>} {<b> 2nd </b>}]
         }
         3 {
            set title [list {<b> 1st </b>} {<b> 2nd </b>} {<b> 3rd </b>}]
         }
      }
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         $title \
         $freqs
         ]]
      return
   }

   # Rotation
   proc tableId15 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,rot,x,,1,1 } err
      set minUX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxUX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,rot,y,,1,1 } err
      set minUY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxUY [format "%12.4g" [ans_getvalue sort,,max]]
      if {[ans_getvalue active,,dof,max] > 2} {
         catch {ans_sendcommand )nsort,rot,z,,1,1 } err
         set minUZ [format "%12.4g" [ans_getvalue sort,,min]]
         set maxUZ [format "%12.4g" [ans_getvalue sort,,max]]
      } else {
         set minUZ [format "%12.4g" 0.0]
         set maxUZ [format "%12.4g" 0.0]
      }
      catch {ans_sendcommand )nsort,rot,sum,,,1 } err
      set minUSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxUSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Maximum </b>} {{} $maxUX {}} {{} $maxUY {}} {{} $maxUZ {}} {{} $maxUSUM {}}] \
         ]]

      return
   }

   # Temperature
   proc tableId16 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,ntemp,,,1,1 } err
      set minNTEMP [format "%12.4g" [ans_getvalue sort,,min]]
      set maxNTEMP [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Temperature" </b>}] \
         [list {<b> Maximum </b>} {{} $maxNTEMP {}}] \
         ]]

      return
   }

   # Pressure
   proc tableId17 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,pres,,,1,1 } err
      set minPRES [format "%12.4g" [ans_getvalue sort,,min]]
      set maxPRES [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Pressure" </b>}] \
         [list {<b> Maximum </b>} {{} $maxPRES {}}] \
         ]]

      return
   }

   # Electric Potential
   proc tableId18 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,volt,,,1,1 } err
      set minVOLT [format "%12.4g" [ans_getvalue sort,,min]]
      set maxVOLT [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Electric Potential" </b>}] \
         [list {<b> Maximum </b>} {{} $maxVOLT {}}] \
         ]]

      return
   }

   # Fluid Velocity
   proc tableId19 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,v,x,,1,1 } err
      set minVX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxVX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,v,y,,1,1 } err
      set minVY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxVY [format "%12.4g" [ans_getvalue sort,,max]]
      if {[ans_getvalue active,,dof,max] > 2} {
         catch {ans_sendcommand )nsort,v,z,,1,1 } err
         set minVZ [format "%12.4g" [ans_getvalue sort,,min]]
         set maxVZ [format "%12.4g" [ans_getvalue sort,,max]]
      } else {
         set minVZ [format "%12.4g" 0.0]
         set maxVZ [format "%12.4g" 0.0]
      }
      catch {ans_sendcommand )nsort,v,sum,,,1 } err
      set minVSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxVSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Maximum </b>} {{} $maxVX {}} {{} $maxVY {}} {{} $maxVZ {}} {{} $maxVSUM {}}] \
         ]]

      return
   }

   # Current
   proc tableId20 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,curr,,,1,1 } err
      set minCURR [format "%12.4g" [ans_getvalue sort,,min]]
      set maxCURR [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Current" </b>}] \
         [list {<b> Maximum </b>} {{} $maxCURR {}}] \
         ]]

      return
   }

   # Electromotive force drop
   proc tableId21 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,emf,,,1,1 } err
      set minEMF [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEMF [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Electromotive Force Drop" </b>}] \
         [list {<b> Maximum </b>} {{} $maxEMF {}}] \
         ]]

      return
   }

   # Turbulent Kinetic Energy
   proc tableId22 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,enke,,,1,1 } err
      set minENKE [format "%12.4g" [ans_getvalue sort,,min]]
      set maxENKE [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Turbulent Kinetic Energy" </b>}] \
         [list {<b> Maximum </b>} {{} $maxENKE {}}] \
         ]]

      return
   }

   # Turbulent Energy Dissipation
   proc tableId23 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,ntemp,,,1,1 } err
      set minENDS [format "%12.4g" [ans_getvalue sort,,min]]
      set maxENDS [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Turbulent Energy Dissipation" </b>}] \
         [list {<b> Maximum </b>} {{} $maxENDS {}}] \
         ]]

      return
   }

   # Component Total Strain
   proc tableId24 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epto,x } err
      # send the command twice to get around a bug (reloads the PG object)
      catch {ans_sendcommand )nsort,epto,x } err
      set minEPTOX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTOX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epto,y } err
      set minEPTOY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTOY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epto,z } err
      set minEPTOZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTOZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPTOX {}} {{} $minEPTOY {}} {{} $minEPTOZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPTOX {}} {{} $maxEPTOY {}} {{} $maxEPTOZ {}}] \
         ]]
      return
   }

   # Shear Total Strain
   proc tableId25 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epto,xy } err
      set minEPTOXY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTOXY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epto,yz } err
      set minEPTOYZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTOYZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epto,xz } err
      set minEPTOXZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTOXZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "XY" </b>} {<b> "YZ" </b>} {<b> "XZ" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPTOXY {}} {{} $minEPTOYZ {}} {{} $minEPTOXZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPTOXY {}} {{} $maxEPTOYZ {}} {{} $maxEPTOXZ {}}] \
         ]]
      return
   }

   # Pricipal Total Strain
   proc tableId26 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epto,1 } err
      set minEPTO1 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTO1 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epto,2 } err
      set minEPTO2 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTO2 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epto,3 } err
      set minEPTO3 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTO3 [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "1st" </b>} {<b> "2nd" </b>} {<b> "3rd" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPTO1 {}} {{} $minEPTO2 {}} {{} $minEPTO3 {}}] \
         [list {<b> Maximum </b>} {{} $maxEPTO1 {}} {{} $maxEPTO2 {}} {{} $maxEPTO3 {}}] \
         ]]
      return
   }

   # Total Strain Intensity & Total Equivalent Strain
   proc tableId27 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epto,eqv } err
      set minEPTOE [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTOE [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epto,int } err
      set minEPTOI [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTOI [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Strain Intensity" </b>} {<b> "Equivalent Strain" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPTOI {}} {{} $minEPTOE {}}] \
         [list {<b> Maximum </b>} {{} $maxEPTOI {}} {{} $maxEPTOE {}}] \
         ]]
      return
   }

   # Component Elastic Strain
   proc tableId28 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epel,x } err
      # send the command twice to get around a bug (reloads the PG object)
      catch {ans_sendcommand )nsort,epel,x } err
      set minEPELX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPELX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epel,y } err
      set minEPELY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPELY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epel,z } err
      set minEPELZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPELZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPELX {}} {{} $minEPELY {}} {{} $minEPELZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPELX {}} {{} $maxEPELY {}} {{} $maxEPELZ {}}] \
         ]]
      return
   }

   # Shear Elastic Strain
   proc tableId29 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epel,xy } err
      set minEPELXY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPELXY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epel,yz } err
      set minEPELYZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPELYZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epel,xz } err
      set minEPELXZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPELXZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "XY" </b>} {<b> "YZ" </b>} {<b> "XZ" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPELXY {}} {{} $minEPELYZ {}} {{} $minEPELXZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPELXY {}} {{} $maxEPELYZ {}} {{} $maxEPELXZ {}}] \
         ]]
      return
   }

   # Pricipal Elastic Strain
   proc tableId30 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epel,1 } err
      set minEPEL1 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPEL1 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epel,2 } err
      set minEPEL2 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPEL2 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epel,3 } err
      set minEPEL3 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPEL3 [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "1st" </b>} {<b> "2nd" </b>} {<b> "3rd" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPEL1 {}} {{} $minEPEL2 {}} {{} $minEPEL3 {}}] \
         [list {<b> Maximum </b>} {{} $maxEPEL1 {}} {{} $maxEPEL2 {}} {{} $maxEPEL3 {}}] \
         ]]
      return
   }

   # Elastic Strain Intensity & Elastic Equivalent Strain
   proc tableId31 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epel,eqv } err
      set minEPELE [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPELE [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epel,int } err
      set minEPELI [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPELI [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Strain Intensity" </b>} {<b> "Equivalent Strain" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPELI {}} {{} $minEPELE {}}] \
         [list {<b> Maximum </b>} {{} $maxEPELI {}} {{} $maxEPELE {}}] \
         ]]
      return
   }

   # Component Plastic Strain
   proc tableId32 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,eppl,x } err
      # send the command twice to get around a bug (reloads the PG object)
      catch {ans_sendcommand )nsort,eppl,x } err
      set minEPPLX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPLX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,eppl,y } err
      set minEPPLY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPLY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,eppl,z } err
      set minEPPLZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPLZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPPLX {}} {{} $minEPPLY {}} {{} $minEPPLZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPPLX {}} {{} $maxEPPLY {}} {{} $maxEPPLZ {}}] \
         ]]
      return
   }

   # Shear Plastic Strain
   proc tableId33 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,eppl,xy } err
      set minEPPLXY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPLXY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,eppl,yz } err
      set minEPPLYZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPLYZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,eppl,xz } err
      set minEPPLXZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPLXZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "XY" </b>} {<b> "YZ" </b>} {<b> "XZ" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPPLXY {}} {{} $minEPPLYZ {}} {{} $minEPPLXZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPPLXY {}} {{} $maxEPPLYZ {}} {{} $maxEPPLXZ {}}] \
         ]]
      return
   }

   # Pricipal Plastic Strain
   proc tableId34 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,eppl,1 } err
      set minEPPL1 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPL1 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,eppl,2 } err
      set minEPPL2 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPL2 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,eppl,3 } err
      set minEPPL3 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPL3 [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "1st" </b>} {<b> "2nd" </b>} {<b> "3rd" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPPL1 {}} {{} $minEPPL2 {}} {{} $minEPPL3 {}}] \
         [list {<b> Maximum </b>} {{} $maxEPPL1 {}} {{} $maxEPPL2 {}} {{} $maxEPPL3 {}}] \
         ]]
      return
   }

   # Plastic Strain Intensity & Plastic Equivalent Strain
   proc tableId35 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,eppl,eqv } err
      set minEPPLE [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPLE [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,eppl,int } err
      set minEPPLI [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPPLI [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Strain Intensity" </b>} {<b> "Equivalent Strain" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPPLI {}} {{} $minEPPLE {}}] \
         [list {<b> Maximum </b>} {{} $maxEPPLI {}} {{} $maxEPPLE {}}] \
         ]]
      return
   }

   # Component Creep Strain
   proc tableId36 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epcr,x } err
      # send the command twice to get around a bug (reloads the PG object)
      catch {ans_sendcommand )nsort,epcr,x } err
      set minEPCRX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCRX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epcr,y } err
      set minEPCRY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCRY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epcr,z } err
      set minEPCRZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCRZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPCRX {}} {{} $minEPCRY {}} {{} $minEPCRZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPCRX {}} {{} $maxEPCRY {}} {{} $maxEPCRZ {}}] \
         ]]
      return
   }

   # Shear Creep Strain
   proc tableId37 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epcr,xy } err
      set minEPCRXY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCRXY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epcr,yz } err
      set minEPCRYZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCRYZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epcr,xz } err
      set minEPCRXZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCRXZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "XY" </b>} {<b> "YZ" </b>} {<b> "XZ" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPCRXY {}} {{} $minEPCRYZ {}} {{} $minEPCRXZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPCRXY {}} {{} $maxEPCRYZ {}} {{} $maxEPCRXZ {}}] \
         ]]
      return
   }

   # Pricipal Creep Strain
   proc tableId38 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epcr,1 } err
      set minEPCR1 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCR1 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epcr,2 } err
      set minEPCR2 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCR2 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epcr,3 } err
      set minEPCR3 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCR3 [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "1st" </b>} {<b> "2nd" </b>} {<b> "3rd" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPCR1 {}} {{} $minEPCR2 {}} {{} $minEPCR3 {}}] \
         [list {<b> Maximum </b>} {{} $maxEPCR1 {}} {{} $maxEPCR2 {}} {{} $maxEPCR3 {}}] \
         ]]
      return
   }

   # Creep Strain Intensity & Creep Equivalent Strain
   proc tableId39 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epcr,eqv } err
      set minEPCRE [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCRE [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epcr,int } err
      set minEPCRI [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPCRI [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Strain Intensity" </b>} {<b> "Equivalent Strain" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPCRI {}} {{} $minEPCRE {}}] \
         [list {<b> Maximum </b>} {{} $maxEPCRI {}} {{} $maxEPCRE {}}] \
         ]]
      return
   }

   # Component Thermal Strain
   proc tableId40 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epth,x } err
      # send the command twice to get around a bug (reloads the PG object)
      catch {ans_sendcommand )nsort,epth,x } err
      set minEPTHX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTHX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epth,y } err
      set minEPTHY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTHY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epth,z } err
      set minEPTHZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTHZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPTHX {}} {{} $minEPTHY {}} {{} $minEPTHZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPTHX {}} {{} $maxEPTHY {}} {{} $maxEPTHZ {}}] \
         ]]
      return
   }

   # Shear Thermal Strain
   proc tableId41 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epth,xy } err
      set minEPTHXY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTHXY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epth,yz } err
      set minEPTHYZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTHYZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epth,xz } err
      set minEPTHXZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTHXZ [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "XY" </b>} {<b> "YZ" </b>} {<b> "XZ" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPTHXY {}} {{} $minEPTHYZ {}} {{} $minEPTHXZ {}}] \
         [list {<b> Maximum </b>} {{} $maxEPTHXY {}} {{} $maxEPTHYZ {}} {{} $maxEPTHXZ {}}] \
         ]]
      return
   }

   # Principal Thermal Strain
   proc tableId42 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epth,1 } err
      set minEPTH1 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTH1 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epth,2 } err
      set minEPTH2 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTH2 [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epth,3 } err
      set minEPTH3 [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTH3 [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "1st" </b>} {<b> "2nd" </b>} {<b> "3rd" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPTH1 {}} {{} $minEPTH2 {}} {{} $minEPTH3 {}}] \
         [list {<b> Maximum </b>} {{} $maxEPTH1 {}} {{} $maxEPTH2 {}} {{} $maxEPTH3 {}}] \
         ]]
      return
   }

   # Thermal Strain Intensity & Thermal Equivalent Strain
   proc tableId43 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,epth,eqv } err
      set minEPTHE [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTHE [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,epth,int } err
      set minEPTHI [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEPTHI [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "Strain Intensity" </b>} {<b> "Equivalent Strain" </b>}] \
         [list {<b> Minimum </b>} {{} $minEPTHI {}} {{} $minEPTHE {}}] \
         [list {<b> Maximum </b>} {{} $maxEPTHI {}} {{} $maxEPTHE {}}] \
         ]]
      return
   }

   # Component Pressure Gradient & Sum
   proc tableId44 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,pg,x } err
      set minPGX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxPGX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,pg,y } err
      set minPGY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxPGY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,pg,z } err
      set minPGZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxPGZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,pg,sum } err
      set minPGSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxPGSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Minimum </b>} {{} $minPGX {}} {{} $minPGY {}} {{} $minPGZ {}} {{} $minPGSUM {}}] \
         [list {<b> Maximum </b>} {{} $maxPGX {}} {{} $maxPGY {}} {{} $maxPGZ {}} {{} $maxPGSUM {}}] \
         ]]
      return
   }

   # Component Electric Field & Sum
   proc tableId45 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,ef,x } err
      set minEFX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEFX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,ef,y } err
      set minEFY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEFY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,ef,z } err
      set minEFZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEFZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,ef,sum } err
      set minEFSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxEFSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Minimum </b>} {{} $minEFX {}} {{} $minEFY {}} {{} $minEFZ {}} {{} $minEFSUM {}}] \
         [list {<b> Maximum </b>} {{} $maxEFX {}} {{} $maxEFY {}} {{} $maxEFZ {}} {{} $maxEFSUM {}}] \
         ]]
      return
   }

   # Component Electric Flux Density & Sum
   proc tableId46 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,d,x } err
      set minDX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxDX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,d,y } err
      set minDY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxDY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,d,z } err
      set minDZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxDZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,d,sum } err
      set minDSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxDSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Minimum </b>} {{} $minDX {}} {{} $minDY {}} {{} $minDZ {}} {{} $minDSUM {}}] \
         [list {<b> Maximum </b>} {{} $maxDX {}} {{} $maxDY {}} {{} $maxDZ {}} {{} $maxDSUM {}}] \
         ]]
      return
   }

   # Component Magnetic Field Intensity & Sum
   proc tableId47 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,h,x } err
      set minHX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxHX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,h,y } err
      set minHY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxHY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,h,z } err
      set minHZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxHZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,h,sum } err
      set minHSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxHSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Minimum </b>} {{} $minHX {}} {{} $minHY {}} {{} $minHZ {}} {{} $minHSUM {}}] \
         [list {<b> Maximum </b>} {{} $maxHX {}} {{} $maxHY {}} {{} $maxHZ {}} {{} $maxHSUM {}}] \
         ]]
      return
   }

   # Component Magnetic Flux Density & Sum
   proc tableId48 { caption } {

      ::apdl::noprint 1
      catch {ans_sendcommand )nsort,b,x } err
      set minBX [format "%12.4g" [ans_getvalue sort,,min]]
      set maxBX [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,b,y } err
      set minBY [format "%12.4g" [ans_getvalue sort,,min]]
      set maxBY [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,b,z } err
      set minBZ [format "%12.4g" [ans_getvalue sort,,min]]
      set maxBZ [format "%12.4g" [ans_getvalue sort,,max]]
      catch {ans_sendcommand )nsort,b,sum } err
      set minBSUM [format "%12.4g" [ans_getvalue sort,,min]]
      set maxBSUM [format "%12.4g" [ans_getvalue sort,,max]]
      ::apdl::noprint 0

      genHTMLTable $caption [subst [list \
         [list {} {<b> "X" </b>} {<b> "Y" </b>} {<b> "Z" </b>} {<b> "Vector Sum" </b>}] \
         [list {<b> Minimum </b>} {{} $minBX {}} {{} $minBY {}} {{} $minBZ {}} {{} $minBSUM {}}] \
         [list {<b> Maximum </b>} {{} $maxBX {}} {{} $maxBY {}} {{} $maxBZ {}} {{} $maxBSUM {}}] \
         ]]
      return
   }

   proc tableId0 { caption tblWidth tblHeight } {
      variable ReportArray
      variable CustomTableData

      for {set i 0} {$i < $tblHeight} {incr i} {
         for {set j 0} {$j < $tblWidth} {incr j} {
            if {[string match -nocase "\*get,*" "$CustomTableData($i,$j)"]} {
               set getString [split $CustomTableData($i,$j) ,]
               set getString [lrange $getString 2 end]
               set getParm [join $getString ,]
               if {[catch {string trim [ans_getvalue $getParm]} getVal]} {
                  set getVal [expr pow(2,-100)]
               }
               if {[info exists CustomTableData($i,$j,tags)]} {
                  lappend row($i) [list \
                        [lindex $CustomTableData($i,$j,tags) 0] \
                        [format "%12.4g" $getVal] \
                        [lindex $CustomTableData($i,$j,tags) 1]]
               } else {
                  lappend row($i) [list {} [format "%12.4g" $getVal] {}]
               }
            } else {
               if {[info exists CustomTableData($i,$j,tags)]} {
                  lappend row($i) [list \
                        [lindex $CustomTableData($i,$j,tags) 0] \
                        [::apdl::substitute $CustomTableData($i,$j)] \
                        [lindex $CustomTableData($i,$j,tags) 1]]
               } else {
                  lappend row($i) [list {} \
                        [::apdl::substitute $CustomTableData($i,$j)] {}]
               }
            }
         }
         lappend tableData $row($i)
      }
      genHTMLTable $caption $tableData
  } 


   proc genHTMLTable { caption tableData {rightJ 1}} {
      variable ReportArray

      set fileID [open [file join $ReportArray(reportDir) ansysTables.js] a]

      if {$ReportArray(curTableNumber) == 1} {
         puts $fileID "// JavaScript functions to access ANSYS tables\
               for report"
         if {[ans_getfilter 0 VERIFY]} {
            puts $fileID "// \"$ReportArray(reportName)\" created on DATE "
         } else {
            puts $fileID "// \"$ReportArray(reportName)\" created on [clock format [clock seconds]] "
         }
      }

      puts $fileID "function table_$ReportArray(curTableNumber) (name) {"
	   puts $fileID "   var undefined;"
      puts $fileID "   // jobname = '$ReportArray(jobname)'"
      puts $fileID "   document.writeln('<table>');"
      puts $fileID "   if (name == undefined) {"
      puts $fileID "      document.writeln('<tr><td><B>$caption</B></td></tr>');"
      puts $fileID "   } else {"
      puts $fileID "      document.writeln('<tr><B>' + name + '</B></tr>');"
      puts $fileID "   }"

      puts $fileID "   document.writeln('<tr>');"
      puts $fileID "   document.writeln('<td>');"

      # Create a table in the second cell of the enclosing table
      set cols [llength [lindex $tableData 0]]
      # nL = number of lines
      set nL 0
      set htmlLine([incr nL]) \
            "   document.writeln('<table border=1 cols=$cols cellpadding=7 cellspacing=0 \$widthVar>');"
      catch {unset widthN}
      foreach row $tableData {
         if { $cols != [llength $row] } {
            set cols [llength $row]
            if {[llength $row] == 1 && [string match {{}} $row]} {
               set htmlLine([incr nL]) "   document.writeln('</table>');"
               set htmlLine([incr nL]) "   document.writeln('<BR>');"
               set htmlLine([incr nL]) "   document.writeln('</td>');"
               set htmlLine([incr nL]) "   document.writeln('</tr>');"
            } else {
               set htmlLine([incr nL]) "   document.writeln('<tr>');"
               set htmlLine([incr nL]) "   document.writeln('<td>');"
               set htmlLine([incr nL])  \
            "   document.writeln('<table border=1 cols=$cols cellpadding=7 cellspacing=0 \$widthVar>');"
            }
         }
         set htmlLine([incr nL]) "   document.writeln('<tr>');"
         set i 0

         foreach cell $row {
            set prF [lindex $cell 0]
            set d [lindex $cell 1]
            set poF [lindex $cell 2]
            if {$i == 0 && $rightJ} {
               set htmlLine([incr nL])  \
                  "   document.writeln('<td width=\"\$perWid\">$prF$d$poF</td>');"
            } else {
               set htmlLine([incr nL])  \
                  "   document.writeln('<td align=center>$prF$d$poF</td>');"
            }
            lappend widthN($i) [expr [string length $d] * 10]
            incr i
         }
         set htmlLine([incr nL])  \
               "   document.writeln('</tr>');"
      }

      set nWidth 0
      for {set j 0} {$j < $i} {incr j} {
         set cellW [lindex [lsort -integer -decreasing $widthN($j)] 0]
         if {$cellW < 120} {
            set cellW 120
         }
         set nWidth [expr $nWidth + $cellW]
      }

      set cellW [lindex [lsort -integer -decreasing $widthN(0)] 0]
      set perWid [expr int((1-(($nWidth-${cellW})/${nWidth}.0))*100)]
      set perWid "${perWid}%"
      if {$nWidth > 650} {
         set nWidth 650
      }
      set widthVar "width=$nWidth"

      for {set j 1} {$j <= $nL} {incr j} {
         puts $fileID [subst $htmlLine($j)]
      }

      puts $fileID "   document.writeln('</table>');"

      puts $fileID "   document.writeln('</td>');"
      puts $fileID "   document.writeln('</tr>');"
      puts $fileID "   document.writeln('</table>');"

      puts $fileID "// The following lines are tab delimited and can be pasted in tables that"
      puts $fileID "// recognize tab delimited text as a table. The table should be one more"
      puts $fileID "// than needed to allow deletion of the first column to remove the //."
      puts $fileID ""
      puts $fileID [format "//\t%s" $caption]
      foreach row $tableData {
         puts -nonewline $fileID [format "//"]
         foreach cell $row {
            set d [lindex $cell 1]
            puts -nonewline $fileID [format "\t%s" $d]
         }
         puts $fileID ""
      }
      puts $fileID "}"
      close $fileID
   }

   # END OF TABLE PROCS }

   # START OF IMAGE PROCS {

   # --------------------------------------------------------------------------
   # GLOBAL PROCEDURE:  imageoverlay ansysPlotCommand
   #
   # Purpose:        This procedure is called to allow for the overlaying
   #                 of images and must be used instead of /NOERASE.
   #
   # Arguments:      
   #                 ansysPlotCommand
   #                    The first plot that will begin the overlaying of images.
   #
   # Return Value:   N/A
   #
   # Comments:       N/A
   #
   # See Also:       N/A
   #
   # --------------------------------------------------------------------------
   proc imageoverlay { ansPlotCmd {imgFormat PNG} } {
      variable ReportArray

      if {[string match {} $ReportArray(reportDir)]} {
         return -code error \
               -errorinfo "The report directory must be set with setdirectory" \
               "The report directory must be set with setdirectory"
      }

      set ReportArray(imageSeq) 1

      if {!$ReportArray(menu) || $ReportArray(batchCapture)} {
         setShowIMG 1 $imgFormat
      }
#      sizeGraphics down
      # A plot must be done to initialize the file
      catch {ans_sendcommand )$ansPlotCmd} err
      catch {ans_sendcommand )/NOERASE} err

   }

   # --------------------------------------------------------------------------
   # GLOBAL PROCEDURE:  imagecapture caption
   #
   # Purpose:        This procedure is called to capture an image of the
   #                 ANSYS plot.
   #
   # Arguments:      
   #                 caption
   #                    The caption to show below this image.
   #
   # Return Value:   N/A
   #
   # Comments:       N/A
   #
   # See Also:       N/A
   #
   # --------------------------------------------------------------------------
   proc imagecapture { caption {imgFormat PNG} } {
      variable ReportArray
      global tcl_platform

      update
      if {[string match {} $ReportArray(reportDir)]} {
         return -code error \
               -errorinfo "The report directory must be set with setdirectory" \
               "The report directory must be set with setdirectory"
      }

      # Check the image format for validity
      switch -- [string toupper $imgFormat] {
         JPEG -
         PNG {
         }
         default {
            return -code error \
               -errorinfo "The image format argument must be JPEG or PNG" \
                  "The image format argument must be JPEG or PNG" 
         }
      }

      # Set up the images directory for this report
      if {![file exists [file join $ReportArray(reportDir) images]]} {
         file mkdir [file join $ReportArray(reportDir) images]
      }
      set ReportArray(imageDir) [file join $ReportArray(reportDir) images]

      incr ReportArray(curImageNumber)

      set caption [::apdl::substitute $caption]
      # Get the current jobname from ANSYS it may have changed
      set jobname [getReportInfo jobname]

      set imageAdded 0

      # Generate the image
      if {$ReportArray(menu) && !$ReportArray(batchCapture)} {
         catch {ans_sendcommand )/UI,RAISE} err
         update
         # If the graphics window is on then we do a WYSIWYG
         if {![ans_getvalue graph,,erase] \
               && [uplevel #0 info exists ans_guiMode] 
               && !$ReportArray(imageSeq)} {
            tk_messageBox -type ok \
               -message "If the intent is to use /NOERASE to capture an overlay\
               of images in a batch run then use the GPLOT capability of \
               ANSYS or ansys::report::imageoverlay must be used."
            update idletasks
         }
         ::apdl::noprint 1
         catch {ans_sendcommand )/DEV,PSFN,NINC} err
         set imgFormat [string toupper $imgFormat]
         catch {ans_sendcommand )/UI,COPY,SAVE,$imgFormat,GRAPH,COLOR,NORMAL,PORTRAIT,ON,-1} err
         catch {ans_sendcommand )/DEV,PSFN} err
         set imgFormat [string tolower $imgFormat]
         ::apdl::noprint 0
         if {[file exists ${jobname}.$imgFormat]} {
            file copy -force ${jobname}.$imgFormat [file join $ReportArray(imageDir) \
                  image${ReportArray(curImageNumber)}.$imgFormat]
            file delete -force ${jobname}.$imgFormat
            set imageAdded 1
         }
         if {$ReportArray(imageSeq)} {
            set ReportArray(imageSeq) 0
            catch {ans_sendcommand )/ERASE} err
         }

         # On UNIX systems, the cursor does not change back from the
         # hourglass.  This was added to force refreshing of the graphics
         # window and updating the cursor back to the normal pointer.
         if { ![string match windows $tcl_platform(platform)] } {
            ::uidl::refreshCursor
         }

#         sizeGraphics up
      } else {
         # This is likely a batch run
         if {![ans_getvalue graph,,erase] && !$ReportArray(imageSeq)} {
            ans_senderror 1 "The image capture will use /REPLOT to capture the image of the last plotted image and does not include any previous plots that used /NOERASE, also the /ERASE command will be issued turning off /NOERASE"
         }
         # If the GUI is off then we can't do a WYSIWYG
         ::apdl::noprint 1
         set imgFormat [string toupper $imgFormat]
         if {!$ReportArray(imageSeq)} {
            setShowIMG 1 $imgFormat
            catch {ans_sendcommand )/REPLOT} err
#            sizeGraphics up
         } else {
            set ReportArray(imageSeq) 0
            catch {ans_sendcommand )/ERASE} err
         }
         setShowIMG 0 $imgFormat
         ::apdl::noprint 0

         set imgFormat [string tolower $imgFormat]
         if {[file exists ${jobname}.$imgFormat]} {
            file copy -force ${jobname}.$imgFormat [file join $ReportArray(imageDir) \
                  image${ReportArray(curImageNumber)}.$imgFormat]
            file delete -force ${jobname}.$imgFormat
            set imageAdded 1
         }
      }

      if {$imageAdded} {
         switch -- $ReportArray(format) {
            javascript {
# Append JavaScript information to the end of ansysImages.js
set fileID [open [file join $ReportArray(reportDir) ansysImages.js] a]
if {$ReportArray(curImageNumber) == 1} {
   puts $fileID "// JavaScript functions to access ANSYS images\
      for report"
   if {[ans_getfilter 0 VERIFY]} {
      puts $fileID "// \"$ReportArray(reportName)\" created on DATE "
   } else {
      puts $fileID "// \"$ReportArray(reportName)\" created on \
         [clock format [clock seconds]] "
   }
}
puts $fileID "function image_$ReportArray(curImageNumber) (imgTitle) \{"
puts $fileID "   var undefined;"
puts $fileID "   // jobname = '$ReportArray(jobname)'"
puts $fileID "   if (imgTitle == undefined) {"
puts $fileID "      imgTitle = '$caption';"
puts $fileID "   }"
puts $fileID "   document.writeln('<TABLE COLS=1 WIDTH=\"$ReportArray(width)\"><TR>');"
puts $fileID "   document.writeln('<TD VALIGN=TOP>');"
puts $fileID "   document.writeln('<IMG SRC=\"images/image$ReportArray(curImageNumber).png\">');"
puts $fileID "   document.writeln('</TD>');"
puts $fileID "   document.writeln('</TR>');"
puts $fileID "   document.writeln('<TR><TD VALIGN=TOP>');"
puts $fileID "   document.writeln('<B>' + imgTitle + '</B>');"
puts $fileID "   document.writeln('</TD></TR></TABLE>');"
puts $fileID "\}"
close $fileID
            }
            html {
            }
            ds {
               if {[catch {getObjects} err]} {
                  # DS is not present
               } else {
                  # Insert a post output object for the image
                  ::AnsTComClient::eval \
                     {set imgObject [$dsPost_obj AddPostOutput]}
                  #
                  ::AnsTComClient::eval {for {set intOne 1} {$intOne <= 0} {incr i} {}}
                  ::AnsTComClient::eval {$imgObject SourceType $intOne}
                  #
                  #
                  ::AnsTComClient::eval [list set caption $caption]
                  ::AnsTComClient::eval {$imgObject Name $caption}
                  #
                  ::AnsTComClient::eval [list regsub -all {\\} [file nativename \
                        [file join $ReportArray(imageDir) \
                              image${ReportArray(curImageNumber)}.$imgFormat]] {\\\\} imgPath]
                  ::AnsTComClient::eval {$imgObject ImagePath $imgPath}
               }
            }
         }

      }
   }

   proc setShowIMG { key {format PNG} } {
      # This is to only be called in a non-graphics mode (should only be batch)
      variable ReportArray
      variable ShowArray
      switch -- $key {
         0 {
            switch -- $ShowArray(active) {
               -1 -
               0 {
                  # No /SHOW command had been issued
                  catch {ans_sendcommand )/SHOW,CLOSE} err
#                  catch {ans_sendcommand )/SHOW,OFF} err
                  if { [ans_getfilter 0 aao_btch] } {
                     catch {ans_sendcommand )/SHOW,report,grph,,8} err
                  }
                  catch {ans_sendcommand )/DEV,PSFN} err
                  catch {ans_sendcommand )/GFILE} err
               }
               1 {
                  catch {ans_sendcommand )/SHOW,CLOSE} err
#                  catch {ans_sendcommand )/SHOW,OFF} err
                  if { [ans_getfilter 0 aao_btch] } {
                     catch {ans_sendcommand )/SHOW,report,grph,,8} err
                  }
                  # The user was plotting out to a file
                  # Issue the /SHOW command from the stored data
                  #catch {ans_sendcommand /SHOW,$ShowArray(filename),\
                  #      $ShowArray(fileext),$ShowArray(mode),\
                  #      $ShowArray(planes)} err
                  catch {ans_sendcommand )/DEV,PSFN} err
                  catch {ans_sendcommand )/GFILE} err
               }
               5 {
                  catch {ans_sendcommand )/SHOW,CLOSE} err
                  # X11, X11c, WIN32, or WIN32c 
                  # Issue the /SHOW command from the stored data
                  catch {ans_sendcommand )/SHOW,,$ShowArray(mode),\
                        $ShowArray(planes)} err
                  catch {ans_sendcommand )/DEV,PSFN} err
                  catch {ans_sendcommand )/GFILE} err
               }
            }
         }
         1 {
            # Need to capture the current status of /SHOW command and store
            # Appears the only way to get filename and ext is by using /PSTATUS
            # and parsing out the data.
            #set ShowArray(filename)
            #set ShowArray(ext)
            set numContours [ans_getvalue common,,d3com,,int,78]
            set ShowArray(active) [ans_getvalue common,,d2com,,int,7]; #ngrtyp
            set ShowArray(mode) [ans_getvalue common,,d3com,,int,21]; #kras3d
            set ShowArray(planes) [ans_getvalue common,,d2com,,int,12]; # npln2d
            set ShowArray(active) 0
            # Close any currently open file
            catch {ans_sendcommand )/SHOW,CLOSE} err
            # Turn off number incrementing
            catch {ans_sendcommand )/DEV,PSFN,NINC} err
            # Set the pixel size to get an image of prescribed size
            catch {ans_sendcommand )/GFILE,$ReportArray(imgSize)} err
            catch {ans_sendcommand )/DEV,FONT,1,courier,medium,r,8} err
            # Reverse Video
            if {$ReportArray(reverseVideo)} {
               ::apdl::noprint 1
               catch {ans_sendcommand )/RGB,INDEX,100,100,100, 0} err
               catch {ans_sendcommand )/RGB,INDEX,  0,  0,  0,15} err
               catch {ans_sendcommand )/COLOR,PBAK,OFF} err
               ::apdl::noprint 0
            }         
            catch {ans_sendcommand )/CONT,1,$numContours,AUTO} err
            catch {ans_sendcommand )/SHOW,$format,,,8} err
         }
      }
   }
   # END OF IMAGE PROCS }

   # START OF ANIMATION PROCS {

   # --------------------------------------------------------------------------
   # GLOBAL PROCEDURE:  animcapture caption ?animSize 100?
   #
   # Purpose:        This procedure is called to capture an animation sequence
   #                 of the ansys animation commands.
   #
   # Arguments:      
   #                 caption
   #                    The caption to show below this image.
   #                 animSize
   #                    The percentage that the animation should be of the
   #                    still image size.  The default is 100 percent.  This
   #                    must be a value between 25 and 100.
   #
   # Return Value:   N/A
   #
   # Comments:       N/A
   #
   # See Also:       N/A
   #
   # --------------------------------------------------------------------------
   proc animcapture { caption {animSize 100} } {
      global env
      global ansysLanguage
      variable ReportArray
      variable AnimArray

      set AnimArray(seqCaptured) 0

      update
      if {[string match {} $ReportArray(reportDir)]} {
         return -code error \
               -errorinfo "The report directory must be set with setdirectory" \
               "The report directory must be set with setdirectory"
      }

      if { $ReportArray(menu) && !$ReportArray(batchCapture) } {
         catch {ans_sendcommand )/UI,RAISE} err
      }

      incr ReportArray(curAnimNumber)
      
      set ReportArray(animSize) [expr $animSize > 100 ? 100 : $animSize]
      set ReportArray(animSize) [expr $animSize < 25 ? 25 : $animSize]
      set ReportArray(animSize) [expr $animSize / 100.0]

      set AnimArray(curFrame) 0

      # Get the current jobname from ANSYS it may have changed
      set jobname [getReportInfo jobname]

      # Set up the animation button image directory
      if {![file exists [file join $ReportArray(reportDir) \
            anim_images]] } {
         set imgDir [file join $env(EUIDL_DIR) gui $ansysLanguage images]
         set tgtDir [file join $ReportArray(reportDir) anim_images]
         file mkdir $tgtDir

         catch {file copy -force [file join $imgDir playbtnup34x18.png] \
            [file join $tgtDir playbtnup34x18.png]}
         catch {file copy -force [file join $imgDir playbtndown34x18.png] \
            [file join $tgtDir playbtndown34x18.png]}
         catch {file copy -force [file join $imgDir playbtnactive34x18.png] \
            [file join $tgtDir playbtnactive34x18.png]}
         catch {file copy -force [file join $imgDir pausebtnup34x18.png] \
            [file join $tgtDir pausebtnup34x18.png]}
         catch {file copy -force [file join $imgDir pausebtndown34x18.png] \
            [file join $tgtDir pausebtndown34x18.png]}
         catch {file copy -force [file join $imgDir pausebtnactive34x18.png] \
            [file join $tgtDir pausebtnactive34x18.png]}
      }

      # Set up the animation directory for this report
      if {![file exists [file join $ReportArray(reportDir) \
            animseq_$ReportArray(curAnimNumber)]]} {
         file mkdir [file join $ReportArray(reportDir) \
            animseq_$ReportArray(curAnimNumber)]
      }
      set AnimArray(animDir) [file join $ReportArray(reportDir) \
            animseq_$ReportArray(curAnimNumber)]

      set caption [::apdl::substitute $caption]

      set AnimArray(caption) $caption
      set AnimArray(seqNum) $ReportArray(curAnimNumber)

      uplevel #0 trace variable ansys::report::AnimDone w [list [namespace code animWrite]]

      # Turn on JavaScript for the animation macros
      ::apdl::noprint 1
      catch {ans_sendcommand )_JAVASCR=1} err
      ::apdl::noprint 0
      # Turn noerase off if it is on
      if {![ans_getvalue graph,,erase]} {
         catch {ans_sendcommand )/ERASE} err
      }
      set ReportArray(animActive) 1
      sizeGraphics down
   }

   proc animImage {} {
      variable ReportArray
      variable AnimArray

      catch {ans_sendcommand )/DEV,PSFN,NINC} err
      if {$ReportArray(menu) && !$ReportArray(batchCapture)} {
         catch {ans_sendcommand )/REPLOT} err
         catch {ans_sendcommand )/UI,COPY,SAVE,PNG,GRAPH,COLOR,NORM,PORTRAIT,ON,1} err
      } else {
         setShowIMG 1
         catch {ans_sendcommand )/REPLOT} err
         setShowIMG 0
#         catch {ans_sendcommand )/GFILE} err
      }
      catch {ans_sendcommand )/DEV,PSFN} err
      if {[file exists ${ReportArray(jobname)}.png]} {
         file copy -force ${ReportArray(jobname)}.png \
               [file join $AnimArray(animDir) frame${AnimArray(curFrame)}.png]
         file delete -force ${ReportArray(jobname)}.png
         incr AnimArray(curFrame)
         set AnimArray(seqCaptured) 1
      } else {
         set AnimArray(seqCaptured) 0
      }

   }

   proc animWrite { args } {
      variable ReportArray
      variable AnimArray
      variable AnimDone

      sizeGraphics up
      uplevel #0 trace vdelete \
            ansys::report::AnimDone w [list [namespace code animWrite]]

      set ReportArray(animActive) 0

      ::apdl::noprint 1
      catch {ans_sendcommand )_JAVASCR=} err
      ::apdl::noprint 0

      if { $AnimDone && $AnimArray(seqCaptured) } {

         if {$AnimArray(seqNum) == 1} {
            ansysAnimationFile
         }
   
         set fileID [open [file join $ReportArray(reportDir) ansysAnimations.js] a]
         if {$AnimArray(seqNum) == 1} {
            puts $fileID "// START OF ANIMATION FUNCTIONS  "
         }
         puts $fileID ""
         puts $fileID "function animseq_$AnimArray(seqNum) (animTitle,animTime, animDirect) {"
         puts $fileID "   var undefined;"
         puts $fileID "   // jobname = '$ReportArray(jobname)'"
         puts $fileID "   if (animTitle == undefined) {"
         puts $fileID "      animTitle = '$AnimArray(caption)';"
         puts $fileID "   }"
         puts $fileID "   if (animTime == undefined) {"
         puts $fileID "      animTime = 500;"
         puts $fileID "   }"
         puts $fileID "   if (animDirect == undefined) {"
         puts $fileID "      animDirect = 'back';"
         puts $fileID "   }"
         puts -nonewline $fileID "   animseq$AnimArray(seqNum) = new SlideShow("
         puts -nonewline $fileID "'animseq_$AnimArray(seqNum)/',"
         puts -nonewline $fileID "'animseq$AnimArray(seqNum)',"
         puts -nonewline $fileID "animTitle,$AnimArray(width),$AnimArray(height),"
         puts $fileID "$AnimArray(curFrame),animTime,animDirect);"
         puts $fileID "}"
         close $fileID
      } else {
         incr ReportArray(curAnimNumber) -1
      }
   }

   proc ansysAnimationFile {} {
      variable ReportArray

      set fileID [open [file join $ReportArray(reportDir) ansysAnimations.js] w]
      puts $fileID "// JavaScript functions to access ANSYS animations\
               for report"
      if {[ans_getfilter 0 VERIFY]} {
         puts $fileID "// \"$ReportArray(reportName)\" created on DATE "
      } else {
         puts $fileID "// \"$ReportArray(reportName)\" created on [clock format [clock seconds]] "
      }
      puts $fileID ""
      puts $fileID "// The SlideShow constructor"
      puts $fileID "// function SlideShow(animPath, animName, animTitle, animWidth, animHeight, animFrames, animTime, animDirect)"
      puts $fileID "//   animPath    -  This is the path relative to the location of this "
      puts $fileID "//                  document it"
      puts $fileID "//                  is a string and so must be wrapped with single quotes."
      puts $fileID "//   animName    -  The object name is passed in as a string and so must be "
      puts $fileID "//                  wrapped with single quotes."
      puts $fileID "//   animTitle   -  The title to show for this animimation sequence. This"
      puts $fileID "//                  is a string and so must be wrapped with single quotes."
      puts $fileID "//                  It can include HTML tags around the text as well."
      puts $fileID "//   animWidth   -  The width of the figures used for the animation."
      puts $fileID "//   animHeight  -  The height of the figures used for the animation."
      puts $fileID "//   animFrames  -  The number of animation sequences to show."
      puts $fileID "//   animTime    -  The time delay (mili-seconds) between displays of animation "
      puts $fileID "//                  frames.  This value is limited by machine performance and "
      puts $fileID "//                  the slide show will wait for every slide to show regardless "
      puts $fileID "//                  of how small this value is set to."
      puts $fileID "//   animDirect  -  The direction of play:"
      puts $fileID "//                     'forward'  - When the last frame is played continues"
      puts $fileID "//                                  to the first frame and increments."
      puts $fileID "//                     'back'     - When the last frame is played continues"
      puts $fileID "//                                  to the previous frame and decrements until"
      puts $fileID "//                                  the first frame is played then increments"
      puts $fileID "//                                  again."
      puts $fileID "//"
      puts $fileID "function SlideShow(animPath, animName, animTitle, animWidth, animHeight, animFrames, animTime, animDirect)"
      puts $fileID "{"
      puts $fileID "   // Initialize object properties"
      puts $fileID ""
      puts $fileID "   // Set the status to prevent annoying flashes only works in IE though"
      puts $fileID "   // Netscape seems to ignore it for the most part."
      puts $fileID "   if (window.defaultStatus == \"\") {"
      puts $fileID "      window.defaultStatus = 'ANSYS Analysis Report';"
      puts $fileID "   }"
      puts $fileID ""
      puts $fileID "   // Set browser-determined global variables"
      puts $fileID "   this.ns4 = (document.layers ? true : false);"
      puts $fileID "   this.ie4 = (document.all ? true : false);"
      puts $fileID "   if (this.ns4) {"
      puts $fileID "      //this.print = nsPrint;"
      puts $fileID "      // for now we will just set printing true"
      puts $fileID "      this.print = true;"
      puts $fileID "   } else {"
      puts $fileID "      this.print = true;"
      puts $fileID "   }"
      puts $fileID ""
      puts $fileID "   // These are used by the animator"
      puts $fileID "   this.animActive = 0;"
      puts $fileID "   this.animTitle = animTitle;"
      puts $fileID "   this.animFrames = animFrames;"
      puts $fileID "   this.animHeight = animHeight;"
      puts $fileID "   this.animName = animName;"
      puts $fileID "   this.animPictName = 'Pic_' + animName;"
      puts $fileID "   this.animOffset = 1;"
      puts $fileID "   this.animPath = animPath;"
      puts $fileID "   this.animWidth = animWidth;"
      puts $fileID "   this.curSlide = 0;"
      puts $fileID "   this.imgLoaded = 1;"
      puts $fileID "   if (animDirect == 'back') {"
      puts $fileID "      this.showDirection = 1;"
      puts $fileID "   } else {"
      puts $fileID "      this.showDirection = 0;"
      puts $fileID "   }"
      puts $fileID "   this.showIncrement = 1;"
      puts $fileID "   this.showSpeed = animTime;"
      puts $fileID "   this.btnWidth = 34;"
      puts $fileID "   this.btnHeight = 18;"
      puts $fileID ""
      puts $fileID "   this.playup = new Image (this.btnWidth,this.btnHeight);"
      puts $fileID "   this.playup.src = \"anim_images/playbtnup34x18.png\";"
      puts $fileID "   this.playdown = new Image (this.btnWidth,this.btnHeight);"
      puts $fileID "   this.playdown.src = \"anim_images/playbtndown34x18.png\";"
      puts $fileID "   this.playactive = new Image (this.btnWidth,this.btnHeight);"
      puts $fileID "   this.playactive.src = \"anim_images/playbtnactive34x18.png\";"
      puts $fileID "   this.playButton = 'Picbtn_' + animName;"
      puts $fileID "   this.pauseButton = 'Pausebtn_' + animName;"
      puts $fileID "   this.pauseup = new Image (this.btnWidth,this.btnHeight);"
      puts $fileID "   this.pauseup.src = \"anim_images/pausebtnup34x18.png\";"
      puts $fileID "   this.pausedown = new Image (this.btnWidth,this.btnHeight);"
      puts $fileID "   this.pausedown.src = \"anim_images/pausebtndown34x18.png\";"
      puts $fileID "   this.pauseactive = new Image (this.btnWidth,this.btnHeight);"
      puts $fileID "   this.pauseactive.src = \"anim_images/pausebtnactive34x18.png\";"
      puts $fileID ""
      puts $fileID "   var index = '';"
      puts $fileID ""
      puts $fileID "   // Set the current slide to the last animation frame"
      puts $fileID "   this.curSlide = this.animFrames - 1;"
      puts $fileID ""
      puts $fileID "   // Create the slide show region using a table this includes:"
      puts $fileID "   //    an image"
      puts $fileID "   //    a title          playButton stopButton"
      puts $fileID "   //"
      puts $fileID "   document.writeln('<TABLE COLS=2 WIDTH=\"' + this.animWidth + '\"><TR>');"
      puts $fileID "   document.writeln('<TD VALIGN=TOP COLSPAN=\"2\" \\"
      puts $fileID "      HEIGHT=\"' + this.animHeight + '\">');"
      puts $fileID "   document.writeln('<CENTER>');"
      puts $fileID "   document.writeln('<IMG NAME=\"' + this.animPictName + '\" \\"
      puts $fileID "      SRC=\"' + this.animPath + 'frame' + this.curSlide + '.png\">');"
      puts $fileID "   document.writeln('</CENTER>');"
      puts $fileID "   document.writeln('</TD>');"
      puts $fileID "   document.writeln('</TR>');"
      puts $fileID "   document.writeln('<TR><TD VALIGN=TOP>');"
      puts $fileID "   document.writeln('<B>' + this.animTitle + '</B>');"
      puts $fileID "   document.writeln('</TD>');"
      puts $fileID "   document.writeln('<TD VALIGN=TOP>');"
      puts $fileID "   document.writeln('<DIV ALIGN=RIGHT><FORM>');"
      puts $fileID "   document.writeln('<IMG SRC=\"' + this.playup.src + '\" \\"
      puts $fileID "      WIDTH=\"' + this.btnWidth + '\" \\"
      puts $fileID "      HEIGHT=\"' + this.btnHeight + '\" \\"
      puts $fileID "      BORDER=\"0\" \\"
      puts $fileID "      NAME=\"' + this.playButton + '\" \\"
      puts $fileID "      ONMOUSEDOWN=\"javascript: ' + this.animName + '.buttonPressed(1)\" \\"
      puts $fileID "      ONMOUSEUP=\"javascript: ' + this.animName + '.buttonReleased(1)\">');"
      puts $fileID "   document.writeln('<IMG SRC=\"' + this.pauseactive.src + '\" \\"
      puts $fileID "      WIDTH=\"' + this.btnWidth + '\" \\"
      puts $fileID "      HEIGHT=\"' + this.btnHeight + '\" \\"
      puts $fileID "      BORDER=\"0\" \\"
      puts $fileID "      NAME=\"' + this.pauseButton + '\" \\"
      puts $fileID "      ONMOUSEDOWN=\"javascript: ' + this.animName + '.buttonPressed(0)\" \\"
      puts $fileID "      ONMOUSEUP=\"javascript: ' + this.animName + '.buttonReleased(0)\">');"
      puts $fileID "   document.writeln('</FORM></DIV></TD></TR></TABLE>');"
      puts $fileID "   //"
      puts $fileID "}"
      puts $fileID ""
      puts $fileID "// call as - listPropertyNamesValues(obj,\"objName\");"
      puts $fileID "function listPropertyNames(obj, objName) {"
      puts $fileID "   var undefined;"
      puts $fileID "   var names = \"\";"
      puts $fileID "   var j = 0;"
      puts $fileID "   if (objName == undefined) {"
      puts $fileID "      for (var i in obj) {"
      puts $fileID "         names += i + \"\\n\";"
      puts $fileID "         if (j > 10) {"
      puts $fileID "            alert(names);"
      puts $fileID "            names = \"\";"
      puts $fileID "            j = 0;"
      puts $fileID "         }"
      puts $fileID "         j++;"
      puts $fileID "      }"
      puts $fileID "      alert(names);"
      puts $fileID "   } else {"
      puts $fileID "      for (var i in obj) {"
      puts $fileID "         names += i + \" value = \" + eval(objName + '.' + i) + \"\\n\";"
      puts $fileID "         if (j > 10) {"
      puts $fileID "            alert(names);"
      puts $fileID "            names = \"\";"
      puts $fileID "            j = 0;"
      puts $fileID "         }"
      puts $fileID "         j++;"
      puts $fileID "      }"
      puts $fileID "      alert(names);"
      puts $fileID "   }"
      puts $fileID "}"
      puts $fileID "SlideShow.prototype.listPropertyNames = listPropertyNames;"
      puts $fileID ""
      puts $fileID "// Function to change slides"
      puts $fileID "function changeSlide() {"
      puts $fileID "   if (this.imgLoaded) {"
      puts $fileID "      this.imgLoaded = 0;"
      puts $fileID "      eval ('document.' + this.playButton + '.src = \"' + this.playactive.src + '\"');"
      puts $fileID "      eval ('document.' + this.pauseButton + '.src = \"' + this.pauseup.src + '\"');"
      puts $fileID "   offset = this.animOffset;"
      puts $fileID "	// If the motion is back and forth we check to see if we need to change the"
      puts $fileID "   // direction"
      puts $fileID "   if (this.showDirection) {"
      puts $fileID "      // We want to decrement if we hit the end"
      puts $fileID "	   this.showIncrement = (this.curSlide + (offset*this.showIncrement) < 0 ? 1 : (this.curSlide + (offset*this.showIncrement) == this.animFrames ? -1 : this.showIncrement));"
      puts $fileID "      offset = this.showIncrement * offset;"
      puts $fileID "   }"
      puts $fileID "	// Calculate the next slide index number"
      puts $fileID "	this.curSlide = (this.curSlide + offset < 0 ? this.animFrames - 1 : (this.curSlide + offset == this.animFrames ? 0 : this.curSlide + offset));"
      puts $fileID ""
      puts $fileID "	// Show the next slide"
      puts $fileID "   // Netscape requires the document. is used."
      puts $fileID "   eval ('document.' + this.animPictName + '.src = \"' + this.animPath + 'frame' + this.curSlide + '.png' + '\"');"
      puts $fileID "   localObj = new Function (this.animName + '.setImgLoaded()');"
      puts $fileID "   eval ('document.' + this.animPictName + '.onload = ' + localObj);"
      puts $fileID "   } else {"
      puts $fileID "      // Do nothing just return"
      puts $fileID "   }"
      puts $fileID "}"
      puts $fileID "SlideShow.prototype.changeSlide = changeSlide;"
      puts $fileID ""
      puts $fileID "function setImgLoaded() {"
      puts $fileID "   this.imgLoaded = 1;"
      puts $fileID "}"
      puts $fileID "SlideShow.prototype.setImgLoaded = setImgLoaded;"
      puts $fileID ""
      puts $fileID ""
      puts $fileID ""
      puts $fileID "function setImgLoaded() {"
      puts $fileID "   this.imgLoaded = 1;"
      puts $fileID "}"
      puts $fileID "SlideShow.prototype.setImgLoaded = setImgLoaded;"
      puts $fileID ""
      puts $fileID "function animateSeq(anim) {"
      puts $fileID "	var undefined;"
      puts $fileID "   if (anim) {"
      puts $fileID "      if (!this.animActive) {"
      puts $fileID "         // Begin the animation"
      puts $fileID "         this.animActive = 1;"
      puts $fileID "         this.auto = setInterval(this.animName + '.changeSlide()', this.showSpeed);"
      puts $fileID "         this.imgLoaded = 1;"
      puts $fileID "      }"
      puts $fileID "   }"
      puts $fileID "   if (anim == 0) {"
      puts $fileID "      // Stop the animation"
      puts $fileID "	     if (this.animActive) {"
      puts $fileID "         this.animActive = 0;"
      puts $fileID "         clearInterval(this.auto);"
      puts $fileID "         this.imgLoaded = 1;"
      puts $fileID "      }"
      puts $fileID "   }"
      puts $fileID "}"
      puts $fileID "SlideShow.prototype.animateSeq = animateSeq;"
      puts $fileID ""
      puts $fileID "// START OF BUTTON EVENT FUNCTION"
      puts $fileID ""
      puts $fileID "function buttonPressed(val) {"
      puts $fileID "   if (val == 0) {"
      puts $fileID "      eval ('document.' + this.pauseButton + '.src = \"' + this.pausedown.src + '\"');"
      puts $fileID "   } else {"
      puts $fileID "      eval ('document.' + this.playButton + '.src = \"' + this.playdown.src + '\"');"
      puts $fileID "   }"
      puts $fileID "}"
      puts $fileID "SlideShow.prototype.buttonPressed = buttonPressed;"
      puts $fileID ""
      puts $fileID "function buttonReleased(val) {"
      puts $fileID "   if (val == 0) {"
      puts $fileID "      eval ('document.' + this.pauseButton + '.src = \"' + this.pauseactive.src + '\"');"
      puts $fileID "      eval ('document.' + this.playButton + '.src = \"' + this.playup.src + '\"');"
      puts $fileID "   } else {"
      puts $fileID "      eval ('document.' + this.playButton + '.src = \"' + this.playactive.src + '\"');"
      puts $fileID "      eval ('document.' + this.pauseButton + '.src = \"' + this.pauseup.src + '\"');"
      puts $fileID "   }"
      puts $fileID "   eval (this.animName + '.animateSeq(val)');"
      puts $fileID "}"
      puts $fileID "SlideShow.prototype.buttonReleased = buttonReleased;"

      close $fileID
   }

   # END OF ANIMATION PROCS }

   proc getReportInfo { type } {
      variable ReportArray

      switch -- $type {
         image {
            return [expr $ReportArray(curImageNumber) + 1]
         }
         anim {
            return [expr $ReportArray(curAnimNumber) + 1]
         }
         table {
            return [expr $ReportArray(curTableNumber) + 1]
         }
         listing {
            return [expr $ReportArray(curListingNumber) + 1]
         }
         grphBGimg {
            return $ReportArray(grphBGimg)
         }
         reverseVideo {
            return $ReportArray(reverseVideo)
         }
         jobname {
            set job(1) [ans_getvalue active,,jobname,,start,1]
            set job(2) [ans_getvalue active,,jobname,,start,9]
            set job(3) [ans_getvalue active,,jobname,,start,17]
            set job(4) [ans_getvalue active,,jobname,,start,25]
            set jobname [string trimright "$job(1)$job(2)$job(3)$job(4)"]
            set ReportArray(jobname) [::ansys::getJobname]
            return $ReportArray(jobname)
         }
         reportDir {
            return $ReportArray(reportDir)
         }
      }
   }

   proc append2Report {} {
      variable ReportArray

      # Determine the highest image number
      if {![catch {file rootname [lindex [lsort -dictionary [glob \
            [file join $ReportArray(reportDir) images *]]] end]} \
            lastFile]} {
         set lastFile [lindex [split $lastFile /] end]
         regexp {([0-9]+)} $lastFile match ReportArray(curImageNumber)
      }

      # Determine the highest animation number
      if {![catch {lindex [lsort -dictionary [glob \
            [file join $ReportArray(reportDir) animseq_*]]] end}\
            lastFile]} {
         set lastFile [lindex [split $lastFile /] end]
         regexp {([0-9]+)} $lastFile match ReportArray(curAnimNumber)
      }

      # Determine the highest table number
      if {[file exists [file join $ReportArray(reportDir) ansysTables.js]]} {
         set fileID [open [file join $ReportArray(reportDir) ansysTables.js] r]
         while {[gets $fileID line] >= 0} {
            regexp {function table_([0-9]+)} $line match tableNum
         }
         close $fileID
         set ReportArray(curTableNumber) $tableNum
      } else {
         set ReportArray(curTableNumber) 0
      }
      
      # Determine the highest listing number
      if {[file exists [file join $ReportArray(reportDir) ansysListings.js]]} {
         set listNum 0
         set fileID \
               [open [file join $ReportArray(reportDir) ansysListings.js] r]
         while {[gets $fileID line] >= 0} {
            regexp {function listing_([0-9]+)} $line match listNum
         }
         close $fileID
         set ReportArray(curListingNumber) $listNum
      } else {
         set ReportArray(curListingNumber) 0
      }
   }   

   proc sizeGraphics { direction } {
      variable ReportArray
      variable AnimArray
      global tcl_platform
      global tk_borderwidth
      global tk_titleheight
      global tk_posinternal

      if { [catch {ans_sendcommand )/REPLOT} err] } {
         catch {ans_sendcommand )NPLOT}
      }

      switch -- $direction {
         down {
            if {$ReportArray(menu) && !$ReportArray(batchCapture)} {
               if {$tk_posinternal} {
                  set bd $tk_borderwidth
                  set th $tk_titleheight
               } else {
                  set bd 0
                  set th 0
               }

               # Image Size
               if {!$ReportArray(sizeDownImg)} {
                  # If the window is already shrunk don't shrink it more
                  set ReportArray(winpos,x) [ans_getvalue active,,win,grph,xpos]
                  if {$ReportArray(winpos,x) > 0} {
                     set ReportArray(winpos,x) \
                           [expr $ReportArray(winpos,x)+ $bd]
                  }
                  set ReportArray(winpos,y) [ans_getvalue active,,win,grph,ypos]
                  set ReportArray(winpos,y) \
                        [expr $ReportArray(winpos,y) + $bd + $th]
                  set ReportArray(winpos,width) \
                        [ans_getvalue active,,win,grph,width]
                  set ReportArray(winpos,height) \
                        [ans_getvalue active,,win,grph,height]
                  set wpx $ReportArray(winpos,x)
                  set wpy $ReportArray(winpos,y)
                  set wpw $ReportArray(winpos,width)
                  set wph $ReportArray(winpos,height)
                  # Make sure the graphics window x and y position are positive.
                  if { $ReportArray(winpos,x) < 0 } {
                     set ReportArray(winpos,x) 0
                  }
                  if { $ReportArray(winpos,y) < 0 } {
                     set ReportArray(winpos,y) 0
                  }
                  # If the window is smaller than prescribed to fit on
                  # a page we don't change
                  if {$wpw > $ReportArray(img,width) \
                        || $wph > $ReportArray(img,height)} {
                     set wpw $ReportArray(img,width)
                     set wph $ReportArray(img,height)
                     if {[ans_getvalue common,,mccom,,int,16]} {
                        catch {ans_sendcommand )/ui,wsize,$wpx,$wpy,$wpw,$wph} err
                     } else {
                        ::AnsysGUI::AnsysGraphics::sizeWindow $wpw $wph
                        catch {ans_sendcommand )/ui,wsize,$wpx,$wpy,$wpw,$wph} err
                     }
                  }
                  set ReportArray(width) $wpw
               }
               # Animation Size
               if {$ReportArray(animActive) } {
                  set wpx [ans_getvalue active,,win,grph,xpos]
                  set wpx [expr $wpx + $bd]
                  set wpy [ans_getvalue active,,win,grph,ypos]
                  set wpy [expr $wpy + $th + $bd]
                  set wpw [ans_getvalue active,,win,grph,width]
                  set wpw [expr int($ReportArray(animSize) * $wpw)]
                  set wph [ans_getvalue active,,win,grph,height]
                  set wph [expr int($ReportArray(animSize) * $wph)]
                  if {[ans_getvalue common,,mccom,,int,16]} {
                     catch {ans_sendcommand )/ui,wsize,$wpx,$wpy,$wpw,$wph} err
                  } else {
                     ::AnsysGUI::AnsysGraphics::sizeWindow $wpw $wph
                     catch {ans_sendcommand )/ui,wsize,$wpx,$wpy,$wpw,$wph} err
                  }
                  set ReportArray(sizeDownAnim) 1
                  set AnimArray(width) $wpw
                  set AnimArray(height) $wph
               }
            } else {
               set ReportArray(imgSize) $ReportArray(batchSize)
               if { $ReportArray(animActive) } {
                  set ReportArray(imgSize) \
                        [expr int($ReportArray(animSize) * $ReportArray(batchSize) )]
                  set ReportArray(sizeDownAnim) 1
               }
               set AnimArray(width) $ReportArray(img,width)
               set AnimArray(height) $ReportArray(img,height)
               set ReportArray(width) $ReportArray(img,width)
            }
            incr ReportArray(sizeDownImg) 1
         }
         up {
            if { $ReportArray(sizeDownImg) <= 0 } {
               set ReportArray(sizeDownImg) 0
               return
            }
            incr ReportArray(sizeDownImg) -1

            if {$ReportArray(menu) && !$ReportArray(batchCapture)} {
               if {$tk_posinternal} {
                  set bd $tk_borderwidth
                  set th $tk_titleheight
               } else {
                  set bd 0
                  set th 0
               }
               if {$ReportArray(sizeDownAnim)} {
                  set wpx [ans_getvalue active,,win,grph,xpos]
                  set wpx [expr $wpx + $bd]
                  set wpy [ans_getvalue active,,win,grph,ypos]
                  set wpy [expr $wpy + $th + $bd]
                  set wpw [ans_getvalue active,,win,grph,width]
                  set wph [ans_getvalue active,,win,grph,height]
                  set wpw [expr int($wpw / $ReportArray(animSize))]
                  set wph [expr int($wph / $ReportArray(animSize))]
                  if {[ans_getvalue common,,mccom,,int,16]} {
                     catch {ans_sendcommand )/ui,wsize,$wpx,$wpy,$wpw,$wph} err
                  } else {
                     ::AnsysGUI::AnsysGraphics::sizeWindow $wpw $wph
                     catch {ans_sendcommand )/ui,wsize,$wpx,$wpy,$wpw,$wph} err
                  }
                  set ReportArray(sizeDownAnim) 0
                  set AnimArray(width) $wpw
                  set AnimArray(height) $wph
               }
               if {$ReportArray(sizeDownImg) == 0} {
                  set wpx $ReportArray(winpos,x)
                  set wpy $ReportArray(winpos,y)
                  set wpw $ReportArray(winpos,width)
                  set wph $ReportArray(winpos,height)
                  if {[ans_getvalue common,,mccom,,int,16]} {
                     catch {ans_sendcommand )/ui,wsize,$wpx,$wpy,$wpw,$wph} err
                  } else {
                     ::AnsysGUI::AnsysGraphics::sizeWindow 0
                     catch {ans_sendcommand )/ui,wsize,$wpx,$wpy,$wpw,$wph} err
                  }
                  set ReportArray(width) $wpw
               }
            } else {
               if {$ReportArray(sizeDownImg) == 0} {
                  set ReportArray(imgSize) $ReportArray(batchSize)
                  set ReportArray(width) $ReportArray(img,width)
               }
               if {$ReportArray(sizeDownAnim)} {
                  set ReportArray(sizeDownAnim) 0
                  set AnimArray(width) $ReportArray(img,width)
                  set AnimArray(height) $ReportArray(img,height)
               }
            }

         }
      }
      update
      return
   }

   proc interpdynamicdata { input output } {
      variable tags
      variable dArray
   
      # Get the output file name.
      set dArray(outFile) [file tail $output]
      set dArray(outDir) [file dirname $output]
      set dArray(outExt) html
   
      # Check the extension to see if it is 'html'
      set ext [file extension $dArray(outFile)]
   
      if { [string match ".html" $ext] } {
         # Extension is 'html' so remove extension portion of string from
         # filename.
         set dArray(outFile) [file rootname $dArray(outFile)]
      }
      # Set a local variable to the complete path and file being written.
      set dArray(out) [file join $dArray(outDir) $dArray(outFile).$dArray(outExt)]
   
      apdl::noprint 1
      ans_sendcommand )*DEL,_RDIR
      ans_sendcommand )*DIM,_RDIR,CHAR,4
      ans_sendcommand )_RDIR(1)='[string range $dArray(outDir) 0 7]'
      ans_sendcommand )_RDIR(2)='[string range $dArray(outDir) 8 15]'
      ans_sendcommand )_RDIR(3)='[string range $dArray(outDir) 16 23]'
      ans_sendcommand )_RDIR(4)='[string range $dArray(outDir) 24 31]'
      apdl::noprint 0

      # set up the template file name
      set dArray(inFile) $input
   
      # create a directory for final report data
      if { ![file isdir $dArray(outDir)] } {
         file mkdir $dArray(outDir)
      }
   
      # try to open the template file
      if { [catch { set dArray(fidIn) [open $dArray(inFile) r] } err] } {
         ans_senderror 2 [::msgcat::mc "The file %s could not be read !" $dArray(inFile)]
         return
      }
   
      # the file was opened OK. Now try to read the file.
      set buffer [read $dArray(fidIn)]
      close $dArray(fidIn)
   
      # Open the file to be written to.
      set dArray(fidOut) [open $dArray(out) w]
      if { [string match {} $dArray(fidOut)] } {
         ans_senderror 2 [::msgcat::mc "The file %s could not be opened !" $dArray(outFile)]
         return
      }
   
      # Clear the flags for ANSYS tag data.
      set tagStartFound 0
      set tagStopFound 0
      set insideAnsysTag 0
   
      foreach line [split $buffer \n] {
   
         # see if this is an <ANSYS> start or stop tag
         set tmp [string trim $line " "]
         catch { unset l }
         catch { unset r }
   
         # Check for start tag.
         if { [regexp -nocase (.*)${tags(start)}(.*) $tmp match l r] || \
              [regexp -nocase (.*)${tags(start4)}(.*) $tmp match l r] } {
            set tagStartFound 1
   
         # Check for stop tag.
         } elseif { [regexp -nocase (.*)${tags(stop)}(.*) $tmp match l r] || \
                    [regexp -nocase (.*)${tags(stop4)}(.*) $tmp match l r] } {
            set tagStopFound 1
         }
   
         if { $tagStartFound } {
            # Clear the tag buffer
            set dArray(tag) [list]
   
            set tagStartFound 0
   
            # Set flag stating that lines being processed are inside an ANSYS tag.
            set insideAnsysTag 1
   
            # If the start tag was found, check to see if there is any
            # text before or after the tag.
   
            if { [info exists l] && ![string match {} $l] } {
               # If ansys tag is not at beginning of line copy the
               # data to left of ansys tag to the report file
               writeData $l
            }
   
            if { [info exists r] && ![string match {} $r] } {
               # Check for ending ansys tag.
               set in [string trim $r " "]
               catch { unset l }
               catch { unset r }
               if { [regexp -nocase (.*)${tags(stop)}(.*) $in match l r] ||\
                    [regexp -nocase (.*)${tags(stop4)}(.*) $in match l r] } {
   
                  # Text before ansys tag end gets added to the buffer and 
                  # is processed.
                  if { [info exists l] && ![string match {} $l] } {
                     lappend dArray(tag) $l
                     writeTagData 1
                     # Clear the tag data after it has been written.
                     set dArray(tag) [list]
                  }
   
                  if { [info exists r] && ![string match {} $r] } {
                     writeData $r
                  }
                  # If ANSYS end tag was found, clear the inside ANSYS tag
                  # variable.
                  set insideAnsysTag 0
               }
            }
            continue
         }
   
         if { $tagStopFound } {
            # Write the tag data.
            writeTagData 1
            # Clear the tag data after it has been written.
            set dArray(tag) [list]
   
            set tagStopFound 0
            set insideAnsysTag 0
   
            continue
         }
   
         if { $insideAnsysTag } {
            lappend dArray(tag) $line
         } else {
            writeData $line
         }
      }
      if { ![string match {} $dArray(fidOut)] } {
         close $dArray(fidOut)
      }
      catch { ans_sendcommand )/go } err
      catch { ans_sendcommand )/out,term } err
   
      ans_senderror 1 "Generation of report $dArray(out) is complete."
   
      return
   }
   
   proc writeData { line } {
      variable dArray
   
      if { [string match {} $dArray(fidOut)] } {
         set dArray(fidOut) [open $dArray(out) a]
         if { [string match {} $dArray(fidOut)] } {
            ans_senderror 2 [::msgcat::mc "Unable to open file \"%s\"" $dArray(out)]
            return
         }
      }
      puts $dArray(fidOut) $line
   }
   
   proc writeTagData { {append 0}} {
      variable dArray
   
      if { ![string match {} $dArray(fidOut)] } {
         close $dArray(fidOut)
         set dArray(fidOut) {}
      }
      set fn $dArray(outFile)
      set ext $dArray(outExt)
      set path $dArray(outDir)
   
      ::apdl::noprint 1
      if { $append } {
         catch { ans_sendcommand )/output,$fn,$ext,$path,APPEND }
      } else {
         catch { ans_sendcommand )/output,$fn,$ext,$path }
      }
      ::apdl::noprint 0
      set fid [open scratch.tag w]
      if { [string match {} $fid] } {
         ans_senderror 2 [::msgcat::mc "Unable to open file scratch.tag"]
         return
      }
   
      foreach line $dArray(tag) {
         puts $fid $line
      }
      close $fid
   
      ::apdl::noprint 1
      catch { ans_sendcommand )/input,scratch,tag } err
      catch { ans_sendcommand )/output,term } err
      ::apdl::noprint 0
   
      catch { file delete -force scratch.tag }
   }
}
