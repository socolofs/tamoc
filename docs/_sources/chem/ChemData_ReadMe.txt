###############################
Sources of Data in ChemData.csv
###############################

:Release: |version|
:Date: |today|

The following list documents the sources of data in the ./data/ChemData.csv 
file.

* Molecular weights (M) are from the periodic table in Fine & Beall, 1990,
  Chemistry for Engineers and Scientists, Saunders College Publishing:
  Philedelphia, Pennsylvania. Data are in (g/mol).

* Critical-point data (Pc, Tc) are mostly from McCain, 1990, The Properties of
  Petroleum Fluids, 2nd Edition, PennWell Books: Tulsa, Oklahoma. Exceptions
  are for Argon, which is from P.4-127 of the Handbook of Chemistry and
  Physics, 71st Edition. The data are in psia for pressure and R for
  temperature.

* Acentric factors (omega) are mostly from McCain, 1990, The Properties of
  Petroleum Fluids, 2nd Edition, PennWell Books: Tulsa, Oklahoma. Exceptions
  are for Argon, which is from P.4-127 of the Handbook of Chemistry and
  Physics, 71st Edition. The data are in psia for pressure and R for
  temperature.

* Henry's law constants (kh_0) are from Sander 1999, Compilation of Henry's
  Law Constants for Inorganic and Organic Species of Potential Importance in
  Environmental Chemistry, downloaded from
  http://www.rolf-sander.net/henry/henry.pdf on January 2, 2012. The data are
  in mol/(dm^3 atm).

* Enthalpy of solution (dH_solR) are from Sander 1999, Compilation of Henry's
  Law Constants for Inorganic and Organic Species of Potential Importance in
  Environmental Chemistry, downloaded from
  http://www.rolf-sander.net/henry/henry.pdf on January 2, 2012. The data are
  normalized by the universal gas constant R and reported in K.

* Specific volume at infinite dilution (nu_bar) are from Wilhelm et al., 1977,
  Low-pressure solubility of gases in liquid water, Chemical Reviews, 77(2):
  pp. 219-262.

* Diffusivity model coefficients (B, dE) are from Wise and Houghton, 1966,
  The diffusion coefficients of ten slightly soluble gases in water at 
  10-60 deg C, Chemical Engineering Science, 21: pp. 999-1010.  The data are
  in cm^2/sec (B) and cal/mol (dE).  
