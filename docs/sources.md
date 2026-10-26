# Sources

mojo-cv is written from public documentation and papers, not from another library's code. The 2D ops follow
OpenCV 5.0 semantics, so cv2 5.0 is the oracle for them; the 3D ops are checked against brute force or a closed form.

## 2D vision and geometry

- OpenCV 5.0 documentation: the function pages define the semantics each op matches.
- Rublee et al. 2011 (ORB); Rosten & Drummond 2006 (FAST); Calonder et al. 2010 (BRIEF).
- Lucas & Kanade 1981; Bouguet 2000 (pyramidal LK); Farneback 2003.
- Canny 1986; Suzuki & Abe 1985 (contours); Shi & Tomasi 1994; Rother et al. 2004 (GrabCut);
  Dalal & Triggs 2005 (HOG).
- Lepetit, Moreno-Noguer, Fua 2009 (EPnP); Fischler & Bolles 1981 (RANSAC); Hartley & Zisserman, Multiple
  View Geometry; Nister 2004 (5-point); Hirschmuller 2008 (SGM); Zhang 2000 (calibration);
  Garrido-Jurado et al. 2014 (ArUco).
- Stewenius, Engels & Nister 2006, *Recent developments on direct relative orientation*, ISPRS J. 60(4):284-294,
  [doi:10.1016/j.isprsjprs.2006.03.005](https://doi.org/10.1016/j.isprsjprs.2006.03.005): the action-matrix variant
  of Nister 2004 used for `findEssentialMat`.

## 3D, time and CAD

- Zobrist 1970; Meagher 1982 (octrees).
- Umeyama 1991 (least-squares similarity between point patterns).
- Mallat, A Wavelet Tour of Signal Processing; Kuglin & Hines 1975 (phase correlation).
- Koppula et al. 2024 (TAPVid-3D); Wronski et al. 2019 (handheld burst super-resolution).
- Piegl & Tiller, The NURBS Book; Khronos glTF 2.0 specification; ISO 10303-21 and the AP242 schema.
- Any continuum mechanics text for von Mises and principal stresses.
