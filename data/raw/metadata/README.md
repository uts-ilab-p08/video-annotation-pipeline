These are the KRTD camera model files for the released MEVA data clips.

Each camera model is associated with a clip name; these files can also be used with other clips as identified in the [meva-clip-camera-and-time-table.txt](../../meva-clip-camera-and-time-table.txt) file; see the associated [readme](../../clip-table-readme.md) for more details.

All cameras *with the exception of the indoor cameras G299 and G330* are registered into a common east, north, up (ENU) coordinate system with meters units and origin (0, 0, 0) located at:

* latitude = 39.04977294       degrees
* longitude = -85.52924953   degrees
* height = 205 meters above WGS84 ellipsoid

This is the same coordinate system used by the provided 3-D models of the grounds, and all camera calibrations should be consistent with the 3-D models.

The data here is stored in KRTD format. KRTD is an ASCII file containing the following matrices:

K - 3x3 upper triangular intrinisic matrix encoding focal length, principal point, skew, pixel aspect ratio.

R - 3x3 orthonormal rotation matrix

T - 3d translation vector

D - Nd vector of lens distortion parameters (using OpenCV conventions)

