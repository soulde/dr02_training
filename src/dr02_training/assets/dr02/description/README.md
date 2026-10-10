# DR02 Pro model sources

The URDF and MJCF are adapted from the DeepRoboticsLab DR02 Pro model for training use by soulde. The adapted files
were recorded in soulde's `robot_lab` repository at
`source/robot_learning_lab_zoo/robots/deeprobotics/dr02_pro_description`, revision
`835b39e3bcf5f4450d22e0c6f3efbf4893e9d817`. They are not byte-for-byte identical to the current upstream URDF/MJCF.
The 36 STL meshes are byte-for-byte identical to the DR02 Pro meshes in
[DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model/tree/75824b516ecc8fa3f8f7ce5d6577e1e86d6b614f/DR02/urdf/pro/meshes)
at revision `75824b516ecc8fa3f8f7ce5d6577e1e86d6b614f`. The upstream repository's BSD 3-Clause license is included
in `LICENSE.txt` in this directory.

The URDF is the articulation imported by Isaac Lab. The MJCF is retained as the MuJoCo source. Inspection and parser
checks confirm the same 40 named bodies, matching body-parent edges, and the same 29 actuated joints with matching
axes and position limits. The MJCF also has one unnamed free root joint. This establishes the task's name/topology
mapping; it does not claim full dynamics parity across the two models.

The task controls the 29 revolute joints in the order recorded in `profile.json`. Its AMP and tracking motion clips use a
32-point body order, including the left and right toe sites reconstructed from fixed offsets on the ankle bodies.

soulde's `robot_lab` source directory did not include a separate license or upstream provenance record. The adapted
URDF/MJCF and matching mesh files are attributed to the upstream model repository and the recorded soulde revision;
see the included BSD 3-Clause license for the model asset terms.
