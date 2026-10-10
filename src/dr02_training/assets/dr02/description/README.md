# DR02 Pro model sources

`urdf/dr02_pro.urdf`, `mjcf/dr02_pro.xml`, and the referenced STL meshes were copied from
`robot_lab/source/robot_learning_lab_zoo/robots/deeprobotics/dr02_pro_description` (robot_lab revision
`835b39e3bcf5f4450d22e0c6f3efbf4893e9d817`).
The URDF is the articulation imported by Isaac Lab. The MJCF is retained as the MuJoCo source. Inspection and parser
checks confirm the same 40 named bodies, matching body-parent edges, and the same 29 actuated joints with matching
axes and position limits. The MJCF also has one unnamed free root joint. This establishes the task's name/topology
mapping; it does not claim full dynamics parity across the two models.

The task controls the 29 revolute joints in the order recorded in `profile.json`. Its AMP and tracking motion clips use a
32-point body order, including the left and right toe sites reconstructed from fixed offsets on the ankle bodies.

The source directory did not include a license or an upstream provenance record for the robot description and meshes.
This package preserves that provenance limitation and does not claim those vendor assets are relicensed. Resolve the
vendor redistribution terms before publishing this package outside its intended internal use.
