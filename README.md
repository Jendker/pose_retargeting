# pose_mapping

### Compile catkin with Python 3 compatibility

This package works with Python 3. To use it with ROS Kinetic or newer:

```
cd ~/catkin_ws
rm -rf devel build
catkin build
source devel/setup.zsh  # or setup.bash
cd src 
git clone https://github.com/ros/geometry
git clone https://github.com/ros/geometry2
cd ..
rosdep install --from-paths src --ignore-src -y -r  # not sure if needed
catkin build --cmake-args \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
source devel/setup.zsh  # or setup.bash
```
(source https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/ and https://github.com/ros/geometry2/issues/259)


