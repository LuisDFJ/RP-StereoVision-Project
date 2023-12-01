# StereoVision Project
This tool is based on Kernelized Correlation Filters (**KCF**) for Visual Object Tracking (**VOT**). Refer to [1] for more information regarding this topic. The tool is build using **Python 3.11** and third-party dependencies, such as: Numpy [2], OpenCV [3], PyQT5, and Matplotlib [4].
## Requirements
Here a list of requirements for utilizing this tool:
-  [Git](https://git-scm.com/) (Optional).
-   [Python 3.11](https://www.python.org/).
- Third-party dependencies. This will be installed automatically.
## Installation
Make sure you have access to python and git from the command line
> git --version
> python --version

You should see something like this:
> git version 2.38.1.windows.1
> Python 3.11.1

Clone this repository in your PC. If you have git, use the following command, otherwise download the .zip version.
> git clone https://github.com/LuisDFJ/mnTracker-mml.git & cd mnTracker-mml

Run the installer (from the command-line or from the file explorer).
> .\install.bat

Run the following command to confirm that your installation was succesful.
> .\check.bat

You should be prompted with something like:
> contourpy == 1.0.7
> cycler == 0.11.0
> fonttools == 4.39.0
> kiwisolver == 1.4.4
> ...
> python-dateutil == 2.8.2
> six == 1.16.0

Now, is time to run the tool.
> .\run.bat

Enjoy! :smile:

## Stereo-Camera Design

The __./CAD__ directory includes four variations of the jig presented in the manuscript of this repository, which can be used for different applications where a closer range for the working space is required, or if a vertical stereo-camera is desired.

![base_horizontal_180](https://github.com/LuisDFJ/RP-StereoVision-Project/blob/main/CAD/base_horizontal_180.png?raw=true)
![base_horizontal_175](https://github.com/LuisDFJ/RP-StereoVision-Project/blob/main/CAD/base_horizontal_175.png?raw=true)
![base_vertical_180](https://github.com/LuisDFJ/RP-StereoVision-Project/blob/main/CAD/base_vertical_180.png?raw=true)
![base_vertical_170](https://github.com/LuisDFJ/RP-StereoVision-Project/blob/main/CAD/base_vertical_170.png?raw=true)

## References
[1] Henriques, J. F., Caseiro, R., Martins, P., & Batista, J. (2014). High-speed tracking with kernelized correlation filters. _IEEE transactions on pattern analysis and machine intelligence_, _37_(3), 583-596.

[2] Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. _Nature_, _585_, 357–362. https://doi.org/10.1038/s41586-020-2649-2

[3] Bradski, G. (2000). The OpenCV Library. _Dr. Dobb&#x27;s Journal of Software Tools_.

[4] J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007(https://doi.org/10.1109/MCSE.2007.55).
