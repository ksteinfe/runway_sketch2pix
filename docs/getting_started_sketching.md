# How to Get Started Sketching
Using the guidelines described above, a custom Pix2Pix model has been developed and [deployed on RunwayML](https://open-app.runwayml.com/?model=ksteinfe/Sketch2Pix). This GitHub repository hosts this model, and much of the contents seen here serve this purpose.

## Sketch2Pix Runway Model Installation and Testing
Listed below are steps for the installation and basic operation of the Sketch2Pix model on Runway.

### 1. Download and install Runway
|||
|-|-|
|[Download Runway from this link](https://runwayml.com/download/), and install according to your platforms requirements. Runway is Beta software, and is updated often - sometimes for the better, sometime for worse.  | ![](http://media.ksteinfe.com/200317/sketch2pix_demo_image14.png) |

### 2. Create a Sketch2Pix Workspace
|||
|-|-|
|In Runway, click on the "Browse Models" tab (CNTL+@), and search for the Sketch2Pix model. Once you’ve found it, click on the "add to workspace" button to add it to a new workspace.  | ![](http://media.ksteinfe.com/200317/sketch2pix_demo_image1.gif) |

### 3. Run the Model
|||
|-|-|
|After choosing any one of the checkpoints (listed on the right panel), click the "Run Remotely" button to start the model. This will take some time. As the model is loading, move on to the next step.  | ![](http://media.ksteinfe.com/200317/sketch2pix_demo_image3.png) |

### 4. Download Test Sketches
|||
|-|-|
|A number of sample sketches are available for download from [this link](https://github.com/ksteinfe/runway_sketch2pix/raw/master/photoshop_plugin/releases/sketch_samples.zip). Download this ZIP file and copy the contents to a dedicated directory your local hard drive (such as a folder called "samples" on your desktop).  | ![](http://media.ksteinfe.com/200317/sketch2pix_demo_image6.jpg) |

### 5. Prepare a Test Sketch
|||
|-|-|
|From the “Input Type” dropdown on the top of the screen, select “File”. Then browse to the directory that you saved the sample sketches, and click “Select Folder”. After that, you should see a list of thumbnails of sample sketches on which to run an inference.  | ![](http://media.ksteinfe.com/200317/sketch2pix_demo_image7.png) |

### 6. Run Inferences
|||
|-|-|
|With the model running, any time an image is selected from the “input” section, it is passed to the remote model for inference, and the result shown in the “output” section, as shown in the nearby animation. At any time, you may choose to stop the model (using the Stop button), re-load a different checkpoint, and run the model again.   | ![](http://media.ksteinfe.com/200317/sketch2pix_demo_image8.gif) |

### 7. Stop the Model
|||
|-|-|
|Remember that is costs money to run models remotely on RunwayML. As such, we should always remember to STOP ALL MODELS before closing Runway. You may do so by clicking on the Stop button.  | ![](http://media.ksteinfe.com/200317/sketch2pix_demo_image10.png) |

## Sketch2Pix Photoshop Plugin Installation and Testing
A plugin has been developed that links Photoshop to RunwayML, and is currently [available for download from this repository](https://github.com/ksteinfe/runway_sketch2pix/tree/master/photoshop_plugin/releases).

Listed below are steps for the installation and basic operation of the Sketch2Pix Photoshop plugin. We assume that the above steps for installing and testing Runway have been completed.

### 1. Download the Sketch2Pix Photoshop Plugin
Using the link above, download the latest release of the Sketch2Pix photoshop plugin. Note that releases are packed into ZIP files that are named for the date of release. To download a ZIP file from GitHub, click on the desired file to bring up the repository page for the file, and then click the "download" button.

### 2. Set Debug Mode
For the Photoshop plugin to work, we'll have to make one adjustment in the deep bowels of our computer.

#### In Windows
First, open the Registry Editor. 

![](http://media.ksteinfe.com/200317/sketch2pix_demo_image2.png)

Then, using the directory tree on the left-hand side, browse to the following location:

	HKEY_CURRENT_USER / 
		Software / 
			Adobe / 
				CSXS.9


Once in the proper location, add a new entry called `PlayerDebugMode` of type `string` with the value of `1`.

![](http://media.ksteinfe.com/200317/sketch2pix_demo_image13.gif)

#### In OSX
Using the Terminal, execute the following command (don’t copy the $):

	$ defaults write com.adobe.CSXS.9 PlayerDebugMode 1


### 3. Copy the Plugin Files
Unzip the contents of the ZIP file downloaded in the step above, and place these into the appropriate subfolder of the Photoshop Extension directory. More information regarding Adobe CEP is available [here](https://github.com/Adobe-CEP/CEP-Resources/blob/master/CEP_9.x/Documentation/CEP%209.0%20HTML%20Extension%20Cookbook.md#extension-folders).

#### In Windows
Locate the Photoshop Extension Directory, which in Windows is located at:

	C:\Program Files\Common Files\Adobe\CEP\extensions

Create a new directory at this location called “Sketch2Pix”. The whole path should now be:

	C:\Program Files\Common Files\Adobe\CEP\extensions\Sketch2Pix

Copy the contents of the ZIP file into this newly-created directory.

#### In OSX
Locate the Photoshop Extension Directory, which in OSX is located at:

	~/Library/Application Support/Adobe/CEP/extensions

Note that the directory mentioned above is located in the “root” directory, or Macintosh HD, which does not appear on the Desktop, nor in new Finder windows unless it is configured to in Finder Preferences. To show Macintosh HD as a desktop icon, click on Finder from the top menu, and in the General Tab, check off Hard Disks. Macintosh HD should now appear on your Desktop. Alternatively in Finder Preferences, set New Finder windows to show Macintosh HD.

Create a new directory at this location called “Sketch2Pix”. The whole path should now be:

	~/Library/Application Support/Adobe/CEP/extensions/Sketch2Pix

Copy the contents of the ZIP file into this newly-created directory.


### 4. Open Runway and Start the Pix2Sketch Model
Following the instructions above, make sure a model is running and is able to serve up predictions.

### 5. Open the Sketch2Pix Panel in Photoshop
Windows -> Extensions -> Sketch2Pix

![](http://media.ksteinfe.com/200317/sketch2pix_demo_image11.png)

### 6. Create and Configure a PSD File
Instructions will be provided on how to properly setup a PSD file for sketching.

### 7. INFER!
With the desired layer selected, press the “abracadabra!” button

![](http://media.ksteinfe.com/200317/sketch2pix_demo_image4.png)

### 8. Rinse and Repeat
A new layer group called “generated” will be created.


## Sketching Using Runway + Photoshop
Here is a nice video that shows how to use the Photoshop plugin in concert with the Sketch2Pix Runway model to run inferences on your own sketches.


[![link to tutorial video](http://media.ksteinfe.com/200317/sketch2pix_in_photoshop.png)](http://media.ksteinfe.com/200317/sketch2pix_in_photoshop.mp4)

