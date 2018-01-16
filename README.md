# IVP-Project
Manu und Johanns IVP Projekt

This is some information about the usage of this code.

To run the Main.py file you need to have the folders /img containing the test images
as well as /lib containing the source code.

Inside the Main.py script all the test images are called in a loop.
Each is then encoded using both Hufmann and Runlength Coding (implemented in
encoding_framework.py) and saved as .bin files
in a folder /bin . For that purpose, dumpHuffman and dumpRunLength are set to True.
Additionally, a rgb file can be saved the same way (dumpRGB).

For the encoding using each method, the time for the processing is printed as well as the
entropy of that image.

After that, the .bin files are read in and decoded using the decoders in decoding_framework.py.

If the showImgs flag is set, the original as well as decoded images are shown using cv2.imshow()
