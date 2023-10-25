# IPVC_project
## TODO
matlab functies omzetten naar python of opencv functies vinden


## Plan
0. Matlab camera constanten -> python

Voor ieder persoon 5 foto's dus uiteindelijk 15 meshes.
Ieder set heeft 3 fotos, dus 1 triplet

### Per triplet
1. Maak mask voor iedere afbeelding (voor de achtergrond)

#### Pre-processing
2. Non-linear lens deformation should be compensated for
3. stereo rectification is needed to facilitate the dense stereo
matching.
4. global colour normalization to make sure that the so-called ‘Constant Brightness Assumption’ holds true. A normalization could be applied with respect to
mean and standard deviation of the colour channels.

Eerst maken we 2 meshes per triplet, die we later samenvoegen. 
#### Mesh maken
4. Stereo matching -> point cloud
5. Bepaal confidence
6. Point cloud -> 3D mesh
5. Meng de 2 meshes

# Taakverdeling
GIJS stereo rectification + Non-linear lens deformation
CHRIS mask
JELISE stereo rectification
DAMIAN global colour normalization + Stereo matching

# Report
Q: How to do XXX

Method: In order to do XXX, I propose method XXX and validate the method by experiment XXX.

Results: the results of experiment XXX show XXX. It reflects our proposed method is working or not.

Discussion: summary of the finding, how this proposed method is different to other state of the art work?  What is the limitation? What can we get out of this result and dig deeper on the scientific discovery.

Conclusion: the proposed method is capable or not capable to do XXX. If so, what could be the impact and potential, if not working, why it is so? How to let others avoid this…
