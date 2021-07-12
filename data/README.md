# Data description
## File
* anonymizedstudents.jld2

An HDF5 file to be handled by JLD2 Julia package (ver. 0.2.4).

## Data structure
When opened via @load macro, an object "anonymizedstudents",
a vector of 10,923 elements each of which represent a student in the Matsumoto city dataset, will be imported.

Each element representing a student has the following fields:
* **isinfected**: Whether the student reported flu infection during the study period
* **onset**: An integer representing the onset date of the student (1 October 2014 corresponds to 1)
* **schoolID, gradeID, classID**: Indicates the school, grade, and class of the student
* **sex**: 0: female, 1: male
* **suscovar**: A vector representing Standardized (i.e. limited to the range [0,1]) covariates potentially associated with susceptibility of students
* **infcovar**: A vector representing Standardized (i.e. limited to the range [0,1]) covariates potentially associated with infectiousness of students
* **classsize**: The size of the class that the student belongs to
* **nclasses**: The number of classes in the grade that the student belongs to
* **householdlikelihoodratio**: The log of the household likelihood ratio used in Eq. S5, Supplementary materials: π in the first term divided by the second. 
* **sampleweight**: The sample weight for the student to account for sampling bias as described in Eq. S11 in the supplementary materials
