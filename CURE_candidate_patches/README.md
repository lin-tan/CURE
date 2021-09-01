# CURE's Candidate Patches

## File Structure
* **meta.txt**: This file includes the meta information of the bugs in Defects4J v1.4, including project name, bug id, buggy file path, start and end line number of the bugs
* **rem.txt**: This file includes the buggy line that need to be removed of the bugs in Defects4J v1.4
* **context.txt**: This file includes the context (surrounding method) of the buggy line of the bugs in Defects4J v1.4
* **candidate_patches.json**: This file includes the generated candidate patches for bugs in Defects4J v1.4. For time reason, we only validated 228 bugs that are more likely to be fixed, which are all list in this json file. CURE validates about 5000 patches for each bug.
