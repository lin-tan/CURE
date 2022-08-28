# CURE's Candidate Patches

## File Structure
* **meta.txt**: This file includes the meta information of the bugs in Defects4J v1.4, including project name, bug id, buggy file path, start and end line number of the bugs
* **defects4j_bpe.txt**: This file includes the input of bugs in Defects4J v1.2 to CURE models. Each line refers to a bug in the following format: buggy line &lt;CTX&gt; surrounding function
* **identifier.txt**: This file includes the valid identifiers for each bug in Defects4J v1.2. Each line refers to the valid identifiers for a bug.
* **identifier.tokens**: This file includes the tokenized identifiers for each bug, which is requires by CURE's model when generating patches.
* **candidate_patches.json**: This file includes the generated candidate patches for bugs in Defects4J v1.4. For time reason, we only validated 228 bugs that are more likely to be fixed, which are all list in this json file. CURE validates about 5000 patches for each bug.
