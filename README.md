![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Summer Paper Implementations

Hey team! Go ahead and clone the repo, add your own branch (just title it your name) with all your files and read.me with your synopsis, and then publish your new branch. Let us know if any of y'all have access issues.

Compatible on Google Collab (use T4 GPU)
Make sure to upload the dataset to drive and after running first cell it will ask to mount drive on collab agree on that.
Additionally, crosscheck dataset path in the notebook!

CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning

Overview
The paper "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning" introduces an advanced deep learning model designed to detect pneumonia from chest X-rays, achieving radiologist-level accuracy. The model employed a 121-layer DenseNet CNN architecture. They also used the ChestX-ray14 dataset, which consists of over 100,000 frontal-view X-ray images of 30,805 unique patients which is labeled with 14 different thoracic diseases. The model's performance demonstrated superior diagnostic accuracy compared to practicing radiologists in detecting pneumonia.

CheXNet was trained to classify 14 different pathologies, with a particular emphasis on pneumonia detection. The model's performance was rigorously evaluated against four board-certified radiologists, demonstrating that CheXNet achieved a higher F1 score than the radiologists, indicating a superior balance of precision and recall in detecting pneumonia. To enhance the model's interpretability, the authors employed Class Activation Maps (CAMs), which highlight regions in the X-ray images that are most indicative of the predicted condition. This feature provides visual insights into the model's decision-making process, potentially increasing trust and transparency in clinical settings. The study justifies the development of CheXNet as a response to the critical need for accurate, scalable, and timely diagnosis of pneumonia, especially in settings where radiological expertise is scarce.

Limitations
•	Diagnosis is based only on frontal chest X-rays.
•	No patient history provided neither to the model nor radiologists. It can significantly impact the diagnosis.
•	Can’t mimic the iterative reasoning.

Strength
•	Diagnostic accuracy for pneumonia detection.
•	Achieved state-of-the-art results across all 14 diseases in ChestX-ray.
•	Visual image of the location of the diagnosed X-ray.

Weakness
•	Low accuracy on torso X-rays.
•	Clinical generalization, demographics of the patients.
•	Risk of misdiagnosis

Novelty
•	Radiologists-Level Accuracy
•	State-of-the-art model
