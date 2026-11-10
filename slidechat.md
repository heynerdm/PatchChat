# **SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding** 

Ying Chen<sup>1</sup><sup>_,_2</sup><sup>_∗_</sup> Guoan Wang<sup>1</sup><sup>_,_3</sup><sup>_∗_</sup> Yuanfeng Ji<sup>4</sup><sup>_∗_†</sup> Yanjun Li<sup>1</sup><sup>_,_3</sup> Jin Ye<sup>1</sup><sup>_,_5</sup> Tianbin Li<sup>1</sup> Ming Hu<sup>1</sup><sup>_,_5</sup> Rongshan Yu<sup>2</sup> Yu Qiao<sup>1</sup> Junjun He<sup>1†</sup> 

1Shanghai AI Laboratory 

2Xiamen University 3East China Normal University Stanford University 5Monash University 

4Stanford University 

## **Abstract** 

_Despite the progress made by multimodal large language models (MLLMs) in computational pathology, they remain limited by a predominant focus on patch-level analysis, missing essential contextual information at the whole-slide level. The lack of large-scale instruction datasets and the gigapixel scale of whole slide images (WSIs) pose significant developmental challenges. In this paper, we present SlideChat, the first vision-language assistant capable of understanding gigapixel whole-slide images, exhibiting excellent multimodal conversational capability and response complex instruction across diverse pathology scenarios. To support its development, we created SlideInstruction, the largest instruction-following dataset for WSIs consisting of 4.2K WSI captions and 176K VQA pairs with multiple categories. Furthermore, we propose SlideBench, a multimodal benchmark that incorporates captioning and VQA tasks to assess SlideChat’s capabilities in various settings such as microscopy, diagnosis and clinical. Compared to both general and specialized MLLMs, SlideChat exhibits exceptional capabilities, achieving state-of-the-art performance on 18 of 22 tasks. For example, it achieved an overall accuracy of 81.17% on SlideBench-VQA (TCGA), and 54.15% on SlideBench-VQA (BCNB). Our code, data, and model is publicly accessible at https://uni-medical.github.io/SlideChat.github.io._ 

## **1. Introduction** 

Computational pathology aims to improve the analysis of digitized tissue samples [15, 25], such as whole slide images (WSIs), by applying artificial intelligence to aid in the diagnosis, identification, and understanding of disease. Recently, the development of this field has gained rapid 

*These authors contributed equally to this work. 

> †Corresponding authors: yfj@stanford.edu, hejunjun@pjlab.org.cn. 



<!-- Start of picture text -->
A B<br>Microscopy<br>4 TeweAchtoctxe Tuner Charactorstis<br>‘Histopathol Changes<br>‘#oytomerphological Charactarsos<br>‘Summarize the key pathological<br>findings‘This pathological fom the whole examination side image reveals the | ‘DiseaseDiagnosisocavo Detecon Cassfcaton@ Stang # Gradng<br>high-grade urothelial carcinomawith 1 biter! Diagnosis<br>Whatthe‘A.pT1B.pT2C. vansiional isthe pathological cellpT3D. carcinoma? stagepT (pT)B.pr2 of “¢PrognoeteSrosimentliniatiment AssssmentuaeGuidance eneRisk Factors<br>Biomarker Anais<br><!-- End of picture text -->

Figure 1. SlideChat is the first large vision-language assistant specifically designed for whole-slide pathology analysis. SlideChat can generate comprehensive descriptions of whole-slide images and provide contextually relevant responses across various applications. 

momentum, mainly driven by breakthroughs in the visual foundation model [9, 28, 31]. These models learn generalized representations by pre-training on large-scale data and perform well in various downstream tasks, such as cancer detection and biomarker prediction. Building on this base, integration with the powerful Large Language Models (LLMs) further advances the development of the Multimodal Large Language Model (MLLMs) [21], which has made great strides in responding to more complex, openended visual queries, enabling it to serve as a versatile assistant at various stages of medical care, including clinical decision-making, education, and research. 

Nevertheless, there are some major challenges that hinder the development and use of pathology MLLMs for realworld applications. First, it is challenging to develop a MLLM architecture that can effectively be capable of gigapixel whole slides ( _e.g._ , 100,000 × 100,000 pixels). Existing models [21, 23, 27] often process whole slides by extracting small patch/ROI-level data for subsequent analysis, resulting in a limited understanding of global slide con- 

text and suboptimal performance in some complex pathological analysis. Second, publicly available multi-modal pathology slide datasets are relatively scarce and of varying quality [7, 8, 13], which limits the development of MLLMs trained on such data. 

In this paper, we present SlideChat (see Fig. 1), the first open-source vision-language assistant capable of understanding gigapixel whole-slide images. First, SlideChat is trained on SlideInstruction, a large-scale multi-modal instruction dataset encompassing data from The Cancer Genome Atlas (TCGA) [16] via our specifically designed data processing pipeline (see Fig. 3 (A)). SlideInstruction contains 4,181 WSI-caption pairs and 175,753 visual question-answer pairs from 3,294 patients, covering 10 cancer types. The question-answer pairs include both openended and closed-ended questions, further divided into 13 subcategories, covering a diverse range of clinical tasks. SlideInstruction is more than 20 times larger than previous public instruction datasets in the number of instructions. Second, we propose SlideChat, a novel architecture in LLaVA style for capable multi-modal analysis of gigapixel whole slides. A gigapixel whole slide is first divided into a series of patches, each of which is individually processed by a patch-level encoder to extract local features. The resulting long sequence of patch tokens is then processed by a slidelevel encoder employing sparse attention to enhance the slide-level features. Finally, enhanced patch tokens are fed into the LLM via a projector, which processes user queries and generates responses. 

To systematically evaluate the performance of SlideChat in real-world scenarios, we establish a comprehensive digital pathology benchmark named SlideBench, encompassing more than 20 clinical tasks, using data from both TCGA and the in-the-wild Early Breast Cancer CoreNeedle Biopsy (BCNB) dataset. This resulted in three test sets: SlideBench-Caption, comprising 734 WSI-caption pairs; SlideBench-VQA (TCGA), comprising 7,827 VQA pairs covering 13 different tasks; and SlideBench-VQA (BCNB), including a total of 7,274 VQA entries from 1,058 patients, covering seven different tasks. Additionally, we compare the performance of SlideChat on another externally proposed dataset, WSI-VQA [8], to further validate its effectiveness. We compare our model with the currently available state-of-the-art general and medical-specialized MLLMs including GPT-4o, LLava-Med [17], MedDr [14] and Quilt-LLaVA [23]. Benefiting from large-scale, highquality training and effective local-global context modeling, SlideChat achieves state-of-the-art performance on 18 out of 22 tasks, with significant improvements over the secondbest method 10% on 9 tasks on four benchmarks. Specifically, SlideChat achieves an average accuracy improvement of 13.47% over the second-best model on SlideBenchVQA (TCGA), an average improvement of 12.59% on 

SlideBench-VQA (BCNB), and an improvement of 5.82% on WSI-VQA. Finally, to accelerate research progress in digital pathology, we make SlideChat fully open-weight, including source code and model weights as well as instruction and benchmark datasets. The key contributions are summarized four-fold in the following: 

- We develop SlideChat, the first vision-language assistant capable of understanding gigapixel wholeslide images, achieving state-of-the-art performance on multiple benchmarks. 

- We create SlideInstruction, a largest comprehensive WSI instruction-following dataset containing 4.2K WSI-caption pairs and 176K VQA pairs. 

- We establish SlideBench, a WSI multi-modal benchmark comprising SlideBench-Caption, SlideBenchVQA (TCGA), and SlideBench-VQA (BCNB), covering 21 different clinical tasks. 

- We release SlideChat, SlideInstruction and SlideBench as open-source resources to facilitate research and development in computational pathology. 

## **2. Related Works** 

**Whole Slide Image Analysis** Whole slide images are pivotal in modern pathology, enabling comprehensive analysis of tissue samples for tasks such as predicting patient prognosis, classifying cancer subtypes, and identifying biomarkers [18, 19, 24–26, 35]. Recent studies have leveraged pathology foundational models [3, 29, 32] to enhance WSIs analysis, either through fine-tuning for specific downstream tasks or by employing zero-shot prediction approaches in CLIP [22] style. Although these models are effective in task-specific applications, their reliance on fine-tuning or limited zero-shot capabilities restricts their generalizability across diverse and complex user instructions. 

**MLLMs in Computational Pathology** The paradigm of MLLMs enables them to effectively respond to more complex, open-ended visual queries while processing pathology images, thus providing significant value across various medical stages. PathChat [21] is a vision-language assistant designed for pathology, developed with 450K private instruction pairs to handle both visual and natural language queries. Quilt-Instruct [23] is a large-scale dataset comprising 107K question-answer pairs. Building on Quilt-Instruct, Quilt-LLAVA [23] is a model designed for diagnostic reasoning across multiple image patches, leveraging its extensive question-answer pairs to accurately interpret complex H&E images. PathAsst [27] combines a pathology-specific CLIP model with Vicuna-13b [10] to create a multimodal generative foundational model tailored for pathology. However, current MLLMs primarily only focus on patch or ROI data, limiting their utility for slide-level clinical applications where broader contextual understanding is crucial. 



<!-- Start of picture text -->
A B<br>#Query: What is the type of tumor in the image? Stage1: Domain Alignment<br>……<br>4.2K WSI-caption pairs<br>SlideChat<br>LLM Projection<br>WSI patching Projection Slide-level Encoder<br>Slide-Level  Stage2: Instruction Tuning<br>… Encoder Large Language Model<br>176K WSI VQAs<br>… Patch-Level  LLM Projection<br>Encoder<br>Slide-level Encoder<br>224 × 224 patch sequence #Response: Lung Adenocarcinoma<br><!-- End of picture text -->

Figure 2. Overview of our SlideChat. (A) SlideChat serializes each input WSI into a sequence of 224×224 patches, converting each into visual embeddings with a patch-level encoder. A slide-level encoder then interacts with these features to generate contextual embeddings. Then, a multimodal projector maps the visual features from the slide-level encoder into a unified space, aligned seamlessly with the LLM. (B) SlideChat was trained for two stages: Cross-Domain Alignment and Visual Instruction Learning. 

## **3. SlideChat** 

### **3.1. Architecture** 

To achieve the goal of analyzing gigapixel whole-slide images in a multimodal setting, as shown in Fig. 2, SlideChat consists of four key designs: the patch-level encoder, the slide-level encoder, the multimodal projector module, and the large language model. Our method starts by partitioning the WSI into smaller 224 × 224 pixel patches, making it computationally feasible to process such large images. These patches are then passed through a well-trained, frozen patch-level encoder [20], which extracts localized features from each individual patch, capturing fine-grained details such as cellular structures. Building on this, we employ LongNet [12, 31] as slide-level encoder to enhance the patch-level embeddings and capture global patterns across the entire slide. This encoder uses sparse attention mechanisms to learn both local and global contextual information, enabling the model to perceive intricate local features while capturing the broader context, which is critical for comprehensive pathological assessments. Following the slide-level encoding, SlideChat incorporates a multimodal projector that maps these patch tokens into a unified space aligned with the LLM. This ensures that the visual features extracted from the WSIs are effectively transformed into representations compatible with the language model, facilitating seamless integration and interaction between visual and textual data. Concurrently, the model accepts natural language instructions from users, such as “What is the type of tumor in the image?”. These textual queries are processed by the LLM, which comprehends the textual input and integrates it with the visual features extracted from the WSIs, enabling accurate and contextually relevant diagnos- 

tic responses. This multimodal reasoning capability allows SlideChat to provide accurate and contextually relevant answers to complex pathology-related questions, thereby supporting clinical decision-making, education, and research across various medical stages. 

### **3.2. Data** 

**SlideInstruction** There is a notable lack of large-scale multimodal pathology datasets supporting the training of vision-language assistants for whole-slide image understanding. To support the training of SlideChat, we develop SlideInstruction, a comprehensive instruction dataset, sourced from the TCGA database, comprising 4,915 WSIreport pairs from 4,028 patients. Fig. 3 illustrates our entire data curation pipeine. We initially prompt GPT-4 to refine the pathology reports, and clean up the noise in the report including _unrelated symbols, technical details of pathology department procedures, specimen handling and processing information, redundant administrative or legal statements, and some repeated information_ . For the refined pathology reports, we further employ GPT-4 to generate high-quality multimodal data, comprising two main components: (1) _WSI-Caption Data_ : We craft concise, clinically relevant summaries for each whole slide image by prompting the language model to extract key pathological findings. These summaries were structured into coherent paragraphs that highlighted crucial clinical details such as diagnostic results, tumor characteristics, margin status, and lymph node involvement, ensuring the caption dataset is both focused and informative. (2) _WSI InstructionFollowing Data_ : To enhance the model’s ability to follow instructions and improve its comprehension of pathology images, we leveraged GPT-4 to generate tailored question- 



<!-- Start of picture text -->
A Prompts Slideinstruction<br>[Report Cleaning}:' [WSI-Capton): xaK<br>WSI-Report1 ‘This image reveals a diagnosis of invasive ductal<br>Remove some redundant irrelevant information Sea cine ia<br>: GPT4<br>== BeGUICe TNC TOC EN #0:[Open-set What specific VOA): nuclear atypia is described in391K<br>= ® Sn — — | Deis observed in the image?<br>== #4: Megium and high nuclear atypia.<br>a era x84K<br>categorized[VQA Generation: under Create several questions #0:‘A.CC. TumorClearWhatis invasionexcisionthe marginsarchitecturalinto vesselsD. Lmphovasculrfeature8. Alveolarin the colapse invasionimage?<br>B<br>[Closed-set VA: = FD Mieroscopy<br>‘A. Normal lobular arrangement B, Parenchymal z Ee<br>- Clinical<br>Test Closed-Set VA from Step A-like Process LLMs Filtering Expert Validation SlideBench-VOA<br>c<br>oh oe en {Caption} er GS [VQA-Microscopy}<br>KpEoseS| | difterentiated#2:24:  DescriveHeer theurothelialxbwhole carcinoma, ovateside imagegrade pot4/4, ‘se> "sSee > +#0:‘wholeastrocytomaWhat slide histological imagefrom differentiates glioblastomafeature observable anaplastic?  in a<br>NEON extending through the bladderneck wall se a ‘#4: Absence of endothelial proliferation<br>wie' : se [VOADiagnosis}#0: From the given whole side image, which #0:IVOA-Cinical]Based on the histopathological features,<br>Whats the primary recommended treatment?<br>histological subtype of lung adenocarcinoma<br>is most predominantly observed? +#A: Simple mastectomy followed by homone<br>#4; Itis acinar subtype observed. therapy.<br><!-- End of picture text -->

Figure 3. (A) Overview of the SlideInstruction generation pipeline. We prompt GPT-4 to extract the WSI-Caption, Open-set VQA and Closed-set VQA from reports. (B) For the generated Closed-set VQA, we employ LLMs to filter low-quality QA pairs and involve pathologists for validation, resulting in the creation of SlideBench-VQA. (C) Examples of WSI caption and instruction-following scenarios in microscopy, diagnostics, and clinical applications. For additional examples, please refer to Supplementary Material. 

and-answer pairs for each WSI report. Drawing inspiration from PathChat [21], we structure these questions into 3 broad categories—microscopy, diagnosis, and clinical considerations—which represent key stages in the pathology workflow, and 13 narrow categories focusing on specific aspects within each stage (Fig. 1 B). To construct a comprehensive instructional dataset, we began by generating two open-ended and two closed-ended QA pairs for each report within each specific category, taking advantage of the rich content found in the majority of TCGA reports. Then, some QA pairs were excluded, such as those lacking corresponding category information. Regarding the train/test split, it is worth noting that the WSI-report datasets from TCGA include 2 types: (a) one report linked to multiple WSIs, and (b) one report linked to a single WSI. For type (a), where specific diagnostic details may not align perfectly with each WSI, we include all WSIs in the training set to introduce some “noisy data”, which can enhance model robustness. For type (b), 80% of WSIs are allocated to the training 

set and 20% to the test set. Finally, there are 4,181 WSIs for training and 734 WSIs for testing. Consequently, we construct a large-scale training set named SlideInstruction, comprising 4,181 WSI captions and 175,753 instructionfollowing VQA pairs across 13 narrow categories. 

**SlideBench** To systematically evaluate the performance of SlideChat, we include the remaining 734 WSI captions from the test set, along with a substantial number of closed-set VQA pairs, to establish a comprehensive evaluation benchmark. First, we construct a set named SlideBench-Caption based on the WSI-Caption data to evaluate the model’s ability to generate accurate and coherent descriptions of whole slide images. Secondly, we construct SlideBench-VQA (TCGA) based on closed-set VQA pairs along with test WSIs, aiming to evaluate various aspects of model performance. As shown in Fig. 3 (B), to improve the quality of the testing benchmarks, we employ four advanced large language models, including GPT-4 [2], InternLM2- 

Chat-7B [6], Qwen-7B-Chat [5], and DeepSeek-7B-Chat, to filter closed-set VQAs by predicting answers based solely on the question text. Any questions for which at least three of these models provided correct answers are subsequently excluded. Following this automated filtering, five expert pathologists are invited to review and amend the remaining questions. The review process is guided by the following criteria: (1) Whether the correct answer necessitates image interpretation; (2) Whether the question and its corresponding answer are logically and coherently structured; and (3) Whether the question aligns appropriately with the designated broad and narrow categories. VQA pairs failing to meet these criteria are excluded by the pathologists. Consequently, the SlideBench-VQA (TCGA) comprises 7,827 VQAs across 13 categories, with some examples illustrated in Fig. 3 C. Additionally, we incorporate the in-the-wild Early Breast Cancer Core-Needle Biopsy (BCNB) WSI dataset [30], which encompasses a diverse patient population and a variety of clinical task labels, to enhance the test set benchmark and more comprehensively assess the model’s generalization capabilities. In detail, we convert the BCNB dataset into a VQA format by rephrasing the classification objectives into a specific template as questions, while transforming the original multi-class labels into selectable options, and integrating it into SlideBench as an external subset, named SlideBench-VQA (BCNB). This dataset comprises 7,247 VQA pairs from 1,058 patients, specifically designed to evaluate SlideChat’s zero-shot generalization capability across 7 distinct classification tasks. 

### **3.3. Two-stage Training** 

**Stage 1: Cross-Domain Alignment.** SlideChat adopts a two-stage training approach (see Fig. 2 B). In the first stage, the primary objective is to align the LLM’s word embeddings with the visual features extracted from whole slide images. This alignment enables the LLM to interpret visual representations from the slide-level encoder, facilitating the effective utilization of the intricate features within the slides. During this stage, SlideChat is trained to generate descriptive captions using 4.2K WSI-caption pairs from SlideInstruction. Specifically, only the slide-level encoder and projection matrix are updated, while the patch-level encoder and LLM weights remain fixed. 

**Stage 2: Visual Instruction Learning.** In the second stage, we focus on visual question-answering tasks to train the model to accurately respond to domain-specific questions concerning whole slide images. During this phase, the model develops the ability to handle a broad range of multimodal instructions, enabling it to generate answers by effectively integrating both visual and textual information. For example, the model must perform various pathology tasks, such as describing the extent of tumor invasion or assess- 

ing the degree of cellular differentiation. To accomplish this, we utilize 176K WSI VQAs from SlideInstruction in the second training stage, allowing the slide encoder, projection layer, and large language model components to be fully trainable to ensure comprehensive adaptability. This approach significantly enhances the model’s capability to handle diverse pathology-related tasks, thereby increasing its effectiveness in real-world clinical and research settings. 

## **4. Experiment** 

We conducted the following experiments to evaluate three key aspects of SlideChat: (1) its whole slide image captioning capability, which assesses proficiency in generating descriptive captions that accurately summarize the critical pathological features of a WSI; (2) its visual questionanswering (VQA) ability across various complex pathological scenarios and its generalizability in zero-shot settings; and (3) SlideChat’s ability to process gigapixel WSIs, capturing both essential global context and intricate details, thereby enhancing its performance compared to patch-level MLLMs. For WSI captioning baselines, we benchmark against MI-Gen [7], a state-of-the-art method specifically designed for this task. Given that existing MLLMs cannot handle the gigapixel scale of whole slide images, we establish baseline comparisons using two approaches: (1) randomly selecting 30 patches from each WSI and inputting them into MLLMs (e.g., GPT-4o [2], LLaVA-Med [17], MedDr [17], Quilt-LLaVA [23]), followed by a majority voting scheme to generate slide-level predictions; and (2) directly inputting a WSI thumbnail, resized to 1024×1024 pixels, referred to as Slide (T), into the MLLMs. For VQA tasks, we further evaluate performance by comparing against random prediction baselines and text-only models, thereby assessing the incremental contribution of visual information. Unless otherwise specified, SlideChat is configured with the patch-level encoder CONCH [20], the slidelevel encoder LongNet [12], and utilizes the Qwen2.5-7BInstruct [33] as LLM. 

Table 1. Captioning performance across different methods on SlideBench-Caption. 

|Methods|Input|BLEU-1|BLEU-2|BLEU-3|BLEU-4|Rouge-L|<sup>GPT</sup><br>score|
|---|---|---|---|---|---|---|---|
|GPT-4o|Patch|0.16|0.03|0.01|0.01|0.13|1.54|
|GPT-4o|Slide (T)|0.10|0.03|0.01|0.01|0.11|1.03|
|Quilt-LLaVA|Patch|0.30|0.19|0.11|0.05|0.18|3.87|
|Quilt-LLaVA|Slide (T)|0.23|0.09|0.04|0.01|0.16|1.89|
|MI-Gen|Slide|0.37|0.24|0.15|0.10|0.25|4.14|
|SlideChat|Slide|0.37|0.21|0.12|0.08|0.24|4.11|



**SlideBench-Caption** We report BLEU, ROUGE, and GPT scores to evaluate caption generation performance in 

Table 2. VQA performance with different methods. 

|Methods|Input|Sl<br>|ideBench-VQ<br>|A (TCGA)<br>|S<br>|lideBench-VQA<br>(BCNB)|WSI-VQA<sup>*</sup>|
|---|---|---|---|---|---|---|---|
|||Microscopy|Diagnosis|Clinical|Overall|||
|Random|Tt|24.44|24.91|26.44|25.02|24.40|24.14|
|GPT-4|ex|38.28|29.09|45.00|37.25|0|18.60|
|GPT-4o||62.89|46.69|66.77|57.91|41.43|30.41|
|MedDr|Ph|73.30|57.78|74.25|67.70|33.67|54.36|
|LLaVA-Med|atc|47.34|32.78|47.96|42.00|30.1|26.31|
|Quilt-LLaVA||57.76|35.96|53.07|48.07|32.19|44.43|
|GPT-4o||38.28|23.10|43.42|34.07|0|14.03|
|MedDr|Slid T|70.48|52.47|72.80|64.25|35.48|50.95|
|LLaVA-Med|e ()|45.82|27.58|40.84|37.39|0|18.79|
|Quilt-LLaVA||49.12|26.97|44.75|39.39|41.55|35.40|
|SlidCht|Slid|87.64|73.27|84.26|81.17|54.14|60.18|
|ea|e|(+14.34)|(+15.49)|(+10.01)|(+13.47)|(+12.59)|(+5.82)|



Tab. 1. For the GPT score, we use GPT-4 to assess the similarity between the generated captions and the ground truth, providing an overall score on a scale of 1 to 10, with higher scores indicating better performance. When utilizing patchlevel inputs, GPT-4o generates individual descriptions for each patch, which are subsequently integrated to create the final slide-level caption. However, this approach yields poor performance, as evidenced by a BLEU-1 score of 0.16 and a GPT-score of 1.54. These results suggest that the patchbased method fails to adequately capture the broader context necessary for accurate WSI captioning. When the WSI thumbnail of size 1024×1024 pixels is used as input to GPT4o, performance decreases further, with a BLEU-1 score of 0.10 and a GPT-score of 1.03. This suggests that while the thumbnail offers a global view of the slide, it may lack the resolution and detail necessary for generating precise and informative captions. In contrast, MI-Gen, a model specifically designed for WSI captioning, demonstrates significantly superior performance across all metrics, achieving a BLEU-1 score of 0.37 and a GPT score of 4.14. Similarly, SlideChat, shows competitive results with a BLEU-1 score of 0.37 and a GPT score of 4.11. These outcomes highlight SlideChat’s ability to effectively integrate both local and global information from the slides and confirm its efficiency in describing whole-slide images. 

**SlideBench-VQA (TCGA)** We further evaluate SlideChat’s overall performance on the multiple-choice VQA benchmark. The results on SlideBench-VQA (TCGA) in Tab. 2, compare different methods across three categories: microscopy, diagnosis, and clinical, along with an overall performance score. Random selection achieves an overall score of 25.02% accuracy, serving as a baseline for answer distribution but demonstrating poor performance. While GPT-4, relying solely on text input, 

outperforms random predictions, it continues to struggle with accurately answering questions. When GPT-4o incorporates patch-level inputs, its performance improves markedly, reaching a score of 57.91% and underscoring the crucial role of detailed visual data. However, using a WSI thumbnail results in a lower score of 34.07%, as the reduced detail restricts its ability to deliver precise answers. MedDr performs well, achieving an overall score of 67.70% with patch-level inputs, though this drops slightly to 64.25% when using the slide thumbnail due to the loss of fine visual details. SlideChat outperforms all other methods, attaining a leading overall accuracy of 81.17%, excelling across all categories and significantly surpassing the competition. Even in more fine-grained pathological scenarios, as depicted in the left portion of Fig. 4, SlideChat remains the top-performing model across 13 tasks, particularly in areas such as cytomorphological characteristics, histopathological changes, disease detection, disease classification, and staging and grading, which require the identification of complex pathological visual features. Compared with baselines taking some patches or slide thumbnail as inputs, SlideChat has the capability to analyze a significantly greater number of pathological features with enhanced detail, effectively capturing both localized features and overarching global patterns, allowing SlideChat to provide more accurate and nuanced insights into pathological variations. 

**SlideBench-VQA (BCNB)** In the zero-shot VQA setting, SlideChat significantly outperforms all other models, achieving the highest overall score of 54.14% on SlideBench-VQA (BCNB). It particularly excels in identifying tumor types, far surpassing other baselines in this task. This performance highlights SlideChat’s generalization capability across a wide range of tasks. When taking patches 



<!-- Start of picture text -->
DiseaseectonAHistopathologicalChangesCharacins~cytomerphologicalCheraciristes StatusPR- ca _n_ Fandom tet)<br>s<br>ClassificationDisease /SC SS\\ CharacteristicsTumor HERZ / \suus ee)Fr)<br>Staging | / \ \ \ Tissue Architecture ‘Status M GPT uavaadoendo (thuenbeal)<br>| | andarrangement Tumor — UavAted unten<br>\ |} 1YP wet ten<br>Grading \ / rearowic ep » one<br>\ Ya Expression o y, et pn)<br>Diterenat Risk ~ Molecular Lava tums<br>Treatment Biomarker<br>PaoneHE Guidanceaims’ Analysis.Blmattor Histologicalistoage Subtype ct<br><!-- End of picture text -->

Figure 4. Accuracy on different tasks in SlideBench-VQA (TCGA) (left) and SlideBench-VQA (BCNB) (right). 

as inputs, GPT-4o outperforms both MedDr and LLaVAMed, achieving a score of 41.43%, though it still falls short of SlideChat by 12.59%. Notably, GPT-4o and LLaVA-Med performed very poorly, achieving a score of zero across all tasks when evaluated using slide thumbnails from this testing set. MedDr and Quilt-LLaVA both achieve moderate results, with MedDr scoring 33.67% on patches and 35.48% on thumbnails, while Quilt-LLaVA attains 32.19% on patches and 41.55% on thumbnails. This outcome highlights that, for complex WSIs, relying solely on relatively sufficient visual features is inadequate for effectively supporting a range of tasks. In the more fine-grained pathological tasks of the BCNB benchmark, SlideChat attains stateof-the-art performance on 5 out of 7 tasks (Fig. 4), further demonstrating the effectiveness of SlideChat. 

**WSI-VQA**<sup>*****</sup> We also curated a subset of closed-set VQA pairs from the public WSI-VQA [8] dataset, referred to as WSI-VQA<sup>*</sup> , which exclusively consists of samples that were not in the training set, to evaluate the model’s performance. SlideChat demonstrates the highest performance with a score of 60.18. Although MedDr performs well with both patch inputs (54.36%) and slide thumbnail inputs (50.95%), it still falls short compared to SlideChat. GPT-4o struggles significantly, especially with slide thumbnails, scoring only 14.03%, which highlights the limitations of using lower-resolution inputs. SlideChat leverages both detailed local information and broader context to process WSIs, making it the most effective model for this benchmark. This further emphasizes its superior capability in handling whole-slide data for VQA tasks. 

**Ablation** We performed ablation experiments from different perspectives as follows: a) _Large Language Model Comparison_ : Firstly, we compare the performance of several large language models, each with a parameter scale of approximately 7 billion, including Vicuna-7B-v1.5 [10], Phi-3-Mini-4k-Instruct [1], Qwen1.5-7B-Chat [5], Llama3- 

Table 3. Performance Comparison of LLMs and Slide Encoder on WSI Captioning and VQA Tasks. 

|LLMs|Slide<br>Encoder|<sup>Caption</sup>|VQA<br>(TCGA)|VQA<br>(BCNB)|<sup>WSI-VQA*</sup>|
|---|---|---|---|---|---|
|Vicuna-7B-v1.5|✓|3.28|41.43|41.43|31.98|
|Phi-3-Mini-4k-Instruct|✓|2.66|79.93|43.92|60.18|
|Qwen1.5-7B-Chat|✓|2.92|77.63|44.07|56.89|
|Llama3-8B-Instruct|✓|3.30|78.78|42.82|55.25|
|Internlm2-Chat-7B|✓|3.30|79.10|52.13|56.76|
|Qwen2.5-3B-Instruct|✓|3.40|80.32|45.79|56.38|
|Qwen2.5-14B-Instruct|✓|3.39|82.14|51.57|60.94|
|Qwen2.5-7B-Instruct|✓|4.11|81.17|54.14|60.18|
|Qwen2.5-7B-Instruct|_×_|3.95|81.21|45.49|59.67|



8B-Instruct [4], InternLM2-Chat-7B [6], and Qwen2.5-7BInstruct [33]. Specifically, we measured their performance on the SlideBench-Caption using the GPT-score and their accuracy on three VQA datasets. As shown in Tab. 3, the results demonstrate that SlideChat, powered by the Qwen2.5-7B-Instruct model, achieved the highest performance across all tasks, particularly excelling in the captioning task. These findings underscore the significant potential of developing SlideChat with the Qwen2.5-7B-Instruct model. Utilizing the Qwen2.5 model, we further evaluated models of varying scales and discovered that larger models generally exhibited superior performance, particularly the 7B and 14B parameter models. While these two models showed comparable performance across the three benchmarks, the 14B model surpassed the 7B model by 2.57% on the SlideBench-VQA (TCGA). Given computational resource constraints, SlideChat uses the 7B model by default to achieve the best balance between performance and resource efficiency. b) _Slide-level encoder effectiveness_ : We investigate the effectiveness of the slide-level encoder by initially removing it from SlideChat and employing a twostage training approach. In the first stage, we exclusively trained the projection layers. However, this approach failed 



<!-- Start of picture text -->
A; ss : B ope Si) “ a<br>SalactSrtomorpclX Calsthe wahineeasednilesroeyopasmerata,m o stgiea!chrec acurate d e scriptionrstes rom ofthe moos hypercwomatc .5 Whichenue sharpe histological change is typically seenyein the O provided 53<br>hue, end promnert niceok ° 8: Accurnaton of aryl deposts,<br>B Cele wah ror nat and abundantcon ©. Ouse yrehou pera<br>6 als win intl nate sbroraites D nevesitwah caeeston<br>D. Gels wih enlarge, clear rll and srl clea 7<br>5 6<br>[i $< A.Bee<br>3 | tiscemematcnase androse 2 cn spainnd<br><!-- End of picture text -->

Figure 5. Interpretability and visualization. We identify the top five patch tokens with the highest attention scores associated with the output text responses. 

to reduce training loss or generate coherent text effectively, likely due to the difficulty of learning the complex visual features of WSIs without the slide-level encoder. When using a frozen LLM for complex text generation, a simple projection proved inadequate for effectively integrating and aligning visual and textual features. Subsequently, we consider combining data from both stages and training SlideChat without the slide-level encoder by simultaneously updating both the projection layers and the LLM. Under this paradigm, performance on SlideBench-VQA (TCGA) and WSI-VQA<sup>*</sup> , which share the same distribution as the training set, was comparable to the two-stage training configuration with the slide-level encoder. However, a significant decline is observed when evaluating SlideBench-VQA (BCNB), which originates from a different domain; overall performance dropped by over 10% (Tab. 3), indicating a substantial reduction in the model’s generalization ability. Therefore, we recommend incorporating a slide-level encoder to capture the complex visual features of whole slide, as it is particularly effective for cross-domain alignment and enhances the model’s generalization performance. 

five important patches identified by the model closely corresponded with the features described in SlideChat’s output. Our extraction method retrieves attention weights for patch tokens from each generated token, averaging them across layers and heads. We then identify the top five patch tokens with the highest attention weights for further analysis. For example, in Fig. 5 (A), the highlighted patches clearly emphasized regions exhibiting an increased nuclear-to-cytoplasmic ratio, hyperchromatic nuclei, and prominent nucleoli. Similarly, in Fig. 5 (B), the selected patches demonstrated areas with dense collagen deposition and reduced cellularity, as detailed in the model’s response. This alignment between the highlighted image regions and the textual outputs significantly enhances the model’s interpretability, providing increased confidence that it accurately captures and assesses relevant histopathological features. Such consistency deepens our understanding of the model’s reasoning processes regarding pathological slides and underscores the potential for integrating these AI systems into clinical workflows with greater assurance. 

## **5. Conclusion** 

**Interpretability** Despite SlideChat demonstrating promising results, concerns remained regarding the model’s perception of large pathological slides. To further assess the model’s interpretability, we calculated the correlation between the text output and specific image patches, thereby obtaining patch-level attention scores. By identifying the most significant patches, we gained insights into the precise areas the model focused on during response generation. Highlighting the most relevant regions of the tissue slides not only enhances transparency and bolsters the reliability of the model’s outputs but also assists pathologists by directing attention to critical areas requiring closer scrutiny. Ultimately, such interpretability is essential for fostering trust in AI-assisted diagnostics and enhancing the precision and efficiency of clinical evaluations. As shown in Fig. 5, we are pleased to observe that the top 

In this work, we present SlideChat, the first vision-language assistant capable of understanding gigapixel whole-slide images. Furthermore, we creat SlideInstruction, a largest comprehensive WSI instruction-following dataset to develop SlideChat, as well as SlideBench, a multi-modal benchmark designed to evaluate SlideChat across diverse scenarios. SlideChat demonstrates excellent chat abilities and achieves state-of-the-art performance on 18 tasks. We bridge the gap between MLLMs and pathology images at the whole-slide level with SlideChat, and we believe this represents an advancement in the field of general medical artificial intelligence (GMAI). 

**Acknowledgement** The authors acknowledge the funding provided by the National Key Research and Development Program of China (No.2022ZD0160102). 

## **References** 

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. _arXiv preprint arXiv:2404.14219_ , 2024. 7 

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 4, 5 

- [3] Faruk Ahmed, Andrew Sellergren, Lin Yang, Shawn Xu, Boris Babenko, Abbi Ward, Niels Olson, Arash Mohtashamian, Yossi Matias, Greg S Corrado, et al. Pathalign: A vision-language model for whole slide images in histopathology. _arXiv preprint arXiv:2406.19578_ , 2024. 2, 4 

- [4] AI@Meta. Llama 3 model card. 2024. 7 

- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. _arXiv preprint arXiv:2309.16609_ , 2023. 5, 7 

- [6] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. _arXiv preprint arXiv:2403.17297_ , 2024. 5, 7 

- [7] Pingyi Chen, Honglin Li, Chenglu Zhu, Sunyi Zheng, and Lin Yang. Mi-gen: Multiple instance generation of pathology reports for gigapixel whole-slide images. _arXiv preprint arXiv:2311.16480_ , 2023. 2, 5, 4 

- [8] Pingyi Chen, Chenglu Zhu, Sunyi Zheng, Honglin Li, and Lin Yang. Wsi-vqa: Interpreting whole slide images by generative visual question answering. _arXiv preprint arXiv:2407.05603_ , 2024. 2, 7, 4 

- [9] Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Andrew H Song, Bowen Chen, Andrew Zhang, Daniel Shao, Muhammad Shaban, et al. Towards a general-purpose foundation model for computational pathology. _Nature Medicine_ , 30(3):850–862, 2024. 1 

- [10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ , 2(3):6, 2023. 2, 7 

- [11] XTuner Contributors. Xtuner: A toolkit for efficiently finetuning llm, 2023. 4 

- [12] Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, and Furu Wei. Longnet: Scaling transformers to 1,000,000,000 tokens. _arXiv preprint arXiv:2307.02486_ , 2023. 3, 5 

- [13] Zhengrui Guo, Jiabo Ma, Yingxue Xu, Yihui Wang, Liansheng Wang, and Hao Chen. Histgen: Histopathology re- 

port generation via local-global feature encoding and crossmodal context interaction. _arXiv preprint arXiv:2403.05396_ , 2024. 2, 4 

- [14] Sunan He, Yuxiang Nie, Zhixuan Chen, Zhiyuan Cai, Hongmei Wang, Shu Yang, and Hao Chen. Meddr: Diagnosis-guided bootstrapping for large-scale medical vision-language learning. _arXiv preprint arXiv:2404.15127_ , 2024. 2 

- [15] Mahdi S Hosseini, Babak Ehteshami Bejnordi, Vincent Quoc-Huy Trinh, Lyndon Chan, Danial Hasan, Xingwen Li, Stephen Yang, Taehyo Kim, Haochen Zhang, Theodore Wu, et al. Computational pathology: a survey review and the way forward. _Journal of Pathology Informatics_ , page 100357, 2024. 1 

- [16] Carolyn Hutter and Jean Claude Zenklusen. The cancer genome atlas: creating lasting value beyond its data. _Cell_ , 173(2):283–285, 2018. 2 

- [17] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large languageand-vision assistant for biomedicine in one day. _Advances in Neural Information Processing Systems_ , 36, 2024. 2, 5 

- [18] Hao Li, Ying Chen, Yifei Chen, Rongshan Yu, Wenxian Yang, Liansheng Wang, Bowen Ding, and Yuchen Han. Generalizable whole slide image classification with fine-grained visual-semantic interaction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11398–11407, 2024. 2 

- [19] Ming Y Lu, Tiffany Y Chen, Drew FK Williamson, Melissa Zhao, Maha Shady, Jana Lipkova, and Faisal Mahmood. Aibased pathology predicts origins for cancers of unknown primary. _Nature_ , 594(7861):106–110, 2021. 2 

- [20] Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, et al. A visuallanguage foundation model for computational pathology. _Nature Medicine_ , 30(3):863–874, 2024. 3, 5 

- [21] Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Melissa Zhao, Aaron K Chow, Kenji Ikemura, Ahrong Kim, Dimitra Pouli, Ankush Patel, et al. A multimodal generative ai copilot for human pathology. _Nature_ , pages 1–3, 2024. 1, 2, 4 

- [22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 2 

- [23] Mehmet Saygin Seyfioglu, Wisdom O Ikezogwo, Fatemeh Ghezloo, Ranjay Krishna, and Linda Shapiro. Quilt-llava: Visual instruction tuning by extracting localized narratives from open-source histopathology videos. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13183–13192, 2024. 1, 2, 5, 4 

- [24] Zhuchen Shao, Yang Chen, Hao Bian, Jian Zhang, Guojun Liu, and Yongbing Zhang. Hvtsurv: Hierarchical vision transformer for patient-level survival prediction from whole 

slide image. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 2209–2217, 2023. 2 

_IEEE/CVF International Conference on Computer Vision_ , pages 21485–21494, 2023. 2 

- [25] Andrew H Song, Guillaume Jaume, Drew FK Williamson, Ming Y Lu, Anurag Vaidya, Tiffany R Miller, and Faisal Mahmood. Artificial intelligence for digital and computational pathology. _Nature Reviews Bioengineering_ , 1(12): 930–949, 2023. 1 

- [26] Joey Spronck, Thijs Gelton, Leander van Eekelen, Joep Bogaerts, Leslie Tessier, Mart van Rijthoven, Lieke van der Woude, Michel van den Heuvel, Willemijn Theelen, Jeroen van der Laak, et al. nnunet meets pathology: bridging the gap for application to whole-slide images and computational biomarkers. In _Medical Imaging with Deep Learning_ , 2023. 2 

- [27] Yuxuan Sun, Chenglu Zhu, Sunyi Zheng, Kai Zhang, Lin Sun, Zhongyi Shui, Yunlong Zhang, Honglin Li, and Lin Yang. Pathasst: A generative foundation ai assistant towards artificial general intelligence of pathology. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 5034– 5042, 2024. 1, 2 

- [28] Eugene Vorontsov, Alican Bozkurt, Adam Casson, George Shaikovski, Michal Zelechowski, Kristen Severson, Eric Zimmermann, James Hall, Neil Tenenholtz, Nicolo Fusi, et al. A foundation model for clinical-grade computational pathology and rare cancers detection. _Nature Medicine_ , pages 1–12, 2024. 1 

- [29] Xiyue Wang, Junhan Zhao, Eliana Marostica, Wei Yuan, Jietian Jin, Jiayu Zhang, Ruijiang Li, Hongping Tang, Kanran Wang, Yu Li, et al. A pathology foundation model for cancer diagnosis and prognosis prediction. _Nature_ , pages 1–9, 2024. 2 

- [30] Feng Xu, Chuang Zhu, Wenqi Tang, Ying Wang, Yu Zhang, Jie Li, Hongchuan Jiang, Zhongyue Shi, Jun Liu, and Mulan Jin. Predicting axillary lymph node metastasis in early breast cancer using deep learning on primary tumor biopsy slides. _Frontiers in oncology_ , 11:759007, 2021. 5 

- [31] Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Cliff Wong, Zelalem Gero, Javier Gonz´alez, Yu Gu, et al. A whole-slide foundation model for digital pathology from real-world data. _Nature_ , pages 1–8, 2024. 1, 3, 4 

- [32] Yingxue Xu, Yihui Wang, Fengtao Zhou, Jiabo Ma, Shu Yang, Huangjing Lin, Xin Wang, Jiguang Wang, Li Liang, Anjia Han, et al. A multimodal knowledge-enhanced whole-slide pathology foundation model. _arXiv preprint arXiv:2407.15362_ , 2024. 2 

- [33] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ , 2024. 5, 7 

- [34] Heyu Zhang, Yan He, Xiaomin Wu, Peixiang Huang, Wenkang Qin, Fan Wang, Juxiang Ye, Xirui Huang, Yanfang Liao, Hang Chen, et al. Pathnarratives: Data annotation for pathological human-ai collaborative diagnosis. _Frontiers in Medicine_ , 9:1070072, 2023. 4 

- [35] Fengtao Zhou and Hao Chen. Cross-modal translation and alignment for survival analysis. In _Proceedings of the_ 

# **SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding** 

# Supplementary Material 

## **6. SlideInstruction and SlideBench** 

Table 5. The number of VQA corresponding to each category in SlideBench-VQA (TCGA). 

### **6.1. Data Source** 

In this section, we present the sources of the constructed SlideInstruction and SlideBench, which are derived from ten TCGA datasets as well as the BCNB challenge dataset. The Tab. 4 provides a detailed overview of the specific number of WSIs. 

Table 4. Datasets statistics 

|Dataset|WSIs|Report|Organ|Purpose|
|---|---|---|---|---|
|TCGA-BRCA|1068|✓|Breast|Train, Test|
|TCGA-LGG|783|✓|Brain|Train, Test|
|TCGA-GBM|513|✓|Brain|Train, Test|
|TCGA-LUAD|506|✓|Lung|Train, Test|
|TCGA-LUSC|474|✓|Lung|Train, Test|
|TCGA-HNSC|464|✓|Head and Neck|Train, Test|
|TCGA-BLCA|424|✓|Bladder|Train, Test|
|TCGA-COAD|419|✓|Colon|Train, Test|
|TCGA-READ|157|✓|Rectum|Train, Test|
|TCGA-SKCM|107|✓|Skin|Train,Test|
|BCNB|1058|✗|Breast|Test|



### **6.2. Data Statistics** 

We have compiled statistics on the number of VQA instances for each category within SlideBench VQA (TCGA) in Tab. 5. Each subcategory contains over 500 VQA instances, ensuring a robust representation across all areas, which supports comprehensive model evaluation and facilitates in-depth performance analysis. We provide an overview of the sample sizes and detailed original label information for the seven classification tasks within the BCNB dataset in Tab. 6. 

### **6.3. Curation Scope and Prompt** 

In this section, we illustrate the various dimensions of VQAs in SlideInstruction and SlideBench, ensuring comprehensive coverage of diverse pathological scenarios. This includes 3 broad categories and 13 narrow categories. Below are the contents for each category, which help to delineate their scope and meaning, thereby enabling GPT to extract high-quality question-answer pairs more effectively. 

|Broad Category|Narrow Catgory|Number|
|---|---|---|
||Tissue Architecture and Arrangement<br>Tumor Characteristics|696<br>562|
|Microscopy|Cytomorphological Characteristics|601|
||Histopathological Changes|633|
||Disease Detection|581|
||Disease Classification|532|
|Dianosis|Staging|671|
|g|Grading|601|
||Differential Diagnosis|586|
||Treatment Guidance|597|
||Biomarker Analysis|502|
|Clinical|Risk Factors|591|
||Prognostic Assessment|674|



Table 6. The number and options of VQA corresponding to each task in SlideBench-VQA (BCNB). 

|Task|Number|Option|
|---|---|---|
|ER Status|1058|Postive / Negative|
|HR Status|1058|Postive / Negative|
|HER2 Status|1058|Postive / Negative|
|HER2 Expression|1058|0 / 1+ / 2+ / 3+|
|Histological Grading|926|1 / 2 / 3|
|Molecular Subtype|1058|Luminal A / Luminal B<br>/ HER2(+)/ Triple negative|
|||Invasive ductal carcinoma|
|Tumor Type|1058|/ Invasive lobular carcinoma<br>/ Other Type|



#### **6.3.1. Scope** 

**Microscopy** This category involves assessing the ability to generate microscopy descriptions of pathology images, focusing on clinically relevant features: 

- Tissue Architecture and Arrangement: Questions in this category should evaluate the understanding of overall tissue structure and spatial organization within a histological section. 

- Cytomorphological Characteristics: These questions should focus on the detailed description of individual cell morphology, including nuclear and cytoplasmic features. 

- Tumor Characteristics: Questions under this category should assess the ability to identify and describe features specific to tumors, such as tumor differentiation, inva- 



<!-- Start of picture text -->
Diagnosis<br>Microscopy Clinical<br>Upon reviewing the WSI, can you identify<br>any high-grade dysplasia or carcinoma in<br>the distal ureter specimens?<br>Which histopathological change visible in the provided slide image is indicative of necrosis in breast tissue? A. Granulomatous inflammation  B. Central eosinophilic granular material with ghost cells  C. Dense fibrosis and scarring D. Lymphocytic infiltration around ducts Which cytomorphological feature is characteristic in this WSI?  A. Prominent nucleoli and high nuclear-cytoplasmic ratio B. Cytoplasmic vacuolization  C. Cytoplasmic vacuolization  D. Multinucleated giant cells Cytomorphological Characteristics A. Yes, in both ureters B. Yes, only in the left distal ureter C. Yes, only in the right distal ureter  D. No, in both ureters DetectionDisease  What histological classification can be determined for the neoplasm observed in the bladder tissue slide? A. Non-invasive papillary urothelial carcinomaB. Invasive adenocarcinoma C. Invasive squamous cell carcinoma  D. High-grade invasive urothelial carcinoma ClassificationDisease  Examine the histopathological features present in the supplied whole slide image to identify the biomarker that can influence treatment options. A. HER2 C. CD20  Biomarker  B. PD-L1 D. BRAF Analysis Based on the provided image, which factor is most indicative of a poor prognosis?A. 1p/19q co-deletion   B. High MIB-1 labeling index<br>HistopathologicalChanges C. Low mitotic figures  D. Absence of necrosis<br>Prognostic<br>Based on the histological features in  Assessment<br>What feature would you expect to see in the tissue in the image? A. Intact epidermis with preserved dermal structures.  this whole slide image, what grade is assigned to the oligoastrocytoma? A. Grade I C. Grade III  B. Grade II  D. Grade IV Which treatment would likely be avoided for a patient with the histopathological slide showing HER2 negative status?A. Trastuzumab  B.<br>B. Disruption of the epidermal surface with underlying necrosis. Grading ChemotherapyC. Hormonal therapy       D. Surgery<br>C. Presence of papillary dermis with fibroplasia.<br>D. Well-ordered collagen bundles in the dermis. Tissue Architecture  Examine the histological features in the provided image and identify the brain tumor  Treatment Guidance<br>Which feature is indicative of the invasive ductal carcinoma based on this image? A. Well-formed glandular structures B. High mitotic index and prominent nucleoli C. Lobular arrangement of cells with mild nuclear atypiaD. Low nuclear-to-cytoplasmic ratio and Arrangement that can mimic an oligodendroglioma due to its similar appearance. A. Glioblastoma                B. Medulloblastoma C. Diffuse astrocytoma D. Central neurocytoma Differential  Which feature observed in the tissue slide is often associated with an increased risk of cancer?<br>Tumor  What pathologic stage describes the extent of tumor invasion?  Diagnosis A. High-vegetable diet C. High alcohol consumption  B. Irregular physical activity D. Low body mass index<br>Characteristics A. pT1 B. pT2a  C. pT2b  D. pT3<br>Risk Factors<br>Staging<br><!-- End of picture text -->

Figure 6. Examples of generated structural VQAs in pathology across Microscopy, Diagnosis, and Clinical scenarios. 

sion, and specific patterns associated with different types of tumors. 

- Histopathological Changes: This category should include questions that evaluate the recognition and description of pathological changes in tissue, such as necrosis, inflammation, fibrosis, and other alterations that indicate disease processes. 

**Diagnosis** This category tests the ability of models to suggest a reasonable diagnosis based on histological images and relevant clinical context: 

- Disease Detection: Questions in this category should evaluate the model’s ability to identify the presence or absence of a disease based on histological features and clinical information. 

- Disease Classification: These questions should focus on distinguishing between different types or subtypes of diseases, assessing the model’s capability to classify conditions accurately based on morphological and histopathological criteria. 

- Grading: Questions under this category should assess the model’s ability to determine the grade of a disease, particularly tumors, based on the degree of differentiation and cellular atypia observed in histological images. 

- Staging: This category should include questions that evaluate the ability to assign a stage to a disease, particularly in oncology, by assessing the extent of disease 

- spread and involvement of surrounding tissues or organs. 

- Differential Diagnosis: Questions should test the model’s ability to provide a differential diagnosis, distinguishing between multiple potential conditions that may present with similar histological and clinical features. 

**Clinical** This category tests the ability of models to retrieve and apply clinically relevant background knowledge about diseases: 

- Treatment Guidance: Questions in this category should assess the model’s ability to recommend appropriate treatment options based on the disease in question, considering factors such as disease stage, patient demographics, and any specific clinical guidelines. 

- Prognostic Assessment: These questions should focus on evaluating the model’s ability to predict the likely course and outcome of a disease, including survival rates, potential complications, and long-term outcomes based on clinical and pathological data. 

- Risk Factors: Questions under this category should test the model’s knowledge of risk factors associated with specific diseases, including genetic, environmental, and lifestyle factors that may influence disease development or progression. 

- Biomarker Analysis: This category should include questions that evaluate the ability to identify and interpret biomarkers relevant to the diagnosis, prognosis, or treat- 

ment of diseases, emphasizing their role in personalized medicine and targeted therapy. 

Table 8. Question-Answers generation prompts, including system prompt, general prompt and objective prompt. 

#### **6.3.2. Designed Prompts** 

#### **Report Cleaning Prompt.** 

**Report Cleaning Prompt.** The prompt used to clean up the report from the original TCGA report is represented in Tab. 7. This process effectively eliminates extraneous noise from the report, thereby establishing a more solid foundation for caption and QA pairs generation. 

Table 7. Prompts for report clean and caption generation. 

**[Report Clean Prompt]** This is the content from the pathology report. Please remove some redundant irrelevant information from the original report, such as technical details of pathology department procedures, Symbols unrelated to the pathological report, specimen handling and processing information, redundant administrative or legal statements, and some repeated information. Show me the cleaned report content. 

**[Caption Generation Prompt]** Based on the above pathological report content, generate a detailed paragraph that summarizes the essential pathological findings. The paragraph should include key information such as the diagnosis, tumor characteristics, margin status, lymph node involvement, and other relevant pathological findings. The summary should not mention the source being a report and should exclude any specific sizes or measurements. The paragraph should be written in a clear and cohesive manner, covering all important points without unnecessary details. 

**Caption Generation Prompt.** The prompt used for caption generation from the refined report is detailed in Tab. 7, ensuring that the generated caption effectively captures essential summarized information in report. 

**Question-Answers Generation Prompt.** The prompt used to extract QA from reports mainly consist of 4 parts ( _i.e._ , _<_ Cleaned Report _>_ + System Prompt + Objective Prompt + General Prompt), and the detailed content of each part is illustrated in Tab. 8 

**Label Transformation Prompt.** The prompt for transforming BCNB dataset is illustrated in Tab. 9. We employ GPT to transform individual labels into a questionanswer format based on the task type and corresponding classification labels, facilitating the testing of MLLM. For instance, in the context of a tumor type classification task, 

**[System Prompt]** You are an AI assistant proficient in digital pathology. You will receive a pathology report for whole slide images. 

**[General Prompt]** Based on the above pathological report content, your task is to use the provided information, create 2 multi-choice questions amd 2 short-answer questions for each narrow category. The design question should be able to be answered based on the content of the image. Design medical questions very carefully and only ask questions when you are sure of the answer. Answers should be specific and avoid ambiguity. When generating questions, it is necessary to indicate their broad category and narrow category. For multi-choice questions, you should (1) “question type” is “multi-choice questions”. (2) Provide the options and answer and reasoning. Provide four answer choices (A, B, C, and D), ensuring that one choice is correct and the other three are plausible but incorrect. (3) Aim to include one answer that is incorrect but very similar to the correct one to increase the difficulty level. For short-answer questions: (1) “question type” is “short-answer questions”. (2) Generating questions with different content from multiple-choice questions. For all questions: (1) Do not mention that the information source is report in “question”, “anwser”. (2) Return JSON format in “question type”: xxx, “question”: xxx, “options”: [], “answer”: xxx, “broad category”: xxx, “narrow category”: xxx for each question. The “options” section is empty for short-answer questions. 

**[Objective Prompt]** Definition of Broad Category and its corresponding Narrow Categories. “ The required broad category is Microscopy, which involves assessing the ability to generate microscopy descriptions of pathology images, focusing on clinically relevant features. For the narrow category: Tissue Architecture and Arrangement: Questions should evaluate the understanding of overall tissue structure and spatial organization within a histological section.” 

_<_ Task _>_ represents “Tumor Type”, while _<_ label 1 _>_ , _<_ label 2 _>_ , and _<_ label 3 _>_ are “Invasive ductal carcinoma”, “Invasive lobular carcinoma”, and “Other Type”, respectively, enabling the generation of relevant QA pairs. 

### **6.4. Multimodal Dataset Comparsion** 

Recently, several multimodal pathology datasets have been introduced for pathology applications. However, these datasets are often constrained in both scope and scale, as they primarily focus on either patch-level analysis or limited 

Table 9. Prompt for Converting Labels into QA Pairs 

**[Label Transformation Prompt]** Please create prompts for pathology image classification tasks concerning _<_ Task _>_ , transforming traditional labels into a multi-choice question-and-answer format. The original labels include _<_ label 1 _>_ , _<_ label 2 _>_ , ... 

available data. In contrast, our proposed SlideInstruction and SlideBench, provided as open-source resources, significantly expand the dataset size while enhancing its versatility, as shown in Tab. 10. 

Table 10. Comparisons of our datasets with other pathology datasets. 

instruction-following data for 1 epoch, with a learning rate of 0.00002. Both stages are optimized using AdamW. 

### **7.3. Ability Showcase** 

#### **7.3.1. Captioning Ability** 

The examples shown in Fig. 7 illustrate the capability of our model, SlideChat, to effectively perform whole-slide image captioning tasks. SlideChat demonstrates its proficiency in generating detailed and contextually accurate summaries for complex pathological whole-slide images, accurately capturing key clinical findings and pathological features. Whether summarizing broad findings, explaining pivotal details, or highlighting core results, SlideChat showcases an advanced understanding of whole-slide images, providing concise yet informative reports that align with clinical terminology and expectations. 

#### **7.3.2. VQA Ability** 

|Dataset|Level|Data Type|Curation Type|Scope|Number|Availability|
|---|---|---|---|---|---|---|
|PathChat [21]|Patch|Patch and Q/A pairs|Human+GPT|-|257,004|✗|
|Quilt-Instruct[23]|Patch|Patch andQ/Apairs|GPT|-|107,131|✓|
|WSI-VQA [8]|Slide|WSI and Q/A pairs|GPT|-|8,672|✓|
|PathText [7]|Slide|WSI-Caption pairs|GPT|-|9,009|✓|
|HistGen [13]|Slide|WSI-Reports pairs|GPT|-|7,753|✓|
|Prov-Path [31]|Slide|WSI-Reports pairs|GPT|-|17,383|✗|
|CR-PathNarratives [34]|Slide|WSIs with<br>annotations|Human|-|174|✗|
|PathAlign[3]|Slide|WSI-Reportspairs|Human|-|354,089|✗|
|Our SlideInstruction|Slide|WSI and Q/A pairs|GPT|13|179,935|✓|
|Our SlideBench|Slide|WSI and Q/A pairs|Human+GPT|13|15,835|✓|



## **7. Experiment** 

### **7.1. Computational Cost Analysis** 

To evaluate the computational cost of our model architecture, we measured both the inference time and GPU memory consumption throughout the entire pipeline. This pipeline includes the patch-level encoder, slide-level encoder, multimodal projector module, and large language model, all executed on an A100 GPU. After extracting the local and global features of WSIs, the average response time was within 1 second, and GPU memory consumption was approximately 27 GB. The inference time and GPU memory consumption remained well within acceptable limits for gigapixel whole slide images. 

### **7.2. Implementation Details** 

We preprocessed each WSI by segmenting it into 224 × 224 nonoverlapping patches at a 20× magnification level, excluding background regions. We implemented our model using the Xtuner [11] toolkit and trained it across two stages on 8 × NVIDIA A100 GPUs. The training process consists of an alignment phase followed by instruction fine-tuning: Stage 1: We freeze the LLM and train the Projection and Slide Encoder with WSI-caption data for 3 epochs, using a learning rate of 0.001. Stage 2: We unfreeze the LLM, Slide Encoder, and Projection, training the model on WSI 

Fig. 8 showcases the conversational examples of SlideChat, demonstrating its ability to accurately answer a range of questions based on WSIs, covering diverse aspects such as histological classifications, tumor grading, lymph node involvement, and treatment decisions. SlideChat effectively interprets complex pathological data, engages in nuanced question-and-answer exchanges, and delivers clinically relevant responses. This reflects its potential as an intelligent assistant capable of supporting pathologists in diagnostic decision-making by providing insightful, context-aware dialogue grounded in visual pathology data. 

#### **7.3.3. Comparing Model Outputs** 

Fig. 9 presents a comparative analysis of the outputs from SlideChat and other models within SlideBench. The examples illustrate SlideChat’s remarkable capacity to precisely classify tumors, identify distinct histological features, and describe the structural organization of tumor cells from WSIs. SlideChat demonstrates a unique proficiency in capturing both local and global features—seamlessly integrating detailed microscopic characteristics with broader contextual understanding to deliver accurate and clinically meaningful interpretations. In contrast, existing models are limited to processing small pathology images, often yielding ambiguous or incorrect classifications. This underscores SlideChat’s advanced capability in comprehending wholeslide images by incorporating both intricate details and a comprehensive visual perspective. 

### **7.4. Detailed Test Performance** 

#### **7.4.1. Performance on SlideBench-VQA (TCGA)** 

The results presented in the tables demonstrate a comprehensive evaluation of SlideChat’s performance on SlideBench-VQA (TCGA) in comparison to other existing models across microscopy, diagnosis, and clinical tasks. In 

||||SlideBench-VQ|A(TCGA)**M**|**icroscopy**|||
|---|---|---|---|---|---|---|---|
|Method|Input|Tissue Architecture<br>and Arrangement|<br>Tumor<br>Characteristics|Cytomor<br>Charac|phological<br>teristics|Histopathological<br>Changes|Overall|
|Random|Text|23.70|22.42|23|.63|27.80|24.44|
|GPT-4||40.83|40.28|41|.71|37.46|39.62|
|GPT-4o||65.94|66.20|60|.10|59.23|62.89|
|MedDr|Pth|75.04|75.78|70|.10|72.23|73.30|
|LLaVA-Med|ac|50.04|40.63|40|.38|56.95|47.34|
|Quilt-LLaVA||65.26|54.04|50|.66|59.55|57.76|
|GPT-4o||37.07|38.76|39|.93|37.60|38.28|
|MedDr|Slide (T)|71.58|71.27|69|.87|69.05|70.48|
|LLaVA-Med||51.80|45.02|36|.27|49.01|45.82|
|Quilt-LLaVA||53.59|45.37|43|.09|53.24|49.12|
|SlidCh|Slid|88.07|87.01|88|.02|87.36|87.64|
|eat|e|(+13.03)|(+11.23)|(+1|7.92)|(+15.13)|(+14.34)|
||||SlideBench-V<br>|QA(TCGA|)**Diagnosi**|**s**<br>||
|Method|Input|Disease<br>Detection|Disease<br>Classification|Staging|Grading|Differential<br>Diagnosis|Overall|
|Random|Tt|25.82|24.06|24.14|26.12|24.40|24.91|
|GPT-4|ex|27.12|31.07|22.27|27.45|38.70|29.09|
|GPT-4o||50.27|55.94|39.94|39.66|49.66|46.69|
|MedDr|Pth|59.11|61.11|48.66|52.97|68.83|57.78|
|LLaVA-Med|ac|37.25|28.57|30.41|20.71|47.27|32.78|
|Quilt-LLaVA||40.74|39.3|32.32|28.96|39.52|35.96|
|GPT-4o||22.95|26.76|18.06|21.06|27.82|23.10|
|MedDr|Slid|<br>54.29|56.40|48.66|43.52|61.61|52.47|
|LLaVA-Med|e (|)<br>27.87|25.19|24.07|24.96|36.18|27.58|
|Quilt-LLaVA||32.47|28.25|20.18|22.96|32.25|26.97|
|SlidCh|Slid|80.90|76.12|68.41|68.39|73.72|73.27|
|eat|e|(+21.79)|(+15.01)|(+19.75)|(+15.42)|(+4.89)|(+15.49)|



microscopy, SlideChat significantly outperforms its counterparts, achieving a notable overall accuracy improvement of 14.34 points over the nearest model. This strong performance is consistent across sub-tasks, such as tissue architecture analysis, tumor characteristics identification, and cytomorphological assessment, showcasing SlideChat’s advanced capability to analyze both detailed cellular structures and broader histopathological changes. In the diagnostic tasks, SlideChat also demonstrates superior accuracy, with an overall gain of 15.49 points, excelling in disease detection, classification, staging, grading, and differential diagnosis. The clinical analysis results further validate the model’s strength, with SlideChat outperforming other methods by 10.01 points overall, particularly excelling in treatment guidance, biomarker analysis, and risk factor assessment. These results illustrate SlideChat’s capability to seamlessly handle complex medical data and deliver reliable insights across multiple clinical and diagnostic domains, indicating its potential as a robust tool for comprehensive pathology analysis. 

#### **7.4.2. Performance on SlideBench-VQA (BCNB)** 

The evaluation of SlideChat on SlideBench-VQA (BCNB), a real-world dataset designed for zero-shot testing, further underscores its ability to generalize effectively to unseen data. SlideChat demonstrates an overall accuracy improvement of 12.71 points compared to other models, showcasing its ability to generalize well across diverse and complex breast cancer-related tasks. SlideChat’s performance is particularly strong in identifying tumor type, ER status, PR status, and HER2 status, demonstrating a nuanced understanding of critical histopathological features. Nevertheless, in the more complex tasks of HER2 Expression, Histological Grading, and Molecular Subtype classification, SlideChat still exhibits potential for improvement, highlighting specific areas that warrant further refinement to enhance its overall performance. 

|||Slide<br>|Bench-VQA(<br>|TCGA)**C**<br>|**linical**<br>||
|---|---|---|---|---|---|---|
|Method|Input|Treatment<br>Guidance|Biomarker<br>Analysis|Risk<br>Factors|Prognostic<br>Assessment|Overall|
|Random|Text|23.62|31.87|24.36|24.33|26.44|
|GPT-4||49.98|44.63|46.46|39.64|45.00|
|GPT-4o||64.18|57.99|76.99|66.64|66.77|
|MedDr|Pth|74.18|82.99|82.43|60.66|74.25|
|LLaVA-Med|ac|62.04|53.98|53.04|26.54|47.96|
|Quilt-LLaVA||64.79|40.42|63.40|43.06|53.07|
|GPT-4o||50.00|50.08|44.16|32.64|43.42|
|MedDr|Slide (T)|71.43|84.51|78.92|60.24|72.80|
|LLaVA-Med||50.50|48.01|48.90|19.88|40.84|
|Quilt-LLaVA||47.38|32.93|47.71|48.64|44.76|
|SlideChat|Slide|83.42|89.04|91.71|74.93|84.26|
|||(+9.24)|(+4.53)|(+9.28)|(+8.29)|(+10.01)|



|||||Sli|deBench-|VQA(BCNB)||||
|---|---|---|---|---|---|---|---|---|---|
|Method|Input|Tumor<br>Type|ER<br>Type|PR<br>Type|HER2<br>Type|HER2<br>Expression|Histological<br>Grading|Molecular<br>Subtype|Overall|
|Random|Tt|23.82|24.48|25.05|25.05|24.39|24.41|23.63|24.40|
|GPT-4|ex|0|0|0|0|0|0|0|0|
|GPT-4o||34.69|77.50|63.51|36.95|23.95|28.63|23.15|41.43|
|MedDr||45.46|23.53|25.99|71.81|22.73|30.28|15.49|33.67|
|LLaVA-Med|Patch|23.95|36.62|40.19|50.76|23.72|18.99|15.05|30.10|
|Quilt-LLaVA||77.14|68.58|42.63|58.17|23.18|18.23|19.82|44.43|
|GPT-4o||0|0|0|0|0|0|0|0|
|MedDr|Slid T|28.92|45.84|25.71|72.68|20.65|29.96|23.88|35.48|
|LLaVA-Med|e ()|0.01|0|0.01|0.02|0|0|0|0.01|
|Quilt-LLaVA||67.41|66.73|36.58|62.67|15.97|22.89|16.27|41.55|
|SlidCht|Slid|90.17|78.54|68.81|71.93|25.05|23.11|17.49|54.14|
|ea|e|(+44.71)|(+1.04)|(+5.3)|(-0.75)|(+0.66)|(-7.17)|(-6.39)|(+12.71)|





<!-- Start of picture text -->
A s B iPNSBe +<br>5 a $2<br>Sue : var<br>pathologicalDrafta summarystudy that of outlines the wholetheslide mainmage. findingsfrom the pathologicaPresenta c l earanalys s dep i ctionofof the wt h eole keyslide points image.from the j<br>‘The pathological examination ofthe right breast lumpectomy revealsThe pathologa d i calagnosisexaminationof glotlasioma, ofthe right Classitied frontalasrainWHO tumorGrace<br>. specimen revealed a diagnosis of mvasive ductal carcinoma, . IV. The tumor ischaracterized<br>8 Classified by a highly cellular gial<br>6 as Nottngham grade 2. The tumor was characterized 3 rnegplasm with marked pleomorphism, frequent mitetic figures,<br>3 by‘mitotic aigh nuclear grade, moderate tubule fommaton, anda 6 and areas of necrosis. immunohistochemicalstaning shows<br>FA by a noninvasivecount of 2 The component, imasive carcinclud in goma wduct l a s accompanied carcinoma in Po3 ongin,thatthe andtumor the cells MIB-1 areproliferationpositive forindexGFAP, is approx mate yindicat i ng g l ial40%,<br>‘ft (DCIS) and lobusar carcinoma in situ (LCIS), bom of which suggesting a high proliferative activity. The tumor exhiats<br>weretumorthe deepnighinvolvement, withthe nuciearmargn. Addtionaly,grade. Theclosest surgicalthe nipplemarginmarginswasbeingree were0.1of clear tumor. om trom of ‘metnylatonfocal areasIndicativeof of statusaggressivemicrovascularis negative,behavior.proiferation,indicatingThe MGMTnowhichmethyla ion, promois a feature t er<br>Examination of the sentine! lymph node tram the right axa which may have implications for the response to alkylating<br>‘Showed‘negativedetermined(greaterlymph nadethannofor to involvemeevidence carcinoma.2.cmbe pT2,butofnot metastaticpNO, pMr,The moreand pathological unktha i nn owndisease,dicatingt5 cm,distantstaging awiwit t umometastasis.no wasthe r  seg onal node i ze being pathologcalaggressive‘management.agentsinvolvlke temozolamidglioma,arefeaturesnat detailednecessitat ngare e .consmentThe n i stent te margin comprehensivefndings.wit status anda high-grade,Overall,clinicaliympnthe node<br>w Se<br>*<br>; 59<br>e<br>Delivera compact explanation of the pivotal findings in ; Describe succinctly the core discoveries in the pathology Fy<br>pathology a3 revealed by fhe whole sikie image. results from the whole slice mage. 8<br>__8G J Tepinate s a  assed oeiegennan EEThe pathological examination reveals a diagnosis ofcman<br>5o ietmers ommaneany stem ou fiaamacrerssan2 5BZ ‘centralMeletiower masswithoe oftamaximumung dimensionhe emer ischaracierzedby of4.5om ands a<br>3 located in the ung parenchyma, andis cassifed as pT'a. The o nesswi a.max ids<br>©) bronchial margins tree of tumorinvolvement, and there fs 10 3 oa ere are ee<br>'PNO,endence indicatingorymonovascutarno regional iymphmason.node Me metastasis. tumorsGasiedThe as a onSurg e al, margins aeROEtee of tuner nvovemen,ROENR) indeatg<br>resection margins are also free of tumor involvement. The i ier Of ene Ba AIC CeaOREY<br>panologcal(G2grade, staging is determined to be pT 12, pNO, pMe, witr a ‘nodeseeetrom themeawtac leftowermsowowentlobe and thewhol mediastnaleanpled regionnoses<br>regionalpor  lympheSindicating node metastasisa moderately dferentated and an unknown distanttumor with no ‘ beingandZ pO,negative reflectingfor tumor.a localized The  primarytumors stagedtumor wth as pT2a,no regionalpNO,<br>|ymph nade metastasis and no distant metastasis.<br><!-- End of picture text -->

Figure 7. Demonstration of SlideChat’s Capability in Whole-Slide Image Captioning. 



<!-- Start of picture text -->
° ear ° me<br>=3 image?How__tanud/onwouldthe youhistological classify thefeatures identifiedobservedurothelialinthe carcinoma who side 2 3 Howimage?benedwouldonthe youhistological classify the eatresidentifiedobservedurothelialinthe carcinoma whole slide 3<br>8&<br>H 3 3 <aieeaepty amend<br>Basod onthe provided mage, what grade would you assign to pact woteat crcnome<br>. ‘the bladder tumor? Describe the nuclear features typically associated with poorly 3<br>z (iterated<br><< uotetel carcinoma inte mage 3<br>3) ne nace tumor wou be assigned a 3<br>3 | gradefimo of Gradecharacerzed 3, indicatingby pooraterentaion a high-grade 53S3 | BarannideartocyeptaamieLferentated urothelialrote  carcinomaecinare<br>Based onthe tinsuncharacertcs and standard treatment cea corto ard prominentrato treguar neo.<br>protocols, which chemotherapy combination is frequently 3<br>‘sused Metcrraa as a neoadjuvantand Vinctsinetherapy for this condition? heorvedWhich lymph inthe nodesprovided correspondwhole lie to the mage? metastatic involvement -5<br>8 Cartanmud Gectabine @ Aahureter nodes B Right ureter odes<br>g, aaa Commoné ae nodes DL Nane ofthe mph nodes<br>& 3<br>ai “(emer eae<br>Examine Which cheractoritic a ypealy observed inthe issue '<br>tpecentheand provided choose wholthe predominant slide imag desis archit ctureofthe bladd e r Eitumor trchiA£8 rDieorpunced e enre?nance E<br>GaRbeoey of neoplasticcla rangement cas<br>ws —_—_—B.Sald ests win ivaveness gC Wesogonized muscle Bndeswinou ayia @<br>3  C-Glanduler Tubular Es . High mitotic activty<br>% ~[e.sova nets wth ivasveness 3& ~{.¢ werorgenedmise bundes winout apie<br>Cc Fw tas D<br>Vb. BS<br>Az %<br>esl te<br>Whatthe  indicates the presence of lymphovascular invasion in Ea Examinethe provided whole slide image of the loft axilla E<br>3 bladder tumor in this image? og& biopsy. What is the status of the lymph nodes?<br>&  < cxraeaca! ymprovesclrinvason 3<br>] 5) Sspreson snowig tat te tumor has 3 ~[Wometastatic carcinoma inthe nodes<br>& _) ‘Enatressiwed rough poneorvascar<br>‘What{mee grad bee d onthebeyond e  of celldifferentiation the badder provided wholis obsd e rvedkage?in the bladder 3 _ esedonSee the hintopticlogical Sr ee oecine fetes cheerved intheeee whale -<br>3 5 Based on te histopatoegcal<br>5 ‘The bladder tumor image is poory 3S) features, the prognosis is fair, 2s high-<br>©| 5) terentatec macetng aroma! cots 3) cadotumarsa aggresshe nd poce<br>soprecsvenessn'a gneve ordlsrgartaatonad ‘tngnerofcompton<br>Basedfatatoans6A SquemetePoptan earenameon this ct pemclivtewholecarnomaslide image,bekgrecelwhich term best describesreactant E;5 intheBasedSideAgr tagemageon the according ofthe histopathological otParty breasttothe ivasivegeste!features dimension?” ductal observedcarcinoma,in the what whole 3<br>. ©. Carcinoma<br>B —_D.Adencearcinomein stu @ g3 © prs prs<br>43<br>5~(ecocromna 3<br>Based onthe provided mage which specific anatomical : Examine the whole lide Image of grade 3 invasive ductal 5<br>fecumele inreded by fe nner? 3 (arcinme ui close medial tnd iatcor nang un 3<br>2 itusevars prone datermine the most appropriate next stop inte estment<br>_— 2G.eptnetimLana propia = SimpleCdrematerapy macecomy”""B.—B.AdeonlRedaton herapy<br>Dig  Lymphovescuar channels g3 supa excision @<br>a a.<br><!-- End of picture text -->

Figure 8. Demonstration of our SlideChat for answering various questions based on the WSI. 



<!-- Start of picture text -->
Based on the image, what is the —————--—> SlideChat °°<br>classification of the tumor?<br>A. Invasive ductal carcinoma<br>C. Other type<br>30Randomly 224x224 selectedpatchesNote+Vote cloter be) °<br>—> GPT-4o I cannot identify the type of tumor. |@<br>=_ —> MedDr B. Invasive lobular carcinoma ix}<br>tozaxtozg<br>‘Whole Slide image (WS!) Slide thumbnai TP TMRVAMed ~< connot identity the type of tumor)<br>Which tumor feature is in the _ Slidechat °°<br>provided whole slide image?<br>A. High nuclear pleomorphism<br>C.B. LackSignificant of lymphovasculartumor necrosisinvasion @ i] _ GPT-4orVote “TC Sgnitcanttumernewoss °<br>D. Extensive perineural invasionv— mealee —> MedDr+VoteLLaVA-Med xy<br>ittWt Wie 1 ‘fiwv 30 224x224 patches<br>a ve p he F I cannot determine it. x}<br>;<br>toosxioas AMOS °<br>‘Whole Slide image (WS!) Slide thumbnail<br>What is the overall structural ———— Silidechat ry<br>arrangement ofthe tumor cells<br>in the histological image?<br>*C. SinPapillary D.hanTrabecular Bi oooSaeed<br>we nm<br>Randomly selected +Vote A. Glandular Od<br>30 224x224 patches<br>—> GPT4o I cannot determine it. rx}<br>— > meso °<br>way > NAA °<br>Whole Slide Image (WS!) Slide thumbnail<br><!-- End of picture text -->

Figure 9. Comparing model outputs on SlideBench. 

