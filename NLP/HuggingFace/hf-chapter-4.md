[toc]

# <u>**Chapter 4**</u>

## **<u>The Hugging Face Hub</u>**

- This is where all the models and datasets are hosted.
- Each of the models is hosted as a **Git repository** which allows versioning and reproducibility.
- Sharing a model on the Hub automatically **deploys** a hosted Inference API for that model. Anyone in the community is free to test it out on the **model's page**, with custom inputs and appropriate widgets.

****

<u>**N.B.**</u> before using a pretrained model, check how **it was trained**, on which **datasets**, its **limits and biases**. All of these should be found on the model card.

****

### **<u>Sharing pretrained models</u>**

- We can share our models on the hub
  - It's very similar to using Git.
- We manage the repository by
  - using the `push_to_hub` API
  - Using the `huggingface_hub` python library
  - using the web interface
- You can also use `git-lfs` where `lfs` stands for large file system
  - The syntax is very similar to git, which I already know, so I'll skip this.

****

### **<u>Building a model card</u>**

- Model cards are a way to describe things related to your model so that people can use your model easier
- Sections in model cards
  1. Model Description
     - Has basic details about the model, like the **architecture**, version, if it was introduced in a paper, if there is an original implementations, the author and general info about the model.
     - Any copyright should be attributed here.
     - **General** info about training procedures, parameters and important disclaimers can also be mentioned in this section
  2. Intended use & limitations
     - This includes the languages, fields and domains where it can be applied.
     - Can also mention if the model performs **suboptimally** at other areas.
  3. How to use ?
     - Should include code examples.
  4. Training data
     - Indicate the data that it was trained on and maybe a brief description of the dataset
  5. Training procedure
     - Describes all the relevant aspects of the training that are useful from a **reproducibility perspective**.
     - This includes any **preprocessing and postprocessing** done on the data
     - Also details such as # of epochs, batch size, learning rate ...etc
  6. Variable and metrics
     - Describe the metrics used in evaluation
     - Doing so makes it easy to compare your model to other models
  7. Evaluation results
     - Indication on how well the model performs on the evaluation dataset
     - If the model uses a **decision threshold**, either provide the decision threshold used in the evaluation, or provide details on evaluation at different thresholds for the intended uses.

****

- Model cards can have more fields but these are the main ones.

****

### **<u>Model card metadata</u>**

The **metadata** of the model on the Hub define which categories does that model belong to.

i.e. in `camembert-base` model card, you can see the following metadata in the model card header

```markdown
---
language: fr
license: mit
datasets:
- oscar
---
```

This data is parsed by the Hub, so that it identifies `camembert-base` as being a French  model with an MIT **license** trained on the oscar **dataset**.

This [link](https://raw.githubusercontent.com/huggingface/huggingface_hub/main/modelcard.md) shows examples for the metadata that can be defined.

****