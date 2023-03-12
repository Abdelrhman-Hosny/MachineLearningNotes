# Model Deployment and Prediction Service

- **Deploy** is a loose term that generally mean making your model **running and accessible**.
- When the model is deployed, it **leaves** the **development environment**
  - It could be moved to a **staging** environment for **testing**
  - Or a **production** environment to be used by an **end-user**
  - This chapter focuses on **production environment**

- Production is a **spectrum**
  - For some teams, deployment is generating nice plots in organized notebooks to show to the business team.
  - For other teams, production means **keeping your models up and running** for **millions** of users a day.

- There is a statement that says **"Deployment is easy if you ignore the hard parts"**.
  - i.e. you could just wrap your prediction function with a FastAPI endpoint and push your model and its associated container to a cloud service.
  - The **hard part** is
    - Making your model **available** to **millions** of users with **low latency** and **99% uptime**
    - Setting up the infrastructure so that the **right person can be immediately notified when something goes wrong**, and have a relatively easy time figuring out what went wrong and seamlesly deploying the updates to fix it.

- In some companies, whoever writes the model deploys it, in others once the model is ready it is shipped to a deployment team.
  
## ML Deployment Myths

### You Only Deploy One or Two ML Models at a Time

- In academia, you focus on small problems that require one or two models to solve, but that is not the case in practice.
  - Your infrastructure should take into account how many models will be used.

- In practice, companies have a lot of models in productions at any given moment (depending on the size of the company), many of those models could be used in the same application.
****

### If We Don't Do Anything, Model Performance Remains the Same

- Software system face what is called **software rot** or **bit rot** where systems degrade over time.
  - ML models aren't immune to it

- On top of **software rot**, ML models suffer from **data distribution shifts**
  - When the data in the real world changes, so now the data your model sees isn't the same as the data it was trained on.
  - Therefore, an ML model tends to perform best right after training and degrade over time.

****

### You Won't Need to Update Your Models As Much

- People ask Chip "How often should I update my model?", that's the wrong question, the right question should be "How often *can* I update my model?"

- Since model performance decays over time, we want to update it as fast as possible.
- This is an area where MLOps should learn from existing DevOps best practices.
  - Where models are updated as frequently as possible

****

### Most ML Engineers Don't Need to Worry About Scale

- The concept of **scale** varies from application to application, but examples include systems that serve **hundreds** of queries **per second** or **millions** of users a month.
- People argue that only a small number of companies need to worry about it as there is only 1 Google, one FB (i.e. small number of companies that need such scale)
  
- According to Stack Overflow Dev survey in 2019, more than half of the respondents worked for a company of atleast 100 employees.
  - This means that there's a good change you'll need to worry about scale.

****

## Batch Prediction vs Online Prediction

- One decision that you'll have to make that will affect both your end user and developer is how it generates and serves its predictions to end users
  - **Online** prediction vs **Batch** prediction
- The terminologies surrounding batch and online prediction are still quite confusing due to the lack of standardized practices.
- The **three main modes** of **prediction** that Chip hopes I remember are :

  - **Batch** prediction which uses **only batch features**
  - **Online prediction**tat uses **only batch features** (e.g. precomputed embeddings)
  - **Online** prediction that uses **both** **batch** features and **streaming** features
    - aka **Streaming prediction**.
