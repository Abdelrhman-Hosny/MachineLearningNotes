[toc]

# Prompt Engineering

- **Prompt Engineering:** It is the process of **creating a prompting function** $f_{\text{prompt}}(x)$ that results in the most effective performance on the downstream task.
- **Prompt Template Engineering**: A human engineer of algorithm searches for the **best template** for each task and the model is expected to perform.

## Prompt Shape

- There are tow versions of prompts
    1. **cloze prompts**

        e.g.
        
        $f_{\text{prompt}}(x) = \text{[X] Overall, it was a [Z] movie}$
    Wheere $[X]$ is replaced by input and $[Z]$ is left as it is.
    2. **prefix prompts**
    e.g. "long test" TL;DR
    for summarization

## Manual Template Engineering

- Manually creating prompts to maximize performance

## Automated Template Learning

- Problems with manual template engineering
    1. Manual creation is an art that takes time and experience
    2. Even experienced prompt designers may fail to discover optimal prompts

